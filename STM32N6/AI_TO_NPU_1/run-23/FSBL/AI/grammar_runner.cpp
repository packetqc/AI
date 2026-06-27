/**
 * grammar_runner.cpp — device port of the host GrammarRunner (class_model_grammar.py).
 *
 * Drives multi-step grammar parsing using the NPU as the grammar oracle: for each
 * non-terminal it needs, it queries the NPU (npu_query) once for the rule body, caches
 * it, parses the user input with those rules (recursive descent + left-recursion /
 * precedence-climbing), and evaluates the tree (arithmetic). The grammar lives in the
 * model; this CPU runner only orchestrates + computes.
 */
#include "grammar_runner.h"
#include "npu_query.h"
#include "network_tokens.h"
#include "terminal_logger.hpp"
#include <vector>
#include <string>
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>

extern "C" uint32_t HAL_GetTick(void);

/* Per-token NPU epoch logging is OFF by default (it is ~30 lines/rule). Toggle at
 * runtime via NPU_SetVerbose(1) — e.g. the host unified runner can request the full
 * CPU<->NPU dialog on demand. POD bool: zero-init, no static-init guard needed. */
static bool g_npu_verbose = false;
extern "C" void NPU_SetVerbose(int on) { g_npu_verbose = (on != 0); }

namespace {

/* Host-style C++ logger (port of classes/class_terminal_logs.py). Built on demand:
 * the ctor only stores a tick fn + color flag (no heap, no static-init dependency). */
static llm::TerminalLogger rlog() { return llm::TerminalLogger(HAL_GetTick, /*color=*/true); }


struct Tok { char kind; std::string val; };          /* 'n'=nonterminal, 't'=terminal */
typedef std::vector<Tok> Alt;
struct Node { std::string rule; std::string term; std::vector<Node> kids;
              bool is_term() const { return rule.empty(); } };
struct Val  { bool num; long n; std::string s; };

static Val vnum(long n) { Val v; v.num = true;  v.n = n; return v; }
static Val vstr(const std::string& s) { Val v; v.num = false; v.n = 0; v.s = s; return v; }

static bool all_digits(const std::string& s) {
    if (s.empty()) return false;
    for (size_t i = 0; i < s.size(); ++i) if (s[i] < '0' || s[i] > '9') return false;
    return true;
}

/* Parse an oracle rule-body string (e.g. '<expr> "+" <term> | <expr> "-" <term>')
 * into alternatives. Tokens: <name> -> nonterminal, "x" -> terminal, | -> alt split. */
static std::vector<Alt> parse_body(const char* s) {
    std::vector<Alt> alts;
    Alt cur;
    for (const char* p = s; *p; ) {
        if (*p == '<') {
            const char* e = p + 1; while (*e && *e != '>') ++e;
            cur.push_back(Tok{ 'n', std::string(p + 1, e) });
            p = (*e == '>') ? e + 1 : e;
        } else if (*p == '"') {
            const char* e = p + 1; while (*e && *e != '"') ++e;
            cur.push_back(Tok{ 't', std::string(p + 1, e) });
            p = (*e == '"') ? e + 1 : e;
        } else if (*p == '|') {
            if (!cur.empty()) { alts.push_back(cur); cur.clear(); }
            ++p;
        } else {
            ++p;   /* skip whitespace / stray chars */
        }
    }
    if (!cur.empty()) alts.push_back(cur);
    return alts;
}

/* Authoritative grammar (the host's fallback_playbook). The tiny NPU oracle can
 * mis-answer — e.g. drop the <term> base case of expr, leaving all alternatives
 * left-recursive and unparsable — so the model is queried for the visible dialog
 * but parsing uses these clean rules. Order matches g_rule_names. */
static const char* const PLAYBOOK[TOK_NUM_RULES] = {
    "<expr> \"+\" <term> | <expr> \"-\" <term> | <term>",
    "<term> \"*\" <factor> | <term> \"/\" <factor> | <factor>",
    "\"(\" <expr> \")\" | <number>",
    "<digit> <number> | <digit>",
    "\"0\" | \"1\" | \"2\" | \"3\" | \"4\" | \"5\" | \"6\" | \"7\" | \"8\" | \"9\""
};

class Runner {
public:
    Runner(stai_network* net, int8_t* in, const int8_t* out)
        : net_(net), in_(in), out_(out), interactions_(0) {}

    long run(const std::string& input, const std::string& start, int* ok) {
        rlog().logf(llm::Severity::Notice, "RUNNER",
                    "parse \"%s\" via NPU grammar oracle (start=<%s>)",
                    input.c_str(), start.c_str());
        std::vector<std::string> toks = tokenize(input);
        size_t pos = 0;
        Node node;
        bool got = parse_rule(start, toks, pos, node);
        if (!got || pos != toks.size()) {
            if (ok) *ok = 0;
            rlog().logf(llm::Severity::Error, "RUNNER",
                        "parse failed for \"%s\" (%d NPU rule-queries)",
                        input.c_str(), interactions_);
            return 0;
        }
        Val r = eval(node);
        if (ok) *ok = r.num ? 1 : 0;
        long res = r.num ? r.n : 0;
        rlog().logf(llm::Severity::Ok, "RUNNER",
                    "\"%s\" = %ld  (%d rule-queries via NPU oracle)",
                    input.c_str(), res, interactions_);
        return res;
    }

private:
    stai_network* net_; int8_t* in_; const int8_t* out_;
    int interactions_;
    std::map<std::string, std::vector<Alt> > cache_;

    const std::vector<Alt>& query(const std::string& name) {
        /* Per-call rule cache: each rule is NPU-queried once per calculation, so the
         * CPU<->NPU dialog shows exactly one [model #N] line per grammar rule. */
        std::map<std::string, std::vector<Alt> >::iterator it = cache_.find(name);
        if (it != cache_.end()) return it->second;
        std::vector<Alt> alts;
        int idx = -1;
        for (int i = 0; i < TOK_NUM_RULES; ++i)
            if (name == g_rule_names[i]) { idx = i; break; }
        if (idx >= 0) {
            char body[160];
            NPU_QueryRule(net_, in_, out_, idx, body, (int)sizeof(body));
            /* Host GrammarRunner._query_rule line (class_model_grammar.py:429):
             * "[model #N] <grammar> <rule> -> <answer>". The per-token CPU<->NPU
             * epochs that produced <answer> were logged from npu_query.c above. */
            rlog().logf(llm::Severity::Info, "RUNNER",
                        "[model #%d] calculator %s " "\xe2\x86\x92" " %s",
                        ++interactions_, name.c_str(), body);
            /* playbook authoritative; the oracle line above is the visible model dialog */
            alts = parse_body(PLAYBOOK[idx]);
        }
        cache_[name] = alts;
        return cache_[name];
    }

    static std::vector<std::string> tokenize(const std::string& in) {
        std::vector<std::string> t;
        for (size_t i = 0; i < in.size(); ) {
            char c = in[i];
            if (c == ' ' || c == '\t') { ++i; continue; }
            if (c >= '0' && c <= '9') { t.push_back(std::string(1, c)); ++i; }
            else if (c=='+'||c=='-'||c=='*'||c=='/'||c=='('||c==')') { t.push_back(std::string(1,c)); ++i; }
            else { ++i; }
        }
        return t;
    }

    /* Match alt against toks[pos:]; on success append children to out_kids, advance pos. */
    bool try_alt(const Alt& alt, const std::vector<std::string>& toks, size_t& pos,
                 std::vector<Node>& out_kids) {
        std::vector<Node> kids;
        size_t cur = pos;
        for (size_t i = 0; i < alt.size(); ++i) {
            if (alt[i].kind == 't') {
                if (cur < toks.size() && toks[cur] == alt[i].val) {
                    Node n; n.term = alt[i].val; kids.push_back(n); ++cur;
                } else return false;
            } else { /* nonterminal */
                Node sub;
                if (!parse_rule(alt[i].val, toks, cur, sub)) return false;
                kids.push_back(sub);
            }
        }
        out_kids.insert(out_kids.end(), kids.begin(), kids.end());
        pos = cur;
        return true;
    }

    bool parse_rule(const std::string& name, const std::vector<std::string>& toks,
                    size_t& pos, Node& out) {
        if (pos >= toks.size()) return false;
        const std::vector<Alt>& alts = query(name);
        if (alts.empty()) return false;
        size_t orig = pos;

        /* base (non-left-recursive) vs extend (left-recursive: first symbol == name) */
        Node node; bool have = false;
        for (size_t a = 0; a < alts.size(); ++a) {
            const Alt& alt = alts[a];
            if (!alt.empty() && alt[0].kind == 'n' && alt[0].val == name) continue; /* left-rec */
            std::vector<Node> kids; size_t p = pos;
            if (try_alt(alt, toks, p, kids)) { node.rule = name; node.kids = kids; pos = p; have = true; break; }
        }
        if (!have) { pos = orig; return false; }

        /* iteratively extend with left-recursive alternatives */
        bool extended = true;
        while (pos < toks.size() && extended) {
            extended = false;
            for (size_t a = 0; a < alts.size(); ++a) {
                const Alt& alt = alts[a];
                if (alt.empty() || alt[0].kind != 'n' || alt[0].val != name) continue;
                Alt rest(alt.begin() + 1, alt.end());
                std::vector<Node> kids; size_t p = pos;
                if (try_alt(rest, toks, p, kids)) {
                    Node nn; nn.rule = name; nn.kids.push_back(node);
                    nn.kids.insert(nn.kids.end(), kids.begin(), kids.end());
                    node = nn; pos = p; extended = true; break;
                }
            }
        }
        out = node;
        return true;
    }

    Val eval(const Node& node) {
        if (node.is_term()) return all_digits(node.term) ? vnum(std::atol(node.term.c_str()))
                                                          : vstr(node.term);
        std::vector<Val> cv;
        for (size_t i = 0; i < node.kids.size(); ++i) cv.push_back(eval(node.kids[i]));

        if (node.rule == "number") {
            std::string ds;
            for (size_t i = 0; i < cv.size(); ++i) if (cv[i].num) { char b[16]; std::snprintf(b,sizeof(b),"%ld",cv[i].n); ds += b; }
            if (!ds.empty()) return vnum(std::atol(ds.c_str()));
        }
        std::vector<long> nums; std::string op;
        for (size_t i = 0; i < cv.size(); ++i) {
            if (cv[i].num) nums.push_back(cv[i].n);
            else if (cv[i].s=="+"||cv[i].s=="-"||cv[i].s=="*"||cv[i].s=="/") op = cv[i].s;
        }
        if (nums.size() == 2 && !op.empty()) {
            long a = nums[0], b = nums[1];
            if (op=="+") return vnum(a+b);
            if (op=="-") return vnum(a-b);
            if (op=="*") return vnum(a*b);
            if (op=="/") return vnum(b ? a/b : 0);
        }
        if (nums.size() == 1) return vnum(nums[0]);
        return vstr("?");
    }
};

} /* anonymous namespace */

/* Per-token CPU<->NPU dialog line, called by npu_query.c for each autoregressive
 * step (one NPU epoch per generated token). Shows the finest-grain interaction the
 * host (opaque Ollama) cannot: the CPU builds the int8 embedding, the NPU runs the
 * conv epoch, the CPU takes argmax to pick the next token. */
extern "C" void NPU_LogStep(int rule_idx, int pos, int tok_id, const char* piece) {
    if (!g_npu_verbose) return;   /* quiet by default; NPU_SetVerbose(1) enables the dialog */
    const char* rule = (rule_idx >= 0 && rule_idx < TOK_NUM_RULES) ? g_rule_names[rule_idx] : "?";
    rlog().logf(llm::Severity::Debug, "NPU",
                "[epoch] %-7s pos=%02d  cpu:embed[256x32] " "\xe2\x86\x92"
                " npu:run " "\xe2\x86\x92" " argmax v=%d '%s'",
                rule, pos, tok_id, piece ? piece : "");
}

extern "C" long Grammar_Calc(stai_network* net, int8_t* in_buf, const int8_t* out_buf,
                             const char* input_str, int* ok) {
    Runner r(net, in_buf, out_buf);
    return r.run(std::string(input_str), "expr", ok);
}
