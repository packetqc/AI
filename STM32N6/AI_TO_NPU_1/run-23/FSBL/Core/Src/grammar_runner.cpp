/**
 * grammar_runner.cpp — see grammar_runner.hpp.
 * Faithful device port of GrammarRunner (class_model_grammar.py).
 */
#include "grammar_runner.hpp"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef NOCODE_DISPATCH
#include "nocode_dispatch.h"   /* generated (grammar,token)->compiled fn dispatch (host-model->NPU path) */
#endif

namespace llm {

/* ── small char helpers (no <cctype> locale baggage) ────────────────────── */
static inline bool is_digit(char c)  { return c >= '0' && c <= '9'; }
static inline bool is_alpha(char c)  { return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_'; }
static inline bool is_word(char c)   { return is_alpha(c) || is_digit(c); }
static inline bool is_op_char(char c){ return c=='+'||c=='-'||c=='*'||c=='/'; }
static inline bool is_group(char c)  { return c=='('||c==')'||c=='{'||c=='}'||
                                              c=='['||c==']'||c=='*'||c=='+'||c=='?'; }

static bool tok_eq(const Token& t, const char* s, int slen)
{
    return (int)t.len == slen && memcmp(t.val, s, slen) == 0;
}

/* ── ctor ───────────────────────────────────────────────────────────────── */

GrammarRunner::GrammarRunner(const char* grammar_name,
                             const Rule* playbook, int n_rules,
                             const TerminalLogger* logger,
                             QueryFn query_fn, void* user)
    : grammar_name_(grammar_name), playbook_(playbook), n_rules_(n_rules),
      logger_(logger), query_fn_(query_fn), user_(user),
      input_(nullptr), n_input_toks_(0), parse_last_pos_(0),
      node_count_(0), n_cached_(0), interaction_count_(0)
{
}

void GrammarRunner::log(Severity sev, const char* msg) const
{
    if (logger_) logger_->log(sev, "RUNNER", msg);
}

/* ── tokenizers ─────────────────────────────────────────────────────────── */

/* _tokenize_input: one digit, one operator (+ - mul div), a paren, or an
 * identifier — each a non-owning view into input_. */
int GrammarRunner::tokenize_input(const char* s)
{
    n_input_toks_ = 0;
    const char* p = s;
    while (*p && n_input_toks_ < GR_MAX_INPUT_TOKS) {
        char c = *p;
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { p++; continue; }
        Token& t = input_toks_[n_input_toks_];
        if (is_digit(c)) {                       /* single digit             */
            t.kind = Tok::Term; t.val = p; t.len = 1; p++;
        } else if (is_op_char(c) || c=='(' || c==')') {
            t.kind = Tok::Term; t.val = p; t.len = 1; p++;
        } else if (is_alpha(c)) {                /* identifier               */
            const char* q = p; while (*q && is_word(*q)) q++;
            t.kind = Tok::Term; t.val = p; t.len = (uint16_t)(q - p); p = q;
        } else { p++; continue; }                /* skip anything else       */
        n_input_toks_++;
    }
    return n_input_toks_;
}

/* _answer_tokens: <name> | "x" | 'x' | | | bare-name | grouping/quantifier. */
int GrammarRunner::tokenize_answer(const char* s, int slen, Token* out, int max)
{
    int n = 0; int i = 0;
    while (i < slen && n < max) {
        char c = s[i];
        if (c == ' ' || c == '\t') { i++; continue; }
        Token& t = out[n];
        if (c == '<') {                          /* <nonterminal>            */
            int j = i + 1; while (j < slen && s[j] != '>') j++;
            t.kind = Tok::Nt; t.val = s + i + 1; t.len = (uint16_t)(j - (i+1));
            i = (j < slen) ? j + 1 : j; n++;
        } else if (c == '"' || c == '\'') {      /* "terminal" / 'terminal'  */
            char q = c; int j = i + 1; while (j < slen && s[j] != q) j++;
            t.kind = Tok::Term; t.val = s + i + 1; t.len = (uint16_t)(j - (i+1));
            i = (j < slen) ? j + 1 : j; n++;
        } else if (c == '|') {                   /* alternative              */
            t.kind = Tok::Pipe; t.val = s + i; t.len = 1; i++; n++;
        } else if (is_alpha(c)) {                /* bare nonterminal name    */
            int j = i; while (j < slen && is_word(s[j])) j++;
            t.kind = Tok::Nt; t.val = s + i; t.len = (uint16_t)(j - i);
            i = j; n++;
        } else if (is_group(c)) {                /* grouping / quantifier op */
            t.kind = Tok::Op; t.val = s + i; t.len = 1; i++; n++;
        } else { i++; }                          /* skip                     */
    }
    return n;
}

/* ── rule resolution ────────────────────────────────────────────────────── */

const char* GrammarRunner::playbook_body(const char* name) const
{
    for (int i = 0; i < n_rules_; i++)
        if (strcmp(playbook_[i].name, name) == 0) return playbook_[i].body;
    return nullptr;
}

/* Resolve a (possibly non-NUL-terminated) rule name view to the canonical
 * playbook name pointer (NUL-terminated, pointer-stable for cache keys). */
static const char* canon_name(const Rule* pb, int n, const char* name, int len)
{
    for (int i = 0; i < n; i++)
        if ((int)strlen(pb[i].name) == len && memcmp(pb[i].name, name, len) == 0)
            return pb[i].name;
    return nullptr;
}

/*
 * _query_rule: query the NPU oracle once (for the visible interaction +
 * count), then let the authoritative playbook override the parse. Cached.
 */
GrammarRunner::AltSet* GrammarRunner::query_rule(const char* rule_name)
{
    for (int i = 0; i < n_cached_; i++)
        if (rule_cache_[i].name == rule_name) return &rule_cache_[i];
    if (n_cached_ >= GR_MAX_RULES) return nullptr;

    AltSet* as = &rule_cache_[n_cached_++];
    as->name = rule_name;
    as->n_alts = 0;

    /* scratch token stream for the chosen answer body */
    Token toks[GR_MAX_ALTS * GR_MAX_ALT_TOKS];
    int ntok = 0;

    /* 1) NPU oracle (mirrors host get_ollama_answer) — for the dialog + count */
    if (query_fn_) {
        char prompt[96];
        snprintf(prompt, sizeof(prompt), "%s %s", grammar_name_, rule_name);
        int alen = query_fn_(prompt, query_buf_, GR_QUERY_BUF, user_);
        interaction_count_++;
        if (alen > 0) {
            char shortbuf[72];
            int sl = alen < 60 ? alen : 60;
            memcpy(shortbuf, query_buf_, sl);
            shortbuf[sl] = '\0';
            char line[200];
            snprintf(line, sizeof(line), "[model #%d] %s -> %s%s",
                     interaction_count_, prompt, shortbuf, alen > 60 ? "..." : "");
            log(Severity::Info, line);
            ntok = tokenize_answer(query_buf_, alen, toks,
                                   GR_MAX_ALTS * GR_MAX_ALT_TOKS);
        }
    }

    /* 2) Playbook is authoritative: override when it has the rule. */
    const char* body = playbook_body(rule_name);
    if (body) {
        int blen = (int)strlen(body);
        int n = tokenize_answer(body, blen, toks, GR_MAX_ALTS * GR_MAX_ALT_TOKS);
        if (n > 0) ntok = n;   /* playbook wins */
    }

    /* split token stream on Pipe into alternatives (_split_alt_groups) */
    int a = 0, k = 0;
    for (int i = 0; i < ntok; i++) {
        if (toks[i].kind == Tok::Pipe) {
            if (k > 0 && a < GR_MAX_ALTS) { as->alts[a].ntok = k; a++; k = 0; }
            else k = 0;
        } else if (a < GR_MAX_ALTS && k < GR_MAX_ALT_TOKS) {
            as->alts[a].toks[k++] = toks[i];
        }
    }
    if (k > 0 && a < GR_MAX_ALTS) { as->alts[a].ntok = k; a++; }
    as->n_alts = a;
    return as;
}

/* ── arena ──────────────────────────────────────────────────────────────── */

int GrammarRunner::new_terminal(const char* v, uint16_t len)
{
    if (node_count_ >= GR_MAX_NODES) return -1;
    int id = node_count_++;
    Node& nd = node_pool_[id];
    nd.is_terminal = true; nd.term = v; nd.term_len = len;
    nd.rule = nullptr; nd.nchild = 0;
    return id;
}

int GrammarRunner::new_rule_node(const char* rule)
{
    if (node_count_ >= GR_MAX_NODES) return -1;
    int id = node_count_++;
    Node& nd = node_pool_[id];
    nd.is_terminal = false; nd.term = nullptr; nd.term_len = 0;
    nd.rule = rule; nd.nchild = 0;
    return id;
}

/* ── parser ─────────────────────────────────────────────────────────────── */

/* _try_alternative: match alt against input_toks_[pos:]. */
bool GrammarRunner::try_alternative(const Token* alt, int ntok, int pos,
                                    int* out_children, int* out_nchild, int* out_pos)
{
    int cur = pos; int nc = 0;
    for (int i = 0; i < ntok; i++) {
        const Token& tk = alt[i];
        if (tk.kind == Tok::Term) {
            if (cur < n_input_toks_ &&
                tok_eq(input_toks_[cur], tk.val, tk.len)) {
                if (nc >= GR_MAX_CHILDREN) return false;
                int t = new_terminal(tk.val, tk.len);
                if (t < 0) return false;
                out_children[nc++] = t; cur++;
            } else return false;
        } else if (tk.kind == Tok::Nt) {
            const char* cn = canon_name(playbook_, n_rules_, tk.val, tk.len);
            if (!cn) return false;
            int node = parse_rule(cn, cur);
            if (node < 0) return false;
            if (nc >= GR_MAX_CHILDREN) return false;
            out_children[nc++] = node;
            cur = parse_last_pos_;   /* parse_rule publishes its end pos */
        }
        /* Op tokens (EBNF grouping/quantifier) are accepted and skipped. */
    }
    *out_nchild = nc; *out_pos = cur; return true;
}

/* _parse_rule with left-recursion base/extend handling. Publishes end pos in
 * parse_last_pos_. Returns node index or -1. */
int GrammarRunner::parse_rule(const char* rule_name, int pos)
{
    parse_last_pos_ = pos;
    if (pos >= n_input_toks_) return -1;

    AltSet* as = query_rule(rule_name);
    if (!as || as->n_alts == 0) return -1;

    int original_pos = pos;

    /* base alts (not left-recursive) vs extend alts (start with self). */
    int node = -1;
    for (int ai = 0; ai < as->n_alts; ai++) {
        const AltSet::Alt& alt = as->alts[ai];
        bool lrec = alt.ntok > 0 && alt.toks[0].kind == Tok::Nt &&
                    tok_eq(alt.toks[0], rule_name, (int)strlen(rule_name));
        if (lrec) continue;
        int kids[GR_MAX_CHILDREN], nk = 0, np = pos;
        if (try_alternative(alt.toks, alt.ntok, pos, kids, &nk, &np)) {
            node = new_rule_node(rule_name);
            if (node < 0) return -1;
            for (int j = 0; j < nk; j++) node_pool_[node].children[j] = kids[j];
            node_pool_[node].nchild = nk;
            pos = np;
            break;
        }
    }
    if (node < 0) { parse_last_pos_ = original_pos; return -1; }

    /* iteratively extend with left-recursive alts (rest = alt[1:]). */
    bool extended = true;
    while (pos < n_input_toks_ && extended) {
        extended = false;
        for (int ai = 0; ai < as->n_alts; ai++) {
            const AltSet::Alt& alt = as->alts[ai];
            bool lrec = alt.ntok > 0 && alt.toks[0].kind == Tok::Nt &&
                        tok_eq(alt.toks[0], rule_name, (int)strlen(rule_name));
            if (!lrec) continue;
            int kids[GR_MAX_CHILDREN], nk = 0, np = pos;
            if (try_alternative(alt.toks + 1, alt.ntok - 1, pos, kids, &nk, &np)) {
                int parent = new_rule_node(rule_name);
                if (parent < 0) return -1;
                node_pool_[parent].children[0] = node;
                int pc = 1;
                for (int j = 0; j < nk && pc < GR_MAX_CHILDREN; j++)
                    node_pool_[parent].children[pc++] = kids[j];
                node_pool_[parent].nchild = pc;
                node = parent; pos = np; extended = true;
                break;
            }
        }
    }
    parse_last_pos_ = pos;
    return node;
}

int GrammarRunner::parse(const char* input, const char* start_rule)
{
    node_count_ = 0; n_cached_ = 0; interaction_count_ = 0;
    input_ = input;
    if (tokenize_input(input) == 0) { log(Severity::Warning, "Empty input."); return -1; }

    int node = parse_rule(start_rule, 0);
    if (node < 0) { log(Severity::Error, "Parse failed."); return -1; }
    if (parse_last_pos_ < n_input_toks_)
        log(Severity::Warning, "Partial parse — trailing tokens unconsumed.");
    return node;
}

/* ── evaluator ──────────────────────────────────────────────────────────── */

long GrammarRunner::evaluate(int node, bool* ok)
{
    /* Returns the numeric value; *ok=false if the subtree is non-numeric. */
    if (node < 0) { *ok = false; return 0; }
    const Node& nd = node_pool_[node];

    if (nd.is_terminal) {
        /* single-char/identifier terminal → int if all digits, else not-num */
        bool allnum = nd.term_len > 0;
        for (int i = 0; i < nd.term_len; i++) if (!is_digit(nd.term[i])) allnum = false;
        if (allnum) { long v = 0; for (int i=0;i<nd.term_len;i++) v = v*10 + (nd.term[i]-'0'); *ok = true; return v; }
        *ok = false; return 0;   /* e.g. operator terminal "+" */
    }

    /* "number" rule: concatenate numeric children as a decimal integer. */
    bool is_number_rule = nd.rule && strcmp(nd.rule, "number") == 0;
    if (is_number_rule) {
        char buf[24]; int bp = 0;
        for (int i = 0; i < nd.nchild; i++) {
            bool cok; long cv = evaluate(nd.children[i], &cok);
            if (cok) bp += snprintf(buf+bp, (int)sizeof(buf)-bp, "%ld", cv);
        }
        if (bp > 0) { buf[bp] = '\0'; *ok = true; return atol(buf); }
    }

    /* general arithmetic: collect numeric operands and an operator child. */
    long nums[GR_MAX_CHILDREN]; int nn = 0; char op = 0;
    for (int i = 0; i < nd.nchild; i++) {
        int c = nd.children[i];
        const Node& cn = node_pool_[c];
        if (cn.is_terminal && cn.term_len == 1 && is_op_char(cn.term[0])) {
            op = cn.term[0];
        } else {
            bool cok; long cv = evaluate(c, &cok);
            if (cok && nn < GR_MAX_CHILDREN) nums[nn++] = cv;
        }
    }
    if (nn == 2 && op) {
        long a = nums[0], b = nums[1];
#ifdef NOCODE_DISPATCH
        /* nocode dispatch: the op's logic is a COMPILED function selected by (grammar,token) — the
         * device equivalent of the host model emitting the body; the runner stays grammar-agnostic.
         * Falls through to the hardcoded switch if the token is not in the dispatch table. */
        const char* tok = (op=='+')?"op_add":(op=='-')?"op_sub":(op=='*')?"op_mul":(op=='/')?"op_div":0;
        if (tok) {
            NcFn fn = nc_resolve(grammar_name_, tok);
            if (fn) {
                if (op=='/' && b == 0) { log(Severity::Error, "Division by zero."); *ok = false; return 0; }
                NcCtx c; memset(&c, 0, sizeof c); c.a = a; c.b = b;
                *ok = true; return fn(&c);
            }
        }
#endif
        switch (op) {
            case '+': *ok = true; return a + b;
            case '-': *ok = true; return a - b;
            case '*': *ok = true; return a * b;
            case '/': if (b != 0) { *ok = true; return a / b; }
                      log(Severity::Error, "Division by zero."); *ok = false; return 0;
        }
    }
    if (nn == 1) { *ok = true; return nums[0]; }
    *ok = false; return 0;
}

void GrammarRunner::tree_str(int node, char* buf, int n) const
{
    if (n <= 1) return;
    if (node < 0) { snprintf(buf, n, "?"); return; }
    const Node& nd = node_pool_[node];
    if (nd.is_terminal) {
        int l = nd.term_len < n-1 ? nd.term_len : n-1;
        memcpy(buf, nd.term, l); buf[l] = '\0'; return;
    }
    int p = snprintf(buf, n, "%s(", nd.rule ? nd.rule : "?");
    for (int i = 0; i < nd.nchild && p < n-2; i++) {
        if (i) { if (p < n-1) buf[p++] = ' '; buf[p] = '\0'; }
        tree_str(nd.children[i], buf + p, n - p);
        p += (int)strlen(buf + p);
    }
    if (p < n-1) { buf[p++] = ')'; buf[p] = '\0'; }
}

/* ── public: run ────────────────────────────────────────────────────────── */

bool GrammarRunner::run(const char* input, const char* start_rule, long* out_result)
{
    char m[96];
    snprintf(m, sizeof(m), "Parsing '%s' as <%s> via '%s' grammar...",
             input, start_rule, grammar_name_);
    log(Severity::Info, m);

    int node = parse(input, start_rule);
    if (node < 0) return false;

    char tree[160]; tree_str(node, tree, sizeof(tree));
    char tm[200]; snprintf(tm, sizeof(tm), "Parse tree : %s", tree);
    log(Severity::Ok, tm);

    bool ok; long result = evaluate(node, &ok);
    if (ok) {
        char rm[96];
        snprintf(rm, sizeof(rm), "Result     : %ld  (%d model interaction(s))",
                 result, interaction_count_);
        log(Severity::Ok, rm);
        if (out_result) *out_result = result;
    } else {
        log(Severity::Warning, "Non-numeric grammar — see parse tree above.");
    }
    return ok;
}

/* ── public: probe (playbook only, no NPU) ──────────────────────────────── */

bool GrammarRunner::probe(const char* input, const char* start_rule)
{
    QueryFn saved = query_fn_; query_fn_ = nullptr;   /* force playbook-only */
    const TerminalLogger* sl = logger_; logger_ = nullptr; /* silent */
    int node = parse(input, start_rule);
    bool full = (node >= 0) && (parse_last_pos_ == n_input_toks_);
    query_fn_ = saved; logger_ = sl;
    return full;
}

/* ── public: execute-mode (procedure grammars / nocode) ─────────────────── */

void GrammarRunner::execute_rule(const char* rule_name, int depth,
                                 CommandFn cmd, void* cmd_user)
{
    if (depth > 32) return;
    AltSet* as = query_rule(rule_name);
    if (!as || as->n_alts == 0) return;
    /* take the first non-left-recursive alternative — the intended procedure */
    for (int ai = 0; ai < as->n_alts; ai++) {
        const AltSet::Alt& alt = as->alts[ai];
        bool lrec = alt.ntok > 0 && alt.toks[0].kind == Tok::Nt &&
                    tok_eq(alt.toks[0], rule_name, (int)strlen(rule_name));
        if (lrec) continue;
        for (int i = 0; i < alt.ntok; i++) {
            const Token& tk = alt.toks[i];
            if (tk.kind == Tok::Nt) {
                const char* cn = canon_name(playbook_, n_rules_, tk.val, tk.len);
                if (cn) execute_rule(cn, depth + 1, cmd, cmd_user);
            } else if (tk.kind == Tok::Term && cmd) {
                cmd(tk.val, tk.len, cmd_user);   /* fire command hook */
            }
        }
        return;
    }
}

void GrammarRunner::execute(const char* start_rule, CommandFn cmd, void* cmd_user)
{
    n_cached_ = 0; interaction_count_ = 0;
    if (!start_rule && n_rules_ > 0) start_rule = playbook_[0].name;
    if (!start_rule) { log(Severity::Error, "No start rule — cannot execute."); return; }
    char m[96];
    snprintf(m, sizeof(m), "Executing '%s' via <%s>...", grammar_name_, start_rule);
    log(Severity::Info, m);
    execute_rule(start_rule, 0, cmd, cmd_user);
    snprintf(m, sizeof(m), "Done — %d model interaction(s).", interaction_count_);
    log(Severity::Ok, m);
}

} /* namespace llm */
