/**
 * grammar_runner.hpp
 *
 * Device C++ port of GrammarRunner (classes/class_model_grammar.py).
 * Grammar-neutral recursive-descent engine driven by the NPU as a grammar
 * oracle — the device equivalent of the host querying the Ollama model.
 *
 *   parse()   : tokenize input → build a parse tree for any BNF grammar
 *   evaluate(): arithmetic for expression grammars, string repr otherwise
 *   run()     : parse → evaluate → log result (the interactive entry point)
 *   execute() : top-down walk of a procedure grammar, firing command hooks
 *   probe()   : does input fully parse against the playbook? (no NPU calls)
 *
 * Embedded subset: no exceptions/RTTI/heap. All state lives in fixed-capacity
 * pools sized by the constants below. Tokens are non-owning views into the
 * (persistent) playbook bodies and the caller's input buffer — nothing copied.
 *
 * "nocode" concept: the grammar (playbook) defines behaviour; the model is the
 * oracle. Same engine runs the calculator and larger procedure grammars.
 */
#ifndef GRAMMAR_RUNNER_HPP
#define GRAMMAR_RUNNER_HPP

#include <stdint.h>
#include "terminal_logger.hpp"

namespace llm {

/* ── capacities (tune per grammar complexity; sized to fit AXISRAM2 RAM) ─── */
enum {
    GR_MAX_RULES      = 8,    /* distinct rules cached (calculator has 5)    */
    GR_MAX_ALTS       = 12,   /* alternatives per rule (digit has 10)        */
    GR_MAX_ALT_TOKS   = 6,    /* tokens per alternative (e.g. expr "+" term) */
    GR_MAX_INPUT_TOKS = 64,   /* tokens in one user input line               */
    GR_MAX_NODES      = 128,  /* parse-tree node arena                       */
    GR_MAX_CHILDREN   = 8,    /* children per tree node                      */
    GR_QUERY_BUF      = 160   /* NPU answer scratch                          */
};

/* Token kinds — mirror the (kind, value) tuples in the Python tokenizers. */
enum class Tok : uint8_t { Nt, Term, Pipe, Op };

struct Token {
    Tok         kind;
    const char* val;   /* non-owning view (playbook body or input buffer)    */
    uint16_t    len;
};

/* One playbook rule: name → stored answer body, e.g.
 *   name="expr", body="expr \"+\" term | expr \"-\" term | term"           */
struct Rule { const char* name; const char* body; };

/* Parse-tree node (arena-allocated by index). */
struct Node {
    bool        is_terminal;
    const char* term; uint16_t term_len;   /* when is_terminal               */
    const char* rule;                       /* when !is_terminal             */
    int         children[GR_MAX_CHILDREN];
    int         nchild;
};

/*
 * NPU oracle: write the model's answer for `prompt` into out (≤out_max-1 +NUL).
 * Returns answer length, or <0 on error. NULL fn ⇒ probe mode (playbook only).
 * `user` is an opaque context passed back unchanged.
 */
typedef int (*QueryFn)(const char* prompt, char* out, int out_max, void* user);

/* Execute-mode hook: a terminal token matched a registered command. */
typedef void (*CommandFn)(const char* token, uint16_t len, void* user);

class GrammarRunner {
public:
    /* playbook: array of `n_rules` rules (authoritative grammar bodies).
     * logger:   may be null (silent). query_fn: may be null (probe mode).   */
    GrammarRunner(const char* grammar_name,
                  const Rule* playbook, int n_rules,
                  const TerminalLogger* logger = nullptr,
                  QueryFn query_fn = nullptr, void* user = nullptr);

    /* Tokenize + recursive-descent parse from start_rule.
     * Returns node index ≥0, or -1 on failure. Resets caches each call.     */
    int parse(const char* input, const char* start_rule);

    /* Full pipeline: parse → evaluate → log "Result: ...". out_result set on
     * success. Returns true if a value was produced.                        */
    bool run(const char* input, const char* start_rule, long* out_result);

    /* Procedure-grammar walk (no input): query each rule, fire command hooks. */
    void execute(const char* start_rule, CommandFn cmd, void* cmd_user);

    /* True if input fully parses against the playbook (no NPU interaction).  */
    bool probe(const char* input, const char* start_rule);

    int  interaction_count() const { return interaction_count_; }

    /* Toggle the NPU oracle at runtime. null = pure-CPU playbook mode (minimal,
     * instant — the embedded grammar is authoritative); non-null = also query
     * the model per rule for the visible dialog. */
    void set_query(QueryFn q, void* user) { query_fn_ = q; user_ = user; }

private:
    /* ── tokenizers ── */
    int  tokenize_input(const char* s);                 /* → input_toks_[]   */
    int  tokenize_answer(const char* s, int slen,
                         Token* out, int max);           /* playbook/answer  */

    /* ── rule resolution (NPU oracle + authoritative playbook) ── */
    const char* playbook_body(const char* name) const;
    struct AltSet; AltSet* query_rule(const char* rule_name);

    /* ── parser ── */
    int  parse_rule(const char* rule_name, int pos);
    bool try_alternative(const Token* alt, int ntok, int pos,
                         int* out_children, int* out_nchild, int* out_pos);

    /* ── evaluator / display ── */
    long evaluate(int node, bool* ok);
    void tree_str(int node, char* buf, int n) const;

    /* ── execute-mode ── */
    void execute_rule(const char* rule_name, int depth,
                      CommandFn cmd, void* cmd_user);

    /* ── arena ── */
    int  new_terminal(const char* v, uint16_t len);
    int  new_rule_node(const char* rule);

    void log(Severity sev, const char* msg) const;

    /* config */
    const char*           grammar_name_;
    const Rule*           playbook_;
    int                   n_rules_;
    const TerminalLogger* logger_;
    QueryFn               query_fn_;
    void*                 user_;

    /* input token stream (views into caller buffer, valid during parse) */
    const char* input_;
    Token       input_toks_[GR_MAX_INPUT_TOKS];
    int         n_input_toks_;
    int         parse_last_pos_;   /* end position published by parse_rule */

    /* node arena */
    Node node_pool_[GR_MAX_NODES];
    int  node_count_;

    /* rule cache */
    struct AltSet {
        const char* name;
        int  n_alts;
        struct Alt { Token toks[GR_MAX_ALT_TOKS]; int ntok; } alts[GR_MAX_ALTS];
    };
    AltSet rule_cache_[GR_MAX_RULES];
    int    n_cached_;

    char query_buf_[GR_QUERY_BUF];
    int  interaction_count_;
};

} /* namespace llm */

#endif /* GRAMMAR_RUNNER_HPP */
