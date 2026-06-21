import os
import re

from class_model_assets import ModelAssets


class ModelGrammar:
    """Parses BNF/EBNF grammar files and augments a model via ModelAssets.

    A grammar file is converted to two training signals:
      - A playbook subtree  {grammar_name: {rule_name: "alt1 | alt2 | ..."}}
        trained as (prompt, answer) pairs for exact structural recall.
        e.g. "calculator expr" -> 'expr "+" term | expr "-" term | term'
      - A prose paragraph describing each rule, appended to knowledge_texts
        for semantic recall ("what can an expression be?").

    Both feed the same build_pipeline() that JSON playbooks and markdown use, so
    grammar rules are fully persistent (saved in the state file), survive restart,
    and require minimal code in model_create_hf_cl.py.

    Startup seeding (via seed_from_file, runs on fresh start):
        INIT_KNOWLEDGE_FILES = ["playbook_model_calculator.txt"]
        # seed_from_file calls ModelGrammar.load_file() automatically for BNF files

    In-flight loading (interactive /grammar command):
        ModelGrammar.augment(assets, "playbook_model_calculator.txt")
    """

    # Tokenizer for the right-hand side of a rule definition.
    _TOKEN_RE = re.compile(
        r'<([^>]+)>'            # group 1 – nonterminal  <name>
        r'|"([^"]*)"'           # group 2 – terminal     "x"
        r"|'([^']*)'"           # group 3 – terminal     'x'
        r'|(\|)'                # group 4 – alternative  |
        r'|([{}()\[\]])'        # group 5 – EBNF grouping operator
        r'|([*+?])'             # group 6 – EBNF quantifier
    )

    # ------------------------------------------------------------------ parser

    @staticmethod
    def _is_rule_opener(line):
        """Return True if line starts a BNF (::=) or EBNF (<name> = ...) rule."""
        if not line.startswith('<'):
            return False
        if '::=' in line:
            return True
        return bool(re.match(r'^<[^>]+>\s*=\s', line))

    @staticmethod
    def _split_rule_line(line):
        """Split a rule-opener into (rule_name, rhs_text)."""
        if '::=' in line:
            idx = line.index('::=')
            name_raw = line[:idx]
            rhs = line[idx + 3:]
        else:
            # EBNF: find the = that follows the closing >
            close = line.index('>')
            eq = line.index('=', close)
            name_raw = line[:eq]
            rhs = line[eq + 1:]
        return name_raw.strip().strip('<>').strip(), rhs.strip()

    @staticmethod
    def _tokenize_rhs(rhs):
        """Return (kind, value) tuples for every token in the RHS string."""
        tokens = []
        for m in ModelGrammar._TOKEN_RE.finditer(rhs):
            g = m.lastindex
            if g == 1:
                tokens.append(('nt',   m.group(1).strip()))
            elif g in (2, 3):
                tokens.append(('term', m.group(g)))
            elif g == 4:
                tokens.append(('pipe', '|'))
            elif g in (5, 6):
                tokens.append(('op',   m.group(g)))
        return tokens

    @staticmethod
    def _render_alt(tokens):
        """Render one alternative (a list of tokens) as a human-readable string."""
        parts = []
        for kind, value in tokens:
            if kind == 'nt':
                parts.append(value)
            elif kind == 'term':
                parts.append('"' + value + '"' if value else '""')
            else:
                parts.append(value)
        return ' '.join(parts).strip()

    @classmethod
    def _split_into_alts(cls, tokens):
        """Split a flat token list on | delimiters; return rendered alternative strings."""
        groups, current = [], []
        for tok in tokens:
            if tok == ('pipe', '|'):
                rendered = cls._render_alt(current)
                if rendered:
                    groups.append(rendered)
                current = []
            else:
                current.append(tok)
        rendered = cls._render_alt(current)
        if rendered:
            groups.append(rendered)
        return groups

    @classmethod
    def parse(cls, text):
        """Parse BNF/EBNF text. Returns {rule_name: ["alt1", "alt2", ...]} in order."""
        rules = {}
        current_name = None
        rhs_chunks = []

        def _flush():
            if current_name is None:
                return
            rhs = ' '.join(rhs_chunks)
            toks = cls._tokenize_rhs(rhs)
            alts = cls._split_into_alts(toks)
            if current_name in rules:
                rules[current_name].extend(alts)
            else:
                rules[current_name] = alts

        for raw in text.split('\n'):
            line = raw.strip()
            if not line or line.startswith('#') or line.startswith(';'):
                continue
            if cls._is_rule_opener(line):
                _flush()
                rhs_chunks = []
                current_name, rhs = cls._split_rule_line(line)
                if rhs:
                    rhs_chunks.append(rhs)
            elif current_name and line:
                rhs_chunks.append(line)

        _flush()
        return rules

    # --------------------------------------------------------------- converters

    @staticmethod
    def to_playbook_tree(rules, grammar_name):
        """Wrap rules as a named subtree for non-destructive merge into an existing playbook.

        Result: {grammar_name: {rule_name: "alt1 | alt2 | ..."}}

        Nesting under grammar_name keeps rule names from colliding with ops routines
        already in the playbook. flatten_playbook() walks the nested tree and generates:
          ("calculator", "routes: expr, term, ...")
          ("calculator expr", 'expr "+" term | expr "-" term | term')
          ...
        """
        leaves = {name: ' | '.join(alts) for name, alts in rules.items() if alts}
        return {grammar_name: leaves}

    @staticmethod
    def to_prose(rules, grammar_name=None):
        """Generate a prose description of the grammar for semantic knowledge embedding."""
        name = grammar_name or "grammar"
        lines = []
        if rules:
            lines.append(
                "The " + name + " grammar defines rules: "
                + ", ".join(rules.keys()) + "."
            )
        for rule, alts in rules.items():
            if not alts:
                continue
            if len(alts) == 1:
                lines.append("A " + rule + " is defined as " + alts[0] + ".")
            elif len(alts) == 2:
                lines.append("A " + rule + " can be " + alts[0] + " or " + alts[1] + ".")
            else:
                lines.append(
                    "A " + rule + " can be "
                    + ", ".join(alts[:-1]) + ", or " + alts[-1] + "."
                )
        return "\n".join(lines)

    # ----------------------------------------------------------------- file I/O

    @classmethod
    def load_file(cls, path, grammar_name=None, logger=None):
        """Read and parse a BNF/EBNF grammar file.

        grammar_name defaults to the last '_'-delimited segment of the filename stem:
          "playbook_model_calculator.txt" -> "calculator"

        Returns {rules, tree, prose, name} on success, or None on any error.
        Files that contain no rule definitions (no ::= or <name> = ...) return None
        so callers can fall back to treating the file as plain prose.
        """
        def _log(sev, msg):
            if logger is not None:
                logger.log(sev, "GRAMMAR", msg)
            else:
                print("[" + sev.upper() + "] [GRAMMAR] " + msg)

        if not path:
            return None
        if not os.path.isfile(path):
            _log("error", "File not found: " + str(path))
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except Exception as exc:
            _log("error", "Could not read " + str(path) + ": " + str(exc))
            return None

        if not text.strip():
            _log("warning", "Empty file: " + str(path))
            return None

        # Quick check: must contain at least one BNF or EBNF rule separator.
        if '::=' not in text and not re.search(r'<[^>]+>\s*=\s', text):
            return None   # not a grammar file — caller falls back to prose handling

        if grammar_name is None:
            stem = os.path.splitext(os.path.basename(path))[0]
            parts = [p for p in stem.split('_') if p]
            grammar_name = parts[-1] if parts else stem

        rules = cls.parse(text)
        if not rules:
            _log("warning", "Parsed " + str(path) + " but found no rules.")
            return None

        tree = cls.to_playbook_tree(rules, grammar_name)
        prose = cls.to_prose(rules, grammar_name)
        pair_count = len(ModelAssets.flatten_playbook(tree))

        _log("info",
             "Parsed '" + grammar_name + "' from " + str(path)
             + ": " + str(len(rules)) + " rule(s), " + str(pair_count) + " trained pair(s).")
        return {"rules": rules, "tree": tree, "prose": prose, "name": grammar_name}

    # --------------------------------------------------------- model integration

    @classmethod
    def augment(cls, assets, path, grammar_name=None, rebuild=True):
        """Parse a BNF/EBNF file and augment the ModelAssets with it in-flight.

        1. Merges the grammar subtree into assets.playbook under the grammar name.
        2. Appends a prose description to assets.knowledge_texts.
        3. Rebuilds tokenizer + model over the FULL accumulated set (catastrophic
           forgetting proof — anchors + all grammar pairs + all prose docs).
        4. Persists state and optionally re-exports GGUF + Ollama model.

        Returns True on success.

        Called by the '/grammar <file>' interactive command:
            ModelGrammar.augment(assets, user_input.split(maxsplit=1)[1].strip())
        """
        result = cls.load_file(path, grammar_name, logger=assets.logger)
        if result is None:
            return False

        # Merge grammar subtree under its name namespace (non-destructive).
        ModelAssets.merge_tree(assets.playbook, result["tree"])

        # Set grammar_name only when unassigned (preserves an existing ops/playbook name).
        if not assets.grammar_name:
            assets.grammar_name = result["name"]

        assets._refresh_grammar_pairs()

        if result["prose"].strip():
            assets.knowledge_texts.append(result["prose"])

        assets._log("info",
                    "Grammar '" + result["name"] + "' merged ("
                    + str(len(result["rules"])) + " rule(s), "
                    + str(len(assets.grammar_pairs)) + " total grammar pair(s), "
                    + str(len(assets.knowledge_texts)) + " prose doc(s)). Rebuilding...")

        prev_cap = assets.vocab_cap
        assets._set_artifacts(
            assets.builder(assets.knowledge_texts, assets.grammar_pairs,
                           assets.arch, assets.vocab_cap)
        )
        if assets.vocab_cap != prev_cap:
            assets._log("ok",
                        "Vocab ceiling adapted " + str(prev_cap)
                        + " -> " + str(assets.vocab_cap) + " for grammar tokens.")

        assets.save_state()
        if rebuild:
            assets._log("info", "Re-exporting GGUF and rebuilding Ollama model...")
            assets.export_and_rebuild()

        return True


class GrammarRunner:
    """Drives multi-step grammar parsing using the local model as a grammar oracle.

    For each unique non-terminal encountered during parsing, the runner queries the
    model exactly once (prompt: "<grammar_name> <rule_name>") and caches the result.
    This creates a visible multi-step dialog: N model interactions for N unique rules.

    Handles left-recursive grammars automatically (e.g. expr ::= expr "+" term | term)
    by splitting alternatives into base cases and left-recursive extensions, then
    parsing iteratively — effectively operator-precedence climbing generalized to any BNF.

    Grammar-neutral: parse() builds a tree for any BNF grammar; evaluate() computes
    arithmetic for expression grammars and returns a string repr for all others.

    Usage:
        runner = GrammarRunner(
            grammar_name="calculator",
            query_fn=lambda p: get_ollama_answer(p, NAME, host),
            fallback_playbook=assets.playbook.get("calculator", {}),
            logger=logger,
        )
        result = runner.run("3 + 4", start_rule="expr")  # logs 5 model interactions
    """

    # Tokenize user input: each digit separately, operators, parens, identifiers.
    # Digits are single-char to match the grammar's <digit> rule (0-9).
    _INPUT_RE = re.compile(r'\d|[+\-*/()]|[a-zA-Z_]\w*')

    # Tokenize a stored-playbook answer line (bare names, not <angle-bracket> style).
    # The playbook stores rules as: 'expr "+" term | expr "-" term | term'
    # (nonterminals are rendered without angle brackets by ModelGrammar._render_alt).
    _ANSWER_RE = re.compile(
        r'<([^>]+)>'        # group 1 – nonterminal <name>  (BNF-style, just in case)
        r'|"([^"]*)"'       # group 2 – terminal "x"
        r"|'([^']*)'"       # group 3 – terminal 'x'
        r'|(\|)'            # group 4 – alternative separator
        r'|([a-zA-Z_]\w*)' # group 5 – nonterminal bare name
        r'|([(){}[\]*+?])'  # group 6 – EBNF grouping / quantifier
    )

    def __init__(self, grammar_name, query_fn=None, logger=None, fallback_playbook=None):
        self.grammar_name = grammar_name
        self.query_fn = query_fn            # fn(prompt_str) -> answer_str; None = probe mode
        self.logger = logger
        self.fallback_playbook = fallback_playbook or {}
        self._rule_cache = {}               # rule_name -> [[alt_tokens], ...]
        self._interaction_count = 0

    # ------------------------------------------------------------- logging

    def _log(self, sev, msg):
        if self.logger is not None:
            self.logger.log(sev, "RUNNER", msg)
        else:
            print("[" + sev.upper() + "] [RUNNER] " + msg)

    # ------------------------------------------------------------ model query

    @staticmethod
    def _split_alt_groups(tokens):
        """Split a flat token list on pipe delimiters → [[alt_tokens], ...]."""
        groups, current = [], []
        for tok in tokens:
            if tok == ('pipe', '|'):
                if current:
                    groups.append(current)
                current = []
            else:
                current.append(tok)
        if current:
            groups.append(current)
        return groups

    @classmethod
    def _answer_tokens(cls, answer):
        """Tokenize a playbook answer string into (kind, value) tuples.

        Unlike ModelGrammar._tokenize_rhs (which requires <angle-bracket> nonterminals),
        this handles the stored playbook format where nonterminals are bare names:
          'expr "+" term | expr "-" term | term'
        """
        tokens = []
        for m in cls._ANSWER_RE.finditer(answer):
            g = m.lastindex
            if g == 1:
                tokens.append(('nt',   m.group(1).strip()))
            elif g in (2, 3):
                tokens.append(('term', m.group(g)))
            elif g == 4:
                tokens.append(('pipe', '|'))
            elif g == 5:
                tokens.append(('nt',   m.group(5)))
            elif g == 6:
                tokens.append(('op',   m.group(6)))
        return tokens

    def _query_rule(self, rule_name):
        """Query the model for one grammar rule (cached). Returns [[alt_tokens], ...].

        When query_fn is None (probe mode) the model is never called — only the stored
        fallback_playbook is used. This lets probe() run completely offline."""
        if rule_name in self._rule_cache:
            return self._rule_cache[rule_name]

        alts = []

        if self.query_fn is not None:
            prompt = self.grammar_name + " " + rule_name
            answer = self.query_fn(prompt).strip()
            self._interaction_count += 1
            short = answer[:60] + ("..." if len(answer) > 60 else "")
            self._log("info", "[model #" + str(self._interaction_count) + "] "
                      + prompt + " → " + short)
            alts = self._split_alt_groups(self._answer_tokens(answer))

        # Fallback: use stored playbook when model gave nothing, or in probe mode.
        if not alts and rule_name in self.fallback_playbook:
            fb = self.fallback_playbook[rule_name]
            alts = self._split_alt_groups(self._answer_tokens(fb))
            if alts and self.query_fn is not None:
                self._log("info", "  (playbook fallback for '" + rule_name + "')")

        self._rule_cache[rule_name] = alts
        return alts

    @classmethod
    def probe(cls, grammar_name, input_str, fallback_playbook, start_rule=None):
        """Return True if input_str fully parses against fallback_playbook (no model calls).

        Used by the interactive loop to auto-detect whether a user's raw input matches a
        loaded grammar before deciding to route it through GrammarRunner.run() instead of
        sending it to the chat model. Fast and silent — zero model interactions.
        """
        if not fallback_playbook:
            return False
        if start_rule is None:
            start_rule = next(iter(fallback_playbook))
        runner = cls(
            grammar_name=grammar_name,
            query_fn=None,               # probe mode: playbook only
            logger=None,                 # silent
            fallback_playbook=fallback_playbook,
        )
        tokens = runner._tokenize_input(input_str)
        if not tokens:
            return False
        node, pos = runner._parse_rule(start_rule, tokens, 0)
        return node is not None and pos == len(tokens)

    # ----------------------------------------------------------- tokenizer

    def _tokenize_input(self, input_str):
        """Split user input into atomic tokens for grammar matching."""
        return self._INPUT_RE.findall(input_str)

    # ------------------------------------------------------------- parser

    def _try_alternative(self, alt_tokens, tokens, pos):
        """Try to match alt_tokens against tokens[pos:].
        Returns (children, new_pos) on success, (None, pos) on failure."""
        children = []
        cur = pos
        for kind, value in alt_tokens:
            if kind == 'term':
                if cur < len(tokens) and tokens[cur] == value:
                    children.append({"terminal": value})
                    cur += 1
                else:
                    return None, pos
            elif kind == 'nt':
                node, new_cur = self._parse_rule(value, tokens, cur)
                if node is None:
                    return None, pos
                children.append(node)
                cur = new_cur
            # EBNF grouping/quantifier operators are accepted but skipped
        return children, cur

    def _parse_rule(self, rule_name, tokens, pos):
        """Recursive descent parse of rule_name starting at tokens[pos].

        Left-recursive alternatives (first symbol == rule_name itself) are handled
        iteratively: parse a base case first, then extend while more input matches.
        Returns (node, new_pos) on success or (None, pos) on failure.
        """
        if pos >= len(tokens):
            return None, pos

        alts = self._query_rule(rule_name)
        if not alts:
            return None, pos

        original_pos = pos

        # Split: base alts (not left-recursive) vs extend alts (left-recursive).
        base_alts   = [a for a in alts if not (a and a[0] == ('nt', rule_name))]
        extend_alts = [a for a in alts if      a and a[0] == ('nt', rule_name)]

        # Step 1: match a base case.
        node = None
        for alt in base_alts:
            children, new_pos = self._try_alternative(alt, tokens, pos)
            if children is not None:
                node = {"rule": rule_name, "children": children}
                pos = new_pos
                break

        if node is None:
            return None, original_pos

        # Step 2: iteratively extend with left-recursive alternatives.
        while pos < len(tokens):
            extended = False
            for alt in extend_alts:
                rest = alt[1:]   # skip the leading ('nt', rule_name) — already parsed
                children, new_pos = self._try_alternative(rest, tokens, pos)
                if children is not None:
                    node = {"rule": rule_name, "children": [node] + children}
                    pos = new_pos
                    extended = True
                    break
            if not extended:
                break

        return node, pos

    # ----------------------------------------------------------- evaluator

    def evaluate(self, node):
        """Walk a parse tree to a result value.

        For arithmetic expression grammars: computes the numeric result.
        For other grammars: returns a nested string representation.
        Override this method for grammar-specific evaluation.
        """
        if node is None:
            return None
        if "terminal" in node:
            try:
                return int(node["terminal"])
            except (ValueError, TypeError):
                return node["terminal"]

        rule = node.get("rule", "")
        children_vals = [self.evaluate(c) for c in node.get("children", [])]

        # number rule: concatenate digit values as a decimal integer.
        if rule == "number":
            digit_strs = [str(v) for v in children_vals if isinstance(v, (int, float))]
            if digit_strs:
                try:
                    return int("".join(digit_strs))
                except ValueError:
                    pass

        # General arithmetic: two numeric operands + one operator.
        nums = [v for v in children_vals if isinstance(v, (int, float))]
        ops  = [v for v in children_vals if isinstance(v, str) and v in '+-*/']

        if len(nums) == 2 and ops:
            a, b, op = nums[0], nums[1], ops[0]
            if op == '+': return a + b
            if op == '-': return a - b
            if op == '*': return a * b
            if op == '/' and b != 0: return a // b
            if op == '/' and b == 0:
                self._log("error", "Division by zero.")
                return None

        if len(nums) == 1:
            return nums[0]

        # Non-arithmetic fallback: return a parenthesised string of child values.
        return "(" + " ".join(str(v) for v in children_vals if v is not None) + ")"

    # ------------------------------------------------------------ display

    def _tree_str(self, node):
        """Compact string representation of a parse tree."""
        if node is None:
            return "?"
        if "terminal" in node:
            return node["terminal"]
        rule = node.get("rule", "?")
        child_strs = " ".join(self._tree_str(c) for c in node.get("children", []))
        return rule + "(" + child_strs + ")"

    # --------------------------------------------------------- entry points

    def parse(self, input_str, start_rule="expr"):
        """Tokenize input_str and parse it from start_rule.
        Resets rule cache and interaction counter before each parse.
        Returns a parse tree dict or None on failure."""
        self._rule_cache.clear()
        self._interaction_count = 0
        tokens = self._tokenize_input(input_str)
        if not tokens:
            self._log("warning", "Empty input.")
            return None
        node, pos = self._parse_rule(start_rule, tokens, 0)
        if node is None:
            self._log("error", "Parse failed for: '" + input_str + "'")
            return None
        if pos < len(tokens):
            self._log("warning",
                      "Partial parse — consumed " + str(pos) + "/" + str(len(tokens))
                      + " tokens, unparsed: " + str(tokens[pos:]))
        return node

    def run(self, input_str, start_rule="expr"):
        """Full pipeline: parse → evaluate → log result. Returns the result value or None."""
        self._log("info",
                  "Parsing '" + input_str + "' as <" + start_rule + "> "
                  + "via '" + self.grammar_name + "' grammar...")
        node = self.parse(input_str, start_rule)
        if node is None:
            self._log("error", "Could not parse '" + input_str + "'.")
            return None

        tree_str = self._tree_str(node)
        result = self.evaluate(node)

        self._log("ok", "Parse tree : " + tree_str)
        self._log("ok",
                  "Result     : " + str(result)
                  + "  (" + str(self._interaction_count) + " model interaction(s))")
        print("\nResult: " + str(result))
        return result


# ── standalone self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import tempfile

    CALC_BNF = """\
<expr>     ::= <expr> "+" <term> | <expr> "-" <term> | <term>
<term>     ::= <term> "*" <factor> | <term> "/" <factor> | <factor>
<factor>   ::= "(" <expr> ")" | <number>
<number>   ::= <digit> <number> | <digit>
<digit>    ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
"""

    print("=== parse ===")
    rules = ModelGrammar.parse(CALC_BNF)
    for name, alts in rules.items():
        print("  " + name + ": " + str(alts))
    assert list(rules.keys()) == ["expr", "term", "factor", "number", "digit"]
    assert len(rules["expr"]) == 3
    assert len(rules["digit"]) == 10

    print("\n=== playbook tree ===")
    tree = ModelGrammar.to_playbook_tree(rules, "calculator")
    print(json.dumps(tree, indent=2))
    assert "calculator" in tree
    assert "expr" in tree["calculator"]

    print("\n=== trained pairs ===")
    pairs = ModelAssets.flatten_playbook(tree)
    for prompt, answer in pairs:
        print("  '" + prompt + "' -> '" + answer + "'")
    prompts = [p for p, _ in pairs]
    assert "calculator" in prompts
    assert "calculator expr" in prompts
    assert "calculator digit" in prompts

    print("\n=== prose ===")
    prose = ModelGrammar.to_prose(rules, "calculator")
    print(prose)
    assert "calculator" in prose
    assert "expr" in prose

    print("\n=== load_file (named file) ===")
    # Use a predictable filename to test auto-inference of grammar_name from stem.
    named_path = "/tmp/playbook_model_calculator.txt"
    with open(named_path, 'w', encoding='utf-8') as fh:
        fh.write(CALC_BNF)
    result = ModelGrammar.load_file(named_path)
    assert result is not None
    assert result["name"] == "calculator", "Expected 'calculator', got: " + str(result["name"])
    # Override name explicitly
    result2 = ModelGrammar.load_file(named_path, grammar_name="calc")
    assert result2["name"] == "calc"
    assert len(result2["rules"]) == 5
    os.unlink(named_path)

    print("\n=== fake augment ===")
    def _fake_builder(texts, grammar_pairs, arch, vocab_cap):
        return {"model": None, "tokenizer": None, "config": None,
                "tokens": [0], "scores": [], "token_types": [], "merges": [],
                "eos_id": 0, "vocab_cap": vocab_cap, "arch": arch}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
        tf.write(CALC_BNF)
        tmp_path = tf.name

    assets = ModelAssets(
        builder=_fake_builder, knowledge_texts=[], playbook={}, grammar_name=None,
        artifacts=_fake_builder([], [], {"num_layers": 2}, 4096),
        arch={"num_layers": 2}, vocab_cap=4096,
        gguf_path="/tmp/x_grammar_test.gguf", modelfile_path="/tmp/Modelfile_grammar_test",
        ollama_name="x", ollama_host="http://127.0.0.1:11434",
        modelfile_template="<|endoftext|>{{ .Prompt }}", state_path="/tmp/_grammar_test_state.json",
    )

    ok = ModelGrammar.augment(assets, tmp_path, grammar_name="calculator", rebuild=False)
    print("augment ok:", ok)
    assert ok
    assert assets.grammar_name == "calculator"
    assert "calculator" in assets.playbook
    assert list(assets.playbook["calculator"].keys()) == ["expr", "term", "factor", "number", "digit"]
    assert len(assets.knowledge_texts) == 1
    assert len(assets.grammar_pairs) > 0

    print("grammar_name:  ", assets.grammar_name)
    print("playbook keys: ", list(assets.playbook.keys()))
    print("calculator subtree:", list(assets.playbook["calculator"].keys()))
    print("grammar_pairs (" + str(len(assets.grammar_pairs)) + "):")
    for p, a in assets.grammar_pairs:
        print("  '" + p + "' -> '" + a + "'")
    print("knowledge_texts:", len(assets.knowledge_texts), "doc(s)")

    os.unlink(tmp_path)

    # State round-trip: grammar must survive save/load
    assets.save_state()
    state = ModelAssets.load_state("/tmp/_grammar_test_state.json")
    assert state is not None
    assert "calculator" in state.get("playbook", {})
    print("\nState round-trip OK: 'calculator' in saved playbook.")

    # ── GrammarRunner self-test ───────────────────────────────────────────────
    print("\n=== GrammarRunner self-test ===")

    # Build a fake query_fn from the parsed grammar — mimics the trained model's answers.
    CALC_ANSWERS = {name: ' | '.join(alts) for name, alts in rules.items()}

    def fake_query(prompt):
        parts = prompt.split(maxsplit=1)
        rule = parts[1] if len(parts) > 1 else ""
        return CALC_ANSWERS.get(rule, "")

    runner = GrammarRunner(grammar_name="calculator", query_fn=fake_query)

    test_cases = [
        ("3 + 4",       7,  "expr"),
        ("2 + 3 * 4",   14, "expr"),
        ("9 / 3 - 1",   2,  "expr"),
        ("(2 + 3) * 4", 20, "expr"),
        ("5",           5,  "expr"),
    ]

    for expr_in, expected, start in test_cases:
        result = runner.run(expr_in, start_rule=start)
        ok = result == expected
        print(("PASS" if ok else "FAIL") + "  " + expr_in
              + " = " + str(result) + " (expected " + str(expected) + ")"
              + "  [" + str(runner._interaction_count) + " model interaction(s)]")
        assert ok, "Expected " + str(expected) + ", got " + str(result)
        assert runner._interaction_count == 5, \
            "Expected 5 model interactions, got " + str(runner._interaction_count)

    # Partial / invalid input must not crash.
    result_bad = runner.run("3 +", start_rule="expr")
    assert result_bad is None or isinstance(result_bad, (int, float, str)), \
        "Partial parse should not crash"
    print("PASS  partial input '3 +' handled gracefully (result: " + str(result_bad) + ")")

    # ── GrammarRunner.probe() self-test ──────────────────────────────────────
    print("\n=== GrammarRunner.probe() self-test ===")

    # Build a playbook subtree (same structure as assets.playbook["calculator"]).
    calc_subtree = {name: ' | '.join(alts) for name, alts in rules.items()}

    probe_cases = [
        ("1+1",       True,  "valid single-digit arithmetic"),
        ("3 + 4",     True,  "spaced arithmetic"),
        ("(2+3)*4",   True,  "parenthesised expression"),
        ("9/3-1",     True,  "division and subtraction"),
        ("hello",     False, "plain word — not a calculator expression"),
        ("",          False, "empty string"),
        ("3 + + 4",   False, "double operator"),
        ("1 + hello",  False, "identifier after operator — not a calculator expression"),
    ]

    for expr_in, expected, label in probe_cases:
        got = GrammarRunner.probe("calculator", expr_in, calc_subtree)
        ok = got == expected
        print(("PASS" if ok else "FAIL") + "  probe('" + expr_in + "') = "
              + str(got) + "  (" + label + ")")
        assert ok, "probe('" + expr_in + "'): expected " + str(expected) + ", got " + str(got)

    print("\n✅ All assertions passed.")
