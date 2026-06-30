"""class_nocode_grammar.py — grammar-agnostic execution of model-emitted logic.

``NoCodeGrammarRunner`` subclasses the proven ``GrammarRunner`` and changes ONE thing: where the
executable logic comes from.  The base runner hardcodes ``evaluate()`` (expression grammars) and
looks command bodies up in a side-car dict (procedure grammars).  The nocode runner instead sources
the body from the MODEL — the model was trained on (``"<grammar> <token>" -> body``) anchors, so it
can emit the logic — and executes what the model pushed.

How literally "the model pushes the code" is governed by the exec policy (the user's ladder):

  token_select    body comes from the carried vocabulary only (≈ today; deterministic baseline).
  vocab_verified  model emits a body; if it matches the carried vocabulary it is used, otherwise
                  the runner falls back to the verified body (drift / hallucination guard).
  generative      the model-emitted body is executed directly (destiny default — no CPU-side code).

Everything else (tokenizing, parsing, left-recursion, rule caching) is inherited unchanged.
"""
import os
import subprocess

from classes.class_model_grammar import GrammarRunner


class CodeExecPolicy:
    """Where ``NoCodeGrammarRunner`` sources an executable body from."""
    TOKEN_SELECT = "token_select"
    VOCAB_VERIFIED = "vocab_verified"
    GENERATIVE = "generative"
    ALL = (TOKEN_SELECT, VOCAB_VERIFIED, GENERATIVE)
    DEFAULT = VOCAB_VERIFIED   # safe rung; promote to GENERATIVE as the model matures

    @classmethod
    def coerce(cls, value):
        v = (value or "").strip().lower()
        return v if v in cls.ALL else cls.DEFAULT


class NoCodeGrammarRunner(GrammarRunner):
    """A GrammarRunner whose logic is emitted by the model, not hardcoded / side-car-looked-up."""

    def __init__(self, *args, policy=CodeExecPolicy.DEFAULT, mode="execute",
                 eval_token="evaluate", owners=None, defines=None, grammar_tokens=None,
                 vocab_by_owner=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = CodeExecPolicy.coerce(policy)
        self.mode = mode                 # "evaluate" (apply to parse tree) | "execute" (per terminal)
        self.eval_token = eval_token     # token name carrying the evaluator (evaluate-mode)
        self.owners = owners or {}       # token/rule -> canonical (first-declared) owning grammar
        self.defines = defines or {}     # token -> [EVERY grammar that defines it] (collision detection)
        self.grammar_tokens = grammar_tokens or {}   # grammar -> set(tokens it defines) (local shadowing)
        self.vocab_by_owner = vocab_by_owner or {}   # (grammar, token) -> verified body (owner-qualified)
        self._active_grammar = None      # grammar whose procedure is currently running (set per execute())
        self._eval_fn = None             # cached emitted evaluator for this run
        self._context = {}               # shared execute-mode context: prompt args + function outputs

    # --------------------------------------------------- cross-grammar function ownership
    def _owner_of(self, token):
        """Which grammar provides ``token`` in the CURRENT context — by scope + ownership, never by
        load order:
          1. the ACTIVE grammar, if it defines the token — a grammar's own function shadows others;
          2. else the SOLE grammar that defines it — a shared function called across grammars;
          3. else (defined differently by several others) — first-declared + a loud [ambiguous] warn.
        Both the model query namespace AND the verified vocab key derive from this, so they always
        agree (no first/last drift)."""
        ag = self._active_grammar
        if ag and token in self.grammar_tokens.get(ag, ()):      # local definition shadows
            return ag
        defs = self.defines.get(token) or ([self.owners[token]] if token in self.owners else [])
        if not defs:
            return ag or self.grammar_name
        if len(defs) == 1:                                       # one provider = shared function
            return defs[0]
        self._log("warning", "[ambiguous] '%s' defined by %s — using '%s'; declare it in the calling "
                  "grammar to disambiguate" % (token, ", ".join(defs), defs[0]))
        return defs[0]

    def _vocab_body(self, name):
        """Verified body for ``name`` under its resolved owner (falls back to the flat side-car)."""
        vb = self.vocab_by_owner.get((self._owner_of(name), name))
        return vb if vb is not None else (self.commands or {}).get(name)

    # --------------------------------------------------- body sourcing (the policy)

    @staticmethod
    def _norm(s):
        """Whitespace-normalized form for verified comparison (the tiny model rarely matches byte-for-byte)."""
        return "\n".join(line.rstrip() for line in (s or "").strip().splitlines())

    # Continuity protocol: an over-budget ATOMIC body the transposer couldn't decompose is emitted
    # in ordered chunks ending with these markers; the runner reassembles the COMPLETE body before
    # executing. Prompt chain: "<grammar> <token>", "<grammar> <token> §1", "<grammar> <token> §2"...
    CONT_MARK = "[[CONT]]"
    END_MARK = "[[END]]"
    MAX_CHUNKS = 16

    def _assemble(self, token, getter):
        """Fetch ``token``'s body via ``getter(name) -> str|None``, following [[CONT]] continuation
        chunks until [[END]] (or a chunk with no marker). Returns the reassembled body with markers
        stripped. A non-chunked token returns its single body unchanged (backward compatible)."""
        parts = []
        name = token
        for i in range(self.MAX_CHUNKS):
            raw = getter(name)
            if raw is None:
                break
            raw = raw.rstrip()
            cont = raw.endswith(self.CONT_MARK)
            end = raw.endswith(self.END_MARK)
            if cont:
                raw = raw[:-len(self.CONT_MARK)].rstrip("\n")
            elif end:
                raw = raw[:-len(self.END_MARK)].rstrip("\n")
            parts.append(raw)
            if cont:
                name = "%s §%d" % (token, i + 1)
                continue
            break
        if not parts:
            return None
        body = "\n".join(parts)
        if len(parts) > 1:
            self._log("info", "[continuity] reassembled '%s' from %d chunk(s) -> %d char(s)"
                      % (token, len(parts), len(body)))
        return body

    def _query_body(self, token):
        """Ask the model for a token's body (continuity-aware) via prompt '<owner-grammar> <token>'.

        ``owners`` maps a token to the grammar that defined it, so a composed grammar (combo calling
        fibonacci/greeting) queries each body under the namespace it was trained with."""
        if self.query_fn is None:
            return None
        ns = self._owner_of(token)

        def get(name):
            r = self.query_fn(ns + " " + name)
            self._interaction_count += 1
            return r.strip() if r is not None else None

        body = self._assemble(token, get)
        if body is not None:
            self._log("info", "[model body] %s %s -> %d char(s)" % (ns, token, len(body)))
        return body

    def _resolve_body(self, token):
        """Return (body, source_label) for ``token`` per the active exec policy."""
        vocab_body = self._assemble(token, self._vocab_body)

        if self.policy == CodeExecPolicy.TOKEN_SELECT:
            return vocab_body, "vocab"

        emitted = self._query_body(token)

        if self.policy == CodeExecPolicy.GENERATIVE:
            if emitted:
                return emitted, "model"
            return vocab_body, "vocab(fallback:empty-emit)"

        # vocab_verified
        if emitted and vocab_body is not None:
            if self._norm(emitted) == self._norm(vocab_body):
                return vocab_body, "model==vocab"
            self._log("warning", "[drift] emitted body != vocab for '%s' — using verified body" % token)
            return vocab_body, "vocab(drift)"
        if emitted and vocab_body is None:
            return emitted, "model(no-vocab)"
        return vocab_body, "vocab"

    # --------------------------------------------------- exec namespace

    def _make_ns(self):
        """Fresh global namespace for an emitted body: real builtins + a logging shim.

        Not a security sandbox — grammars are authored by trusted users (the bodies do their own
        imports / IO).  The fresh dict just keeps emitted code from clobbering the runner's globals.
        """
        return {
            "__builtins__": __builtins__,
            "_log": lambda sev, msg="": self._log("info", "[emitted] " + str(msg)),
        }

    # --------------------------------------------------- evaluate-mode (expression grammars)

    def evaluate(self, node):
        """Evaluate a parse tree using the model-sourced evaluator (evaluate-mode).

        For non-evaluate grammars, defer to the inherited arithmetic/string evaluator so the
        subclass is safe to use everywhere.
        """
        if self.mode == "evaluate_ops":
            return self._evaluate_ops(node)
        if self.mode != "evaluate":
            return super().evaluate(node)

        if self._eval_fn is None:
            body, src = self._resolve_body(self.eval_token)
            if not body:
                self._log("error", "no '%s' body available (policy=%s)" % (self.eval_token, self.policy))
                return super().evaluate(node)
            ns = self._make_ns()
            try:
                exec(compile(body, "<emitted:%s.%s>" % (self.grammar_name, self.eval_token), "exec"), ns)  # noqa: S102
                self._eval_fn = ns.get(self.eval_token)
                self._log("info", "[evaluate] sourced from %s (policy=%s)" % (src, self.policy))
            except Exception as exc:                                  # noqa: BLE001
                self._log("error", "emitted '%s' failed to compile (%s) — falling back" % (self.eval_token, exc))
                self._eval_fn = None
            if self._eval_fn is None:
                if self.policy == CodeExecPolicy.GENERATIVE and self.eval_token in self.commands:
                    self._log("warning", "generative emit unusable — using verified body")
                    try:
                        ns = self._make_ns()
                        exec(compile(self.commands[self.eval_token], "<verified>", "exec"), ns)  # noqa: S102
                        self._eval_fn = ns.get(self.eval_token)
                    except Exception:                                 # noqa: BLE001
                        self._eval_fn = None
                if self._eval_fn is None:
                    return super().evaluate(node)

        try:
            return self._eval_fn(node)
        except Exception as exc:                                      # noqa: BLE001
            self._log("error", "emitted evaluator raised (%s) — falling back" % exc)
            return super().evaluate(node)

    # --------------------------------------------------- evaluate_ops (decomposed expression grammars)

    _OP_TOKEN = {"+": "op_add", "-": "op_sub", "*": "op_mul", "/": "op_div"}

    def _apply(self, token, inputs):
        """Source ``token``'s small body from the model (per policy), run it with ``inputs`` in the
        namespace, and return the ``result`` it sets.  Falls back to the verified body on failure."""
        body, src = self._resolve_body(token)
        if not body:
            self._log("error", "no body for op token '%s' (policy=%s)" % (token, self.policy))
            return None
        ns = self._make_ns()
        ns.update(inputs)
        try:
            exec(compile(body, "<emitted:%s.%s>" % (self.grammar_name, token), "exec"), ns)  # noqa: S102
            return ns.get("result")
        except Exception as exc:                                      # noqa: BLE001
            self._log("warning", "op '%s' emitted body failed (%s) — trying verified" % (token, exc))
            vb = self._vocab_body(token)
            if vb and vb != body:
                ns2 = self._make_ns()
                ns2.update(inputs)
                try:
                    exec(compile(vb, "<verified:%s>" % token, "exec"), ns2)  # noqa: S102
                    return ns2.get("result")
                except Exception:                                     # noqa: BLE001
                    pass
            return None

    def _evaluate_ops(self, node):
        """Walk the parse tree (generic structure) and compute each node via a small model-emitted
        operation token.  The runner owns only the tree recursion; the arithmetic lives in the model."""
        if node is None:
            return None
        if "terminal" in node:
            try:
                return int(node["terminal"])
            except (ValueError, TypeError):
                return node["terminal"]

        rule = node.get("rule", "")
        cvals = [self._evaluate_ops(c) for c in node.get("children", [])]
        nums = [v for v in cvals if isinstance(v, (int, float))]
        ops = [v for v in cvals if isinstance(v, str) and v in "+-*/"]

        if rule == "number":
            digits = [v for v in cvals if isinstance(v, (int, float))]
            if digits:
                r = self._apply("number", {"digits": digits})
                if r is not None:
                    return r

        if len(nums) == 2 and ops:
            token = self._OP_TOKEN.get(ops[0])
            if token:
                return self._apply(token, {"a": nums[0], "b": nums[1]})

        if len(nums) == 1:
            return nums[0]

        return "(" + " ".join(str(v) for v in cvals if v is not None) + ")"

    # --------------------------------------------------- execute-mode (procedure grammars + data flow)

    def execute(self, start_rule=None, args=None):
        """Run a procedure grammar with a SHARED CONTEXT so functions pass data to one another.

        ``args`` are prompt arguments (the words typed after the grammar name): bound as ``args``
        (list) and ``arg0``, ``arg1`` ... in every body's namespace. As each token body runs, its
        ``result`` is captured under the token name, so a later body can read what an earlier one
        gathered — autonomous data flow / injection (e.g. ``<gather>`` sets ``result``; ``<use>``
        reads ``gather``)."""
        # the grammar whose procedure is running — its own functions shadow same-named ones elsewhere
        self._active_grammar = next((g for g, toks in self.grammar_tokens.items()
                                     if start_rule in toks), None) or self.owners.get(start_rule) \
            or self.grammar_name
        self._context = {"args": list(args or [])}
        for i, a in enumerate(args or []):
            self._context["arg%d" % i] = a
        return super().execute(start_rule)

    def _run_os_command(self, token, cmd):
        """Run one terminal token's logic, sourcing the body per policy, with the shared execution
        context injected (prompt args + upstream function outputs) and this body's ``result``
        captured back into the context for downstream tokens.

        ``cmd`` (the side-car body) is still passed by the base traversal; we treat it as the
        verified vocabulary and let ``_resolve_body`` decide what actually runs.
        """
        body, src = self._resolve_body(token)
        if body is None:
            self._log("error", "no body for token '%s' (policy=%s)" % (token, self.policy))
            return
        exec_mode = (self.commands or {}).get("_exec", "shell")
        self._log("info", "[%s/%s] %s (%s)" % (exec_mode, self.policy, token, src))

        if exec_mode == "python":
            ns = self._make_ns()
            ns.update(self._context)                  # inject prompt args + upstream function outputs
            try:
                exec(compile(body, "<emitted:" + token + ">", "exec"), ns)  # noqa: S102
                if "result" in ns:                    # capture this function's output for downstream
                    self._context[token] = ns["result"]
            except Exception as exc:                                  # noqa: BLE001
                self._log("error", "python exec failed [" + token + "]: " + str(exc))
            return

        # shell mode: expose context scalars as $nc_<name> env vars; capture stdout for downstream
        env = dict(os.environ)
        for k, v in self._context.items():
            if isinstance(v, (str, int, float)):
                env["nc_" + str(k)] = str(v)
        try:
            r = subprocess.run(body, shell=True, timeout=120, env=env,            # noqa: S602
                               capture_output=True, text=True)
            if r.stdout:
                print(r.stdout, end="")
                self._context[token] = r.stdout.strip()
            if r.stderr:
                print(r.stderr, end="")
        except subprocess.TimeoutExpired:
            self._log("error", "Timed out: " + body)
        except Exception as exc:                                      # noqa: BLE001
            self._log("error", "Command failed: " + str(exc))
