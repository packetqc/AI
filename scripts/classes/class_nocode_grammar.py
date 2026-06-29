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
                 eval_token="evaluate", **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = CodeExecPolicy.coerce(policy)
        self.mode = mode                 # "evaluate" (apply to parse tree) | "execute" (per terminal)
        self.eval_token = eval_token     # token name carrying the evaluator (evaluate-mode)
        self._eval_fn = None             # cached emitted evaluator for this run

    # --------------------------------------------------- body sourcing (the policy)

    @staticmethod
    def _norm(s):
        """Whitespace-normalized form for verified comparison (the tiny model rarely matches byte-for-byte)."""
        return "\n".join(line.rstrip() for line in (s or "").strip().splitlines())

    def _query_body(self, token):
        """Ask the model for a token's body using the trained prompt '<grammar> <token>'."""
        if self.query_fn is None:
            return None
        prompt = self.grammar_name + " " + token
        emitted = (self.query_fn(prompt) or "").strip()
        self._interaction_count += 1
        self._log("info", "[model body #%d] %s -> %d char(s)"
                  % (self._interaction_count, prompt, len(emitted)))
        return emitted

    def _resolve_body(self, token):
        """Return (body, source_label) for ``token`` per the active exec policy."""
        vocab_body = self.commands.get(token)

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
            vb = self.commands.get(token)
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

    # --------------------------------------------------- execute-mode (procedure grammars)

    def _run_os_command(self, token, cmd):
        """Run one terminal token's logic, sourcing the body from the model per policy.

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
            try:
                exec(compile(body, "<emitted:" + token + ">", "exec"), self._make_ns())  # noqa: S102
            except Exception as exc:                                  # noqa: BLE001
                self._log("error", "python exec failed [" + token + "]: " + str(exc))
            return

        try:
            subprocess.run(body, shell=True, timeout=120)             # noqa: S602
        except subprocess.TimeoutExpired:
            self._log("error", "Timed out: " + body)
        except Exception as exc:                                      # noqa: BLE001
            self._log("error", "Command failed: " + str(exc))
