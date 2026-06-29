"""class_logic_transposer.py — transpose working CPU-side Python logic into model-trainable tokens.

The "nocode" bridge.  Today a grammar's executable logic lives on the CPU side: either as
hardcoded Python (``GrammarRunner.evaluate`` for expression grammars) or as a hand-authored
side-car command vocabulary that the runner looks up at runtime.  Neither is carried *inside*
the model — so a new/modified grammar still implies CPU-side authoring.

This class lifts that logic into a **function-vocabulary** JSON whose ``(token -> body)`` pairs
are TRAINED into the model (see the pipeline hook in ``model_create_hf_cl.py``).  The model then
EMITS the logic during interaction and a grammar-agnostic runner (``nocode_runner.py``) executes
what the model pushed — no per-grammar CPU code to maintain.

Two granularities, mirroring the conception:

  * ``analyze_grammar(name)``         — GLOBAL: capture the whole CPU logic that serves a grammar.
  * ``transpose_method(file, cls, m)``— PER-METHOD: capture one function/method body as a token.

Design choices:
  * Works at the **AST level on the source file** — never imports the target module, so it pulls
    in zero ML / heavy dependencies and runs fully offline.
  * The transform that makes a bound method standalone is simply: drop the ``self`` parameter and
    rewrite every ``self.X`` reference to the bare name ``X``.  Recursion (``self.evaluate`` ->
    ``evaluate``) resolves to the function defined in the exec namespace; helper references
    (``self._log`` -> ``_log``) are supplied by the runner's exec namespace as shims.

Output schema (a superset of the existing ``command_vocabulary`` the pipeline already understands):

  {
    "_type": "function_vocabulary",   # NEW type — trained as anchors AND loaded as side-car
    "_grammar": "<name>",
    "_exec": "python",
    "_mode": "evaluate" | "execute",  # evaluate = applied to a parse tree; execute = per-terminal
    "_source": "<provenance>",         # where the body was lifted from (audit)
    "<token>": "<standalone body>",
    ...
  }
"""
import ast
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))            # scripts/classes -> scripts -> repo
_GRAMMAR_FILE = os.path.join(_HERE, "class_model_grammar.py")
_TRAINING_DIR = os.path.join(_REPO, "models", "training")


class _StripSelf(ast.NodeTransformer):
    """Rewrite ``self.X`` -> ``X`` and drop the leading ``self`` parameter of the top function."""

    def __init__(self):
        self._top = True

    def visit_FunctionDef(self, node):                       # noqa: N802 (ast API)
        if self._top:
            self._top = False
            if node.args.args and node.args.args[0].arg == "self":
                node.args.args = node.args.args[1:]
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):                         # noqa: N802 (ast API)
        # self.<attr>  ->  <attr>   (bare free name; namespace supplies recursion + shims)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        self.generic_visit(node)
        return node


class LogicTransposer:
    """Analyze working CPU-side logic and emit model-trainable function tokens."""

    # Declarative registry: grammar -> where its CPU logic lives and how it is applied.
    # ``mode == "evaluate"`` lifts a single tree-evaluation method (expression grammars).
    # ``mode == "promote"`` promotes an existing command-vocabulary's bodies to trainable tokens
    # (procedure grammars whose logic is already authored as per-terminal snippets).
    _GRAMMAR_LOGIC = {
        "calculator": {
            "mode": "evaluate",
            "file": _GRAMMAR_FILE,
            "class": "GrammarRunner",
            "method": "evaluate",
            "token": "evaluate",
            "exec": "python",
        },
        "pyhealthcheck":   {"mode": "promote", "vocab": "train_python_healthcheck_commands.json"},
        "linux_healthcheck": {"mode": "promote", "vocab": "train_linux_healthcheck_commands.json"},
        "kali_discovery":  {"mode": "promote", "vocab": "train_kali_discovery_commands.json"},
        "fibonacci":       {"mode": "promote", "vocab": "train_fibonacci_commands.json"},
    }

    def __init__(self, logger=None):
        self.logger = logger

    def _log(self, sev, msg):
        if self.logger is not None:
            self.logger.log(sev, "TRANSPOSE", msg)
        else:
            print("[" + sev.upper() + "] [TRANSPOSE] " + msg)

    # ------------------------------------------------------- per-method (granular)

    @staticmethod
    def transpose_method(source_file, class_name, method_name):
        """Return the standalone source of one method, lifted from ``source_file`` by AST.

        The method is detached from its class: ``self`` is dropped and every ``self.X`` becomes the
        bare name ``X`` (recursion + helper shims are bound by the exec namespace at runtime).
        Never imports the module — parses the file text directly.
        """
        with open(source_file, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())

        func = None
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name == class_name:
                for item in cls.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        func = item
                        break
        if func is None:
            raise ValueError("method %s.%s not found in %s"
                             % (class_name, method_name, source_file))

        func = _StripSelf().visit(func)
        ast.fix_missing_locations(func)
        return ast.unparse(func)

    # ------------------------------------------------------- global (per grammar)

    def analyze_grammar(self, grammar_name):
        """Return the function-vocabulary dict for one grammar (does not write).

        ``evaluate`` grammars contribute their tree-evaluation method as a single trainable token.
        ``promote`` grammars contribute the bodies already authored in their command vocabulary,
        now marked trainable so the model carries (and can emit) them.
        """
        spec = self._GRAMMAR_LOGIC.get(grammar_name)
        if not spec:
            raise ValueError("no transposition spec registered for grammar '%s'" % grammar_name)

        vocab = {
            "_type": "function_vocabulary",
            "_grammar": grammar_name,
            "_exec": spec.get("exec", "python"),
            "_mode": "evaluate" if spec["mode"] == "evaluate" else "execute",
        }

        if spec["mode"] == "evaluate":
            body = self.transpose_method(spec["file"], spec["class"], spec["method"])
            vocab["_source"] = "%s::%s.%s" % (os.path.relpath(spec["file"], _REPO),
                                              spec["class"], spec["method"])
            vocab[spec["token"]] = body
            self._log("ok", "lifted %s -> token '%s' (%d chars)"
                      % (vocab["_source"], spec["token"], len(body)))
            return vocab

        # promote: copy the bodies from the existing command vocabulary
        src = os.path.join(_TRAINING_DIR, spec["vocab"])
        with open(src, "r", encoding="utf-8") as fh:
            cmd = json.load(fh)
        vocab["_exec"] = cmd.get("_exec", "shell")
        vocab["_source"] = os.path.relpath(src, _REPO)
        n = 0
        for k, v in cmd.items():
            if not k.startswith("_") and isinstance(v, str):
                vocab[k] = v
                n += 1
        self._log("ok", "promoted %d body token(s) from %s" % (n, spec["vocab"]))
        return vocab

    def emit(self, grammar_name, out_path=None):
        """Write ``models/training/train_<grammar>_functions.json`` and return its path."""
        vocab = self.analyze_grammar(grammar_name)
        if out_path is None:
            out_path = os.path.join(_TRAINING_DIR, "train_%s_functions.json" % grammar_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(vocab, fh, indent=2, ensure_ascii=False)
        self._log("ok", "wrote %s" % os.path.relpath(out_path, _REPO))
        return out_path

    @classmethod
    def known_grammars(cls):
        return sorted(cls._GRAMMAR_LOGIC)
