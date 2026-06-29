"""class_nocode_runner.py — grammar-agnostic "nocode" runner (host CPU path).

A replicat-and-adapted of ``class_model_runner.py``: same shape (config → backend → REPL), but the
host backend drives ``NoCodeGrammarRunner`` so the executable logic is **emitted by the model** and
run by a grammar-agnostic loop, governed by the exec policy ladder
(``token_select`` → ``vocab_verified`` → ``generative``).

Scope now: host model → CPU (Python bodies).  Device/NPU parity is deferred (same shape — the device
already runs a C++ GrammarRunner; the nocode device path embeds the bodies in the TCN model later).

Reuses the proven helpers from ``class_model_runner`` (logger, Ollama oracle, grammar meta, host
build/export, config defaults) — only the host backend + REPL + the policy are new here.
"""
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                 # scripts/ on path

from classes import class_model_runner as base             # reuse proven helpers
from classes.class_nocode_grammar import NoCodeGrammarRunner, CodeExecPolicy


# --------------------------------------------------------------------------- vocab
def _load_nocode_vocab(grammar_path, log):
    """Load the grammar's declared vocabulary (function_vocabulary preferred, command_vocabulary
    accepted) into a commands dict, and detect the execution mode + evaluator token.

    Returns (commands, mode, eval_token):
      * commands   {token: body, "_exec": ...}
      * mode       "evaluate" (apply a tree evaluator) or "execute" (per-terminal procedure)
      * eval_token name of the token carrying the evaluator (evaluate-mode only)
    """
    commands = {}
    mode = "execute"
    eval_token = "evaluate"
    for f in base._grammar_training(grammar_path):
        p = f if (os.path.isabs(f) or os.sep in f) else os.path.join(base._TRAINING_DIR, f)
        if not os.path.isfile(p):
            log.log("warning", "NOCODE", "declared training file not found: %s" % p)
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (OSError, ValueError) as e:
            log.log("warning", "NOCODE", "could not read %s: %s" % (p, e))
            continue
        if not (isinstance(obj, dict) and obj.get("_type") in ("function_vocabulary", "command_vocabulary")):
            continue
        bodies = {k: v for k, v in obj.items() if not k.startswith("_") and isinstance(v, str)}
        commands.update(bodies)
        if "_exec" in obj:
            commands["_exec"] = obj["_exec"]
        m = obj.get("_mode")
        if m in ("evaluate", "evaluate_ops"):
            mode = m
            if m == "evaluate" and bodies:        # whole-function token (evaluate_ops looks up op tokens)
                eval_token = next(iter(bodies))
        log.log("info", "NOCODE", "loaded %d body token(s) from %s (type=%s, mode=%s, exec=%s)"
                % (len(bodies), os.path.basename(p), obj.get("_type"),
                   obj.get("_mode", "execute"), obj.get("_exec", "shell")))
    return commands, mode, eval_token


# --------------------------------------------------------------------------- backend
def _setup_host(cfg, log):
    """Build the host backend context for the active grammar. Returns host_ctx or None."""
    if not cfg.get("model"):
        log.log("error", "NOCODE", "model is required (host path) — /set model <name>")
        return None
    if not cfg.get("grammar"):
        log.log("error", "NOCODE", "grammar is required — /set grammar <file>")
        return None
    # Accept either a bare filename (composed with models/grammars/) or a full/relative path —
    # so '--grammar playbook_x.txt' and '--grammar models/grammars/playbook_x.txt' both work.
    g = cfg["grammar"]
    grammar_file = next(
        (c for c in (g, os.path.join(base._GRAMMAR_DIR, g),
                     os.path.join(base._GRAMMAR_DIR, os.path.basename(g))) if os.path.isfile(c)),
        os.path.join(base._GRAMMAR_DIR, g),
    )
    from classes.class_model_grammar import ModelGrammar          # lazy (ML chain)
    gf = ModelGrammar.load_file(grammar_file, logger=log)
    if not gf:
        log.log("error", "NOCODE", "could not load grammar '%s'" % grammar_file)
        return None
    gname = gf["name"]
    tree = gf["tree"]
    playbook = tree[gname] if isinstance(tree.get(gname), dict) else tree
    rules = list(playbook.keys())
    start_rule = rules[0] if rules else "expr"
    qfn = base.ollama_query_fn(cfg["model"], cfg.get("host"))
    commands, mode, eval_token = _load_nocode_vocab(grammar_file, log)

    def make_runner():
        return NoCodeGrammarRunner(
            grammar_name=gname, query_fn=qfn, fallback_playbook=playbook, logger=log,
            commands=commands, policy=cfg["policy"], mode=mode, eval_token=eval_token,
        )

    log.log("ok", "NOCODE", "host ready — grammar='%s' mode=%s policy=%s oracle='%s'"
            % (gname, mode, cfg["policy"], cfg["model"]))
    return {"gname": gname, "playbook": playbook, "rules": rules, "start_rule": start_rule,
            "mode": mode, "make_runner": make_runner}


# --------------------------------------------------------------------------- REPL
def _print_help():
    print("  /?              this help")
    print("  /policy [mode]  show or set exec policy (token_select | vocab_verified | generative)")
    print("  /grammar [file] show active grammar rules, or switch grammar")
    print("  /set <k> <v>    set model | grammar | host | policy")
    print("  /create         retrain the host model (transposed logic trained into it)")
    print("  /export         export the host model (CPU ONNX)")
    print("  /bye            quit")
    print("  <input>         expression (evaluate-mode) or procedure trigger (execute-mode)")


def _repl(cfg, host_ctx, log):
    log.log("info", "NOCODE", "mode=%s policy=%s — type input, /? for help, /bye to quit"
            % (host_ctx["mode"], cfg["policy"]))
    gname = host_ctx["gname"]
    rules = host_ctx["rules"]
    start_rule = host_ctx["start_rule"]

    while True:
        try:
            ui = input("nocode> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not ui:
            continue
        parts = ui.split()
        cmd = parts[0].lower()

        if cmd in ("/bye", "exit", "quit"):
            log.log("ok", "NOCODE", "closing nocode runner")
            return None
        if cmd in ("/?", "/help"):
            _print_help()
            continue
        if cmd == "/policy":
            if len(parts) >= 2:
                cfg["policy"] = CodeExecPolicy.coerce(parts[1])
                log.log("ok", "NOCODE", "policy = %s" % cfg["policy"])
                return "reload"     # rebuild backend so make_runner picks up the new policy
            log.log("info", "NOCODE", "policy = %s  (of %s)" % (cfg["policy"], ", ".join(CodeExecPolicy.ALL)))
            continue
        if cmd == "/grammar":
            if len(parts) >= 2 and parts[1].strip():
                cfg["grammar"] = parts[1].strip()
                return "reload"
            for r in rules:
                print("  <%s> ::= %s" % (r, host_ctx["playbook"][r]))
            continue
        if cmd == "/set":
            if len(parts) >= 3 and parts[1].lower() in ("model", "grammar", "host", "policy"):
                key = parts[1].lower()
                val = parts[2]
                cfg[key] = CodeExecPolicy.coerce(val) if key == "policy" else val
                log.log("ok", "NOCODE", "%s = %s" % (key, cfg[key]))
                if key in ("model", "grammar", "policy", "host"):
                    return "reload"
            else:
                log.log("warning", "NOCODE", "usage: /set model|grammar|host|policy <value>")
            continue
        if cmd == "/create":
            base._run_host_build(log, cfg)
            continue
        if cmd == "/export":
            base._run_host_export(log, cfg)
            continue

        # ---- bare input: procedure trigger (execute-mode) or expression (evaluate-mode) ----
        first = parts[0]
        if host_ctx["mode"] == "execute" and (ui == gname or first in rules):
            host_ctx["make_runner"]().execute(start_rule=(first if first in rules else None))
        elif NoCodeGrammarRunner.probe(gname, ui, host_ctx["playbook"], start_rule=start_rule):
            host_ctx["make_runner"]().run(ui, start_rule=start_rule)
        else:
            log.log("warning", "NOCODE", "not a valid %s expression/procedure: '%s'" % (gname, ui))


def run(mode="host", config=None, logger=None):
    log = logger or base._logger()
    cfg = dict(base.DEFAULT_CONFIG)
    cfg.setdefault("policy", CodeExecPolicy.DEFAULT)
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    cfg["policy"] = CodeExecPolicy.coerce(cfg.get("policy"))

    if mode == "device":
        log.log("warning", "NOCODE",
                "device/NPU path is deferred in the nocode track — use host mode for now")
        return

    while True:
        host_ctx = _setup_host(cfg, log)
        if not host_ctx:
            return
        nxt = _repl(cfg, host_ctx, log)
        if nxt != "reload":
            return
