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
def _resolve_grammar(g):
    """Resolve a grammar arg: bare filename (composed with models/grammars/) OR a full/relative path."""
    return next(
        (c for c in (g, os.path.join(base._GRAMMAR_DIR, g),
                     os.path.join(base._GRAMMAR_DIR, os.path.basename(g))) if os.path.isfile(c)),
        os.path.join(base._GRAMMAR_DIR, g),
    )


def _default_model_name():
    """Default model when none is given: the SAME base name as the invoking script — e.g.
    'nocode_runner.py' -> model 'nocode_runner' (convention, not hardcoded). Override with --model."""
    return os.path.splitext(os.path.basename(sys.argv[0] or "nocode_runner"))[0] or "nocode_runner"


def _grammar_name_of(path):
    """The grammar name a file declares ('# Grammar : NAME', else its first <rule>). Cheap, no parse."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            txt = fh.read()
    except OSError:
        return None
    import re
    m = re.search(r"^#\s*Grammar\s*:\s*(\S+)", txt, re.M)
    if m:
        return m.group(1).strip()
    m = re.search(r"<([A-Za-z_]\w*)>\s*::=", txt)
    return m.group(1) if m else None


def _grammars_for_model(model, log):
    """Per-model config: resolve the grammar file(s) a model was built with, from its
    ``<model>.state.json`` playbook roots — so ``--model X`` alone loads the right grammar(s)
    without re-specifying ``--grammar``."""
    if not model:
        return []
    st = os.path.join(base._REPO, "models", "generated", "transformer", model, model + ".state.json")
    if not os.path.isfile(st):
        return []
    try:
        with open(st, "r", encoding="utf-8") as fh:
            roots = list((json.load(fh).get("playbook") or {}).keys())
    except (OSError, ValueError):
        return []
    name_to_file = {}
    for f in base._list_grammar_files():
        n = _grammar_name_of(os.path.join(base._GRAMMAR_DIR, f))
        if n:
            name_to_file.setdefault(n, f)
    files = [name_to_file[r] for r in roots if r in name_to_file]
    if files:
        log.log("info", "NOCODE", "per-model config: %s.state.json -> grammar(s) %s"
                % (model, ", ".join(files)))
    return files


def _setup_host(cfg, log):
    """Build the host backend for one OR MORE grammars. Multiple grammars are merged into one
    playbook + command set (and an owner map) so a composing grammar — e.g.
    ``<combo> ::= <fibonacci> <greeting>`` — can descend into its child grammars and source each
    body under the namespace it was trained with. Returns host_ctx or None."""
    if not cfg.get("model"):
        log.log("error", "NOCODE", "model is required (host path) — /set model <name>")
        return None
    if not cfg.get("grammar"):
        log.log("error", "NOCODE", "grammar is required — /set grammar <file> [file ...]")
        return None
    grammars = cfg["grammar"] if isinstance(cfg["grammar"], (list, tuple)) else [cfg["grammar"]]
    from classes.class_model_grammar import ModelGrammar          # lazy (ML chain)
    from classes.class_model_assets import ModelAssets

    merged, names, owners, commands = {}, [], {}, {}
    mode, eval_token = "execute", "evaluate"
    for g in grammars:
        gp = _resolve_grammar(g)
        gf = ModelGrammar.load_file(gp, logger=log)
        if not gf:
            log.log("error", "NOCODE", "could not load grammar '%s'" % gp)
            continue
        names.append(gf["name"])
        sub = gf["tree"].get(gf["name"]) if isinstance(gf["tree"].get(gf["name"]), dict) else gf["tree"]
        for rule in sub:
            owners.setdefault(rule, gf["name"])          # rule/token -> owning grammar (first wins)
        ModelAssets.merge_tree(merged, sub)
        c, m, et = _load_nocode_vocab(gp, log)
        commands.update(c)
        if m in ("evaluate", "evaluate_ops"):
            mode, eval_token = m, et
    if not names:
        log.log("error", "NOCODE", "no grammar loaded")
        return None

    rules = list(merged.keys())
    gname = names[0]
    start_rule = gname if gname in merged else (rules[0] if rules else "expr")
    qfn = base.ollama_query_fn(cfg["model"], cfg.get("host"))

    def make_runner():
        return NoCodeGrammarRunner(
            grammar_name=gname, query_fn=qfn, fallback_playbook=merged, logger=log,
            commands=commands, policy=cfg["policy"], mode=mode, eval_token=eval_token, owners=owners,
        )

    log.log("ok", "NOCODE", "host ready — grammar(s)=%s mode=%s policy=%s oracle='%s'"
            % (",".join(names), mode, cfg["policy"], cfg["model"]))
    return {"gname": gname, "names": names, "playbook": merged, "rules": rules,
            "start_rule": start_rule, "mode": mode, "make_runner": make_runner}


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


def _install_nocode_completer(host_ctx):
    """TAB completion + persistent prompt history (readline). Completes nocode commands, grammar /
    rule / token names, /policy values, /set keys, and one level deeper into a rule's references."""
    try:
        import readline, atexit, re
        rules = host_ctx["rules"]
        playbook = host_ctx["playbook"]
        cmds = ["/?", "/help", "/policy", "/grammar", "/set", "/create", "/export", "/bye", "exit", "quit"]

        def candidates(buf):
            toks = buf.split()
            if (not toks) or (len(toks) == 1 and not buf.endswith(" ")):
                return cmds + rules
            head = toks[0].lower()
            if head == "/policy":
                return list(CodeExecPolicy.ALL)
            if head == "/set":
                if len(toks) == 1 or (len(toks) == 2 and not buf.endswith(" ")):
                    return ["model", "grammar", "host", "policy"]
                if len(toks) >= 2 and toks[1].lower() == "policy":
                    return list(CodeExecPolicy.ALL)
                if len(toks) >= 2 and toks[1].lower() == "grammar":
                    return base._list_grammar_files()
                return []
            if head == "/grammar":
                return base._list_grammar_files()
            if toks[0] in playbook:                              # one level deeper: rule's references
                return re.findall(r"[a-zA-Z_]\w*", str(playbook[toks[0]]))
            return []

        readline.set_completer_delims(" \t\n")
        histfile = os.path.join(os.path.expanduser("~"), ".nocode_runner_history")
        try:
            readline.read_history_file(histfile)
        except OSError:
            pass
        readline.set_history_length(2000)
        atexit.register(readline.write_history_file, histfile)

        def _c(text, state):
            opts = [c for c in candidates(readline.get_line_buffer()) if c.startswith(text)]
            return opts[state] if state < len(opts) else None

        readline.set_completer(_c)
        readline.parse_and_bind("tab: complete")
    except Exception:                                            # noqa: BLE001
        pass


def _repl(cfg, host_ctx, log):
    _install_nocode_completer(host_ctx)
    log.log("info", "NOCODE", "mode=%s policy=%s — type input (TAB completes, up/down history), /? help, /bye quit"
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
                cfg["grammar"] = parts[1:]              # one or more grammar files (merged/composed)
                return "reload"
            print("  loaded: %s" % ", ".join(host_ctx.get("names", [gname])))
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
    explicit_grammar = bool(config and config.get("grammar"))
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    cfg["policy"] = CodeExecPolicy.coerce(cfg.get("policy"))

    if mode == "device":
        log.log("warning", "NOCODE",
                "device/NPU path is deferred in the nocode track — use host mode for now")
        return

    # default model (CLI --model > config-file model > script-named default)
    if not cfg.get("model"):
        cfg["model"] = _default_model_name()
        log.log("info", "NOCODE", "no model specified — default (script name): '%s'" % cfg["model"])
    # per-model grammar resolution: explicit --grammar > the model's own state.json > config default
    if not explicit_grammar and cfg.get("model"):
        _gm = _grammars_for_model(cfg["model"], log)
        if _gm:
            cfg["grammar"] = _gm

    while True:
        host_ctx = _setup_host(cfg, log)
        if not host_ctx:
            return
        nxt = _repl(cfg, host_ctx, log)
        if nxt != "reload":
            return
