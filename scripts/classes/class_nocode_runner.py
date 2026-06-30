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


def _grammar_meta(merged, commands, owners, exec_rules, names):
    """For every callable rule/token: its ARGUMENT signature (parsed from ``args[N]`` / ``name =
    args[N]`` in the body) and the SUB-FUNCTIONS it calls. Powers the TAB detail display so the user
    sees ``revshell_param  <ip> <port>`` and the callable sub-paths a grammar exposes."""
    import re

    def arg_sig(body):
        body = body if isinstance(body, str) else ""
        named = {}
        for mm in re.finditer(r"(\w+)\s*=\s*(?:int|str|float)?\(?\s*args\[(\d+)\]", body):
            named.setdefault(int(mm.group(2)), mm.group(1))
        idxs = [int(i) for i in re.findall(r"args\[(\d+)\]", body)]
        if not idxs:
            return "<args...>" if re.search(r"\bargs\b", body) else ""
        return " ".join("<%s>" % named.get(i, "arg%d" % i) for i in range(max(idxs) + 1))

    callable_set = set(names) | set(exec_rules) | set(commands)
    meta = {}
    for name in callable_set:
        refs, rb = [], merged.get(name)
        if rb is not None:
            for tok in re.findall(r"[A-Za-z_]\w+", str(rb)):
                if tok in callable_set and tok != name and tok not in refs:
                    refs.append(tok)
        meta[name] = {"args": arg_sig(commands.get(name, "") or ""),
                      "subs": refs, "owner": owners.get(name, "")}
    for name in meta:                                # a root inherits the richest sig of its sub-fns
        sigs = [meta[name]["args"]] + [meta[s]["args"] for s in meta[name]["subs"] if s in meta]
        sigs = [s for s in sigs if s]
        if sigs:
            meta[name]["args"] = max(sigs, key=lambda s: s.count("<"))
    return meta


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
    defines, grammar_tokens, vocab_by_owner = {}, {}, {}
    mode, eval_token = "execute", "evaluate"
    exec_rules, eval_gname, eval_start = set(), None, None
    for g in grammars:
        gp = _resolve_grammar(g)
        gf = ModelGrammar.load_file(gp, logger=log)
        if not gf:
            log.log("error", "NOCODE", "could not load grammar '%s'" % gp)
            continue
        gname_g = gf["name"]
        names.append(gname_g)
        sub = gf["tree"].get(gf["name"]) if isinstance(gf["tree"].get(gf["name"]), dict) else gf["tree"]
        sub_keys = list(sub.keys())
        grammar_tokens.setdefault(gname_g, set()).update(sub_keys)
        for rule in sub:
            owners.setdefault(rule, gname_g)             # canonical (first-declared) owner
            if gname_g not in defines.setdefault(rule, []):
                defines[rule].append(gname_g)            # EVERY grammar that defines it (collision detection)
        ModelAssets.merge_tree(merged, sub)
        c, m, et = _load_nocode_vocab(gp, log)
        commands.update(c)                               # flat side-car (keeps _exec; single-grammar fallback)
        for tok, body in c.items():                      # owner-qualified verified bodies
            if not tok.startswith("_") and isinstance(body, str):
                vocab_by_owner[(gname_g, tok)] = body
        if m in ("evaluate", "evaluate_ops"):            # an expression grammar (evaluate per input)
            mode, eval_token = m, et
            eval_gname = eval_gname or gname_g
            eval_start = eval_start or (sub_keys[0] if sub_keys else gname_g)
        else:                                            # a procedure grammar (run via execute())
            exec_rules.update(sub_keys)
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
            defines=defines, grammar_tokens=grammar_tokens, vocab_by_owner=vocab_by_owner,
        )

    meta = _grammar_meta(merged, commands, owners, exec_rules, names)
    callable_names = sorted(set(names) | exec_rules | set(commands))

    log.log("ok", "NOCODE", "host ready — grammar(s)=%s mode=%s policy=%s oracle='%s'"
            % (",".join(names), mode, cfg["policy"], cfg["model"]))
    return {"gname": gname, "names": names, "playbook": merged, "rules": rules,
            "start_rule": start_rule, "mode": mode, "make_runner": make_runner,
            "exec_rules": exec_rules, "eval_gname": eval_gname, "eval_start": eval_start,
            "meta": meta, "callable": callable_names}


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
    """TAB completion + persistent history (readline). Completes nocode commands, grammar roots AND
    their callable sub-functions, /policy values, /set keys. On TAB it ALSO prints each candidate's
    DETAIL — argument signature (e.g. ``<ip> <port>``) and the sub-functions a grammar calls — so the
    user discovers how to invoke a command and which sub-paths can be called specifically."""
    try:
        import readline, atexit, re, sys
        rules = host_ctx["rules"]
        playbook = host_ctx["playbook"]
        meta = host_ctx.get("meta") or {}
        callable_names = host_ctx.get("callable") or rules
        cmds = ["/?", "/help", "/policy", "/grammar", "/set", "/create", "/export", "/bye", "exit", "quit"]

        def _detail(name):
            m = meta.get(name.split("/")[-1] if "/" in name else name)
            if not m:
                return ""
            bits = []
            if m["args"]:
                bits.append("args " + m["args"])
            if m["subs"]:
                bits.append("→ " + ", ".join(m["subs"]))
            if m.get("owner") and m["owner"] != name:
                bits.append("[%s]" % m["owner"])
            return "   ".join(bits)

        def candidates(buf):
            toks = buf.split()
            if (not toks) or (len(toks) == 1 and not buf.endswith(" ")):
                cur = toks[0] if toks else ""
                if "/" in cur:                            # path: complete a grammar's sub-functions
                    g = cur.split("/", 1)[0]
                    return ["%s/%s" % (g, s) for s in (meta.get(g) or {}).get("subs", [])]
                paths = ["%s/" % n for n in callable_names if (meta.get(n) or {}).get("subs")]
                return cmds + callable_names + paths
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
            if toks[0] in meta and meta[toks[0]]["subs"]:        # deeper: the grammar's callable sub-fns
                return meta[toks[0]]["subs"]
            if toks[0] in playbook:                              # fallback: rule's references
                return re.findall(r"[a-zA-Z_]\w*", str(playbook[toks[0]]))
            return []

        def _display(substitution, matches, longest):           # noqa: ARG001 — readline hook signature
            cmd_hits, groups = [], {}
            for mm in matches:                                   # group candidates BY GRAMMAR (owner)
                if mm in cmds:
                    cmd_hits.append(mm)
                    continue
                key = mm.split("/")[-1] if "/" in mm else mm
                owner = (meta.get(key) or {}).get("owner") or key
                groups.setdefault(owner, []).append(mm)
            sys.stdout.write("\n")
            if cmd_hits:
                sys.stdout.write("  \033[2mcommands\033[0m   " + "  ".join(sorted(cmd_hits)) + "\n")
            for g in sorted(groups):
                sys.stdout.write("  \033[1m%s\033[0m\n" % g)
                for mm in sorted(groups[g], key=lambda x: (x.split("/")[-1] != g, x)):  # root first
                    d = _detail(mm)
                    sys.stdout.write(("    %-22s %s\n" % (mm, d)) if d else ("    %s\n" % mm))
            sys.stdout.write("nocode> " + readline.get_line_buffer())
            sys.stdout.flush()
            readline.redisplay()

        readline.set_completer_delims(" \t\n")
        readline.set_completion_display_matches_hook(_display)
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

        # ---- bare input: AUTO-ROUTE per input — a procedure NAME -> execute; an expression ->
        # evaluate. Mode-agnostic, so a mixed model (e.g. combo procedures + calculator) serves both.
        # A "<grammar>/<function>" path runs that function specifically, under the grammar's namespace
        # (disambiguates a token shared by two grammars). Bare "<function>" works too. ----
        first = parts[0]
        owner_override = None
        if "/" in first:
            gpart, _, fn = first.partition("/")
            if fn and (fn in host_ctx["exec_rules"] or fn in (host_ctx.get("meta") or {})):
                owner_override, first = gpart, fn
        eval_start = host_ctx.get("eval_start")
        if first in host_ctx["exec_rules"]:
            runner = host_ctx["make_runner"]()
            if owner_override:
                runner.owners[first] = owner_override          # fetch the body under this grammar
            runner.execute(start_rule=first, args=parts[1:])
        elif eval_start and NoCodeGrammarRunner.probe(
                host_ctx.get("eval_gname") or gname, ui, host_ctx["playbook"], start_rule=eval_start):
            host_ctx["make_runner"]().run(ui, start_rule=eval_start)
        else:
            log.log("warning", "NOCODE", "not a valid command or expression: '%s'" % ui)


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
