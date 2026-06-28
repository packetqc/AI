"""class_model_runner.py - Unified runner (host / device) for grammar-LM solutions.

Pioneer "nocode" grammar-LM line (dates from file/git history):
  2026-06-06  fsm_language_model.py            genesis - FSM language model
  2026-06-14  model_create_hugging_face.py     nocode model-from-grammar generation
  2026-06-16  model_create_hf_cl.py            interactive grammar client
  2026-06-27  class_model_runner.py            unified host/device runner (this file)

One interactive client, two execution modes that differ by WHERE the CPU logic runs:

  mode=device  the STM32N6 is AUTONOMOUS. The runner is a thin serial terminal: it pushes the
               prompt over the ST-Link VCP and collects the output; the device's own CPU code
               (C++ GrammarRunner) tokenizes, parses, evaluates and drives its Neural-ART NPU
               entirely on-chip. This proves the edge device does the job by itself.

  mode=host    the HOST does the CPU logic. The runner runs GrammarRunner locally and queries an
               Ollama chat model as the grammar oracle, then parses + evaluates on the host.

Device mode has NO ML dependencies (pure serial) — the ML chain is imported lazily only for host
mode. Run device mode with the system python; run host mode from the project venv.

Usage:
  python3 class_model_runner.py --mode device --port /dev/ttyACM0
  python3 class_model_runner.py --mode host   --model model_calculator_test_npu
"""
import os
import sys
import json
import time
import select
import subprocess
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_GRAMMAR = os.path.join(os.path.dirname(_HERE), "..",
                                "models", "grammars", "playbook_model_calculator.txt")


def _logger():
    """Lightweight colour logger (no ML deps)."""
    sys.path.insert(0, os.path.dirname(_HERE))
    from classes.class_terminal_logs import TerminalLogger
    return TerminalLogger()


def _install_completer(words):
    try:
        import readline
        def _c(text, state):
            opts = [w for w in words if w.startswith(text)]
            return opts[state] if state < len(opts) else None
        readline.set_completer(_c)
        readline.parse_and_bind("tab: complete")
    except Exception:                                            # noqa: BLE001
        pass


# ----------------------------------------------------------------------------
# DEVICE mode - thin serial terminal to the autonomous on-chip calculator
# ----------------------------------------------------------------------------
class DeviceTerminal:
    """Push a prompt to the device, collect its output. The device does all the work."""

    PROMPT = "calc> "

    def __init__(self, port="/dev/ttyACM0", baud=115200, logger=None, boot_timeout=12):
        self.port = port
        self.logger = logger
        subprocess.run(["stty", "-F", port, str(baud), "raw", "-echo", "-echoe", "-echok"],
                       check=False)
        self.f = open(port, "r+b", buffering=0)
        # consume the boot banner up to the first prompt (best-effort)
        self._read_until_prompt(boot_timeout, echo=False)

    def _read_until_prompt(self, timeout, echo=True):
        buf = bytearray()
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.f], [], [], max(0.0, end - time.time()))
            if not r:
                break
            c = self.f.read(1)
            if not c:
                continue
            buf += c
            if buf.endswith(self.PROMPT.encode()):
                break
        text = buf.decode(errors="replace")
        # drop the trailing prompt itself
        if text.endswith(self.PROMPT):
            text = text[:-len(self.PROMPT)]
        return text

    def eval_expr(self, expr, timeout=30):
        self.f.write((expr + "\r").encode())
        return self._read_until_prompt(timeout)


def _print_grammar(grammar_file):
    try:
        with open(grammar_file, "r", encoding="utf-8") as fh:
            sys.stdout.write(fh.read())
    except OSError as e:                                         # noqa: BLE001
        print("(grammar unavailable: %s)" % e)


def ollama_query_fn(model, host=None):
    """HOST mode oracle: query an Ollama chat model over the local API."""
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def q(prompt):
        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode()).get("response", "").strip()
    return q


# ----------------------------------------------------------------------------
# Runtime configuration — single source of truth, settable from CLI or /set.
# Runtime keys drive the runner itself; builder keys (None = use the sub-script's
# own default) are forwarded as argv to model_create_npu_tcn.py by /create | /export.
# ----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # runtime — used by the runner itself
    "grammar":      _DEFAULT_GRAMMAR,   # also forwarded to the builder as --grammar
    "port":         "/dev/ttyACM0",     # device mode
    "baud":         115200,             # device mode
    "boot_timeout": 12,                 # device mode
    "model":        None,               # host mode (Ollama model name)
    "host":         None,               # host mode (Ollama URL)
    # builder — model_create_npu_tcn.py (None = the sub-script's own default)
    "name":      None,
    "version":   None,
    "tokenizer": None,
    "embed_dim": None,
    "seq_len":   None,
    "kernel":    None,
    "epochs":    None,
    "lr":        None,
    "out_dir":   None,
    "npu_dir":   None,
    "stedgeai":  None,
    "target":    None,
    "net_name":  None,
    "c_api":     None,
    "opset":     None,
}

_CFG_TYPES = {"baud": int, "boot_timeout": int, "embed_dim": int, "seq_len": int,
              "kernel": int, "epochs": int, "lr": float, "opset": int}
_MODE_KEYS = {"device": ("port", "baud", "boot_timeout"), "host": ("model", "host")}
_RUNTIME_KEYS = ("grammar", "port", "baud", "boot_timeout", "model", "host")
_BUILDER_DISPLAY = ("name", "version", "tokenizer", "embed_dim", "seq_len", "kernel",
                    "epochs", "lr", "out_dir", "npu_dir", "stedgeai",
                    "target", "net_name", "c_api", "opset")
_BUILDER_FLAGS = {
    "name": "--name", "version": "--version", "tokenizer": "--tokenizer",
    "grammar": "--grammar", "embed_dim": "--embed-dim", "seq_len": "--seq-len",
    "kernel": "--kernel", "epochs": "--epochs", "lr": "--lr",
    "out_dir": "--out-dir", "npu_dir": "--npu-dir", "stedgeai": "--stedgeai",
    "target": "--target", "net_name": "--net-name", "c_api": "--c-api", "opset": "--opset",
}
_BUILDER_SCRIPT = os.path.join(os.path.dirname(_HERE), "model_generation", "model_create_npu_tcn.py")
_SECURITY_SCRIPT = os.path.join(os.path.dirname(_HERE), "model_security_re.py")
_SECURITY_SUBCMDS = ("analyze", "reconstruct", "threat", "integrity")


# ----------------------------------------------------------------------------
# shared command handlers (config + builder) — available in every mode
# ----------------------------------------------------------------------------
def _print_help(mode):
    print("  /?              this help")
    print("  /mode           show the active mode + key config")
    print("  /config         show all config        (/get <key> for one value)")
    print("  /set <k> <v>    set a config value      (runtime or builder key)")
    print("  /grammar        print the BNF grammar")
    print("  /create         build: train + export the NPU-native TCN (model_create_npu_tcn.py)")
    print("  /export         build: re-export only   (--export-only)")
    print("  /security [sub] blackbox model analysis (model_security_re.py: analyze|reconstruct|threat|integrity)")
    if mode == "host":
        print("  /rules          recall every grammar rule via the Ollama oracle   [host]")
    print("  /bye            quit the runner" + (" (device stays autonomous)" if mode == "device" else ""))
    if mode == "device":
        print("  <expression>    pushed to the device; it computes on-chip and returns the output")
    else:
        print("  <expression>    parsed + evaluated locally; rules recalled from the model")


def _show_mode(log, mode, cfg):
    if mode == "device":
        log.log("info", "SYSTEM", "mode=device (autonomous)  port=%s  baud=%s"
                % (cfg.get("port"), cfg.get("baud")))
    else:
        log.log("info", "SYSTEM", "mode=host  model=%s  host=%s"
                % (cfg.get("model"), cfg.get("host") or "default"))


def _show_config(cfg, mode, args):
    if args:                                                    # /get <key>
        k = args[0].lower()
        print("  %s = %s" % (k, cfg.get(k) if k in cfg else "(unknown key)"))
        return
    print("  === config (mode=%s) ===" % mode)
    print("  -- runtime --")
    for k in _RUNTIME_KEYS:
        tag = ""
        for m, keys in _MODE_KEYS.items():
            if k in keys and m != mode:
                tag = "   (%s mode)" % m
        print("    %-13s %s%s" % (k, cfg.get(k), tag))
    print("  -- builder (model_create_npu_tcn.py; unset = script default) --")
    for k in _BUILDER_DISPLAY:
        v = cfg.get(k)
        print("    %-13s %s" % (k, v if v is not None else "(default)"))


def _set_config(log, cfg, args):
    if len(args) < 2:
        log.log("warning", "CONFIG", "usage: /set <key> <value>   (keys: %s)" % ", ".join(sorted(cfg)))
        return
    key = args[0].lower()
    val = " ".join(args[1:])
    if key not in cfg:
        log.log("warning", "CONFIG", "unknown key '%s' (known: %s)" % (key, ", ".join(sorted(cfg))))
        return
    caster = _CFG_TYPES.get(key)
    if caster:
        try:
            val = caster(val)
        except ValueError:
            log.log("error", "CONFIG", "'%s' expects %s" % (key, caster.__name__))
            return
    cfg[key] = val
    log.log("ok", "CONFIG", "%s = %s" % (key, val))


def _run_builder(log, cfg, export_only=False):
    """Invoke model_create_npu_tcn.py with the configured keys as argv. Runs as a
    subprocess so device mode stays dependency-free (no in-process ML import)."""
    if not os.path.isfile(_BUILDER_SCRIPT):
        log.log("error", "BUILD", "builder not found: %s" % _BUILDER_SCRIPT)
        return
    argv = [sys.executable, _BUILDER_SCRIPT]
    for key, flag in _BUILDER_FLAGS.items():
        v = cfg.get(key)
        if v is not None:
            argv += [flag, str(v)]
    if export_only:
        argv.append("--export-only")
    log.log("info", "BUILD", ("export-only" if export_only else "create+export") + ": " + " ".join(argv[1:]))
    try:
        rc = subprocess.run(argv).returncode
        log.log("ok" if rc == 0 else "error", "BUILD", "builder exit code %d" % rc)
    except Exception as e:                                       # noqa: BLE001
        log.log("error", "BUILD", "failed to run builder: %s" % e)


def _run_security(log, cfg, args):
    """Invoke model_security_re.py (blackbox model analysis). Subprocess keeps device
    mode dependency-free (gguf-py / Ollama are only needed in the child process).

    Usage:  /security [subcommand] [extra args...]
      subcommand defaults to 'analyze' (analyze | reconstruct | threat | integrity).
      Target defaults to the configured 'model' as --ollama unless --ollama/--gguf given.
      Any extra tokens pass straight through (e.g. --dynamic, --gguf <path>, --out <dir>).
    """
    if not os.path.isfile(_SECURITY_SCRIPT):
        log.log("error", "SECURITY", "tool not found: %s" % _SECURITY_SCRIPT)
        return
    args = list(args)
    if args and not args[0].startswith("-") and args[0].lower() in _SECURITY_SUBCMDS:
        subcmd = args.pop(0).lower()
    else:
        subcmd = "analyze"
    if "--ollama" not in args and "--gguf" not in args:
        if cfg.get("model"):
            args = ["--ollama", cfg["model"]] + args
        else:
            log.log("warning", "SECURITY",
                    "no target — set the Ollama model (/set model <name>) or pass --gguf <path> / --ollama <name>")
            return
    argv = [sys.executable, _SECURITY_SCRIPT, subcmd] + args
    log.log("info", "SECURITY", "%s: %s" % (subcmd, " ".join(argv[1:])))
    try:
        rc = subprocess.run(argv).returncode
        log.log("ok" if rc == 0 else "error", "SECURITY", "model_security_re exit code %d" % rc)
    except Exception as e:                                       # noqa: BLE001
        log.log("error", "SECURITY", "failed to run model_security_re: %s" % e)


# ----------------------------------------------------------------------------
# unified REPL — one client, commands gated by the active mode
# ----------------------------------------------------------------------------
def run(mode, config=None, logger=None):
    log = logger or _logger()
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if k in cfg and v is not None})
    grammar_file = cfg.get("grammar") or _DEFAULT_GRAMMAR

    device = None
    host_ctx = None
    if mode == "device":
        log.log("info", "RUNNER", "device mode - connecting to autonomous STM32N6 on %s ..." % cfg["port"])
        try:
            device = DeviceTerminal(cfg["port"], baud=int(cfg["baud"]), logger=log,
                                    boot_timeout=int(cfg["boot_timeout"]))
        except Exception as e:                                   # noqa: BLE001
            log.log("error", "RUNNER", "cannot open %s: %s" % (cfg["port"], e))
            return 1
        log.log("ok", "RUNNER", "device ready - it parses + evaluates + runs the NPU on-chip")
    else:
        if not cfg.get("model"):
            log.log("error", "RUNNER", "model is required for host mode (--model or /set model <name>)")
            return 1
        sys.path.insert(0, os.path.dirname(_HERE))
        from classes.class_model_grammar import ModelGrammar, GrammarRunner   # lazy (pulls gguf chain)
        gf = ModelGrammar.load_file(grammar_file, logger=log)
        if not gf:
            log.log("error", "RUNNER", "could not load grammar '%s'" % grammar_file)
            return 1
        gname = gf["name"]
        tree = gf["tree"]
        playbook = tree[gname] if isinstance(tree.get(gname), dict) else tree
        rules = list(playbook.keys())
        start_rule = rules[0] if rules else "expr"
        qfn = ollama_query_fn(cfg["model"], cfg.get("host"))

        def make_runner():
            return GrammarRunner(grammar_name=gname, query_fn=qfn,
                                 fallback_playbook=playbook, logger=log)

        host_ctx = {"gname": gname, "playbook": playbook, "rules": rules,
                    "start_rule": start_rule, "make_runner": make_runner,
                    "GrammarRunner": GrammarRunner}
        log.log("info", "RUNNER", "host mode - GrammarRunner local, oracle = Ollama '%s'" % cfg["model"])

    base = ["/?", "/help", "/bye", "/mode", "/grammar", "/config", "/get", "/set",
            "/create", "/export", "/security", "exit", "quit"]
    extra = (["/rules"] + host_ctx["rules"]) if host_ctx else []
    _install_completer(base + extra)
    log.log("info", "SYSTEM", "type an expression (e.g. 3 + 4), /? for help, /bye to quit")

    while True:
        try:
            ui = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not ui:
            continue
        parts = ui.split()
        cmd = parts[0].lower()
        low = ui.lower()

        if low in ("/bye", "exit", "quit"):
            log.log("ok", "SYSTEM", "closing runner"
                    + (" - the device keeps running standalone" if mode == "device" else ""))
            break
        if cmd in ("/?", "/help"):
            _print_help(mode); continue
        if cmd == "/mode":
            _show_mode(log, mode, cfg); continue
        if cmd in ("/config", "/get"):
            _show_config(cfg, mode, parts[1:]); continue
        if cmd == "/set":
            _set_config(log, cfg, parts[1:]); continue
        if cmd == "/grammar":
            if host_ctx:
                for r in host_ctx["rules"]:
                    print("  <%s> ::= %s" % (r, host_ctx["playbook"][r]))
            else:
                _print_grammar(cfg.get("grammar") or grammar_file)
            continue
        if cmd in ("/create", "/export"):
            _run_builder(log, cfg, export_only=(cmd == "/export")); continue
        if cmd == "/security":
            _run_security(log, cfg, parts[1:]); continue
        if cmd == "/rules":
            if not host_ctx:
                log.log("warning", "SYSTEM", "/rules is available in host mode only")
                continue
            r = host_ctx["make_runner"]()
            for rule_name in host_ctx["rules"]:
                r._query_rule(rule_name)
            continue

        # bare input -> an expression for the active backend
        if mode == "device":
            out = device.eval_expr(ui).strip("\r\n")
            if out:
                print(out)
        else:
            hc = host_ctx
            if hc["GrammarRunner"].probe(hc["gname"], ui, hc["playbook"], start_rule=hc["start_rule"]):
                hc["make_runner"]().run(ui, start_rule=hc["start_rule"])
            else:
                log.log("warning", "RUNNER",
                        "not a valid %s expression: '%s' (try: 3 + 4)" % (hc["gname"], ui))
    return 0


# ----------------------------------------------------------------------------
# backward-compatible thin wrappers (older callers / direct imports)
# ----------------------------------------------------------------------------
def run_device(port="/dev/ttyACM0", grammar_file=_DEFAULT_GRAMMAR):
    return run("device", {"port": port, "grammar": grammar_file})


def run_host(grammar_file=_DEFAULT_GRAMMAR, model=None, host=None):
    return run("host", {"grammar": grammar_file, "model": model, "host": host})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Unified grammar runner (autonomous device / host Ollama).")
    ap.add_argument("--mode", choices=["host", "device"], default="device",
                    help="device: autonomous STM32N6 over serial · host: GrammarRunner + Ollama")
    ap.add_argument("--grammar", default=None, help="BNF/EBNF playbook grammar file")
    ap.add_argument("--port", default=None, help="device serial port (device mode)")
    ap.add_argument("--model", default=None, help="Ollama model name (host mode)")
    ap.add_argument("--host", default=None, help="Ollama host URL (host mode)")
    a = ap.parse_args()
    sys.exit(run(a.mode, {k: v for k, v in vars(a).items() if k != "mode" and v is not None}))
