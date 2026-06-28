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
_REPO = os.path.dirname(os.path.dirname(_HERE))            # scripts/classes -> scripts -> repo root
_GRAMMAR_DIR = os.path.join(_REPO, "models", "grammars")  # structural; the grammar filename is user-managed
_TRAINING_DIR = os.path.join(_REPO, "models", "training") # structural; training filename(s) declared in grammar meta


def _logger():
    """Lightweight colour logger (no ML deps)."""
    sys.path.insert(0, os.path.dirname(_HERE))
    from classes.class_terminal_logs import TerminalLogger
    return TerminalLogger()


def _list_grammar_files():
    try:
        return [f for f in sorted(os.listdir(_GRAMMAR_DIR))
                if f.startswith("playbook_") and f.endswith(".txt")]
    except OSError:
        return []


def _dir_names(base):
    """Immediate sub-directory names under base (bare names — for name atoms)."""
    try:
        return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    except OSError:
        return []


def _rel_dirs(base):
    """Immediate sub-directories under base as repo-relative paths (folder-list assistance)."""
    try:
        return sorted(os.path.relpath(os.path.join(base, d), _REPO)
                      for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    except OSError:
        return []


def _rel_files(base, suffix):
    import glob
    return sorted(os.path.relpath(p, _REPO)
                  for p in glob.glob(os.path.join(base, "**", "*" + suffix), recursive=True)
                  if os.path.isfile(p))


def _top_files(base, suffix):
    """Top-level files of a type directly under base (repo-relative) — narrow assistance."""
    try:
        return sorted(os.path.relpath(os.path.join(base, f), _REPO)
                      for f in os.listdir(base)
                      if f.endswith(suffix) and os.path.isfile(os.path.join(base, f)))
    except OSError:
        return []


def _cur(cfg, key):
    """Current value of a config key as a single-item hint list (for free values)."""
    v = (cfg or {}).get(key)
    return [str(v)] if v not in (None, "") else []


def _value_candidates(key, cfg=None):
    """Tab assistance for a config VALUE by key — always a file list, folder list, or known list;
    free numeric/text keys fall back to the current value as a hint, so every key assists."""
    key = key.lower()
    if key == "mode":
        return ["device", "host"]
    if key == "sec_dynamic":
        return ["true", "false"]
    if key == "grammar":
        return _list_grammar_files()
    if key == "model":
        return _ollama_models() or _cur(cfg, key)
    if key == "name":
        return _dir_names(os.path.join(_REPO, "models", "generated", "convolutional")) or _cur(cfg, key)
    if key == "tokenizer":
        return _dir_names(os.path.join(_REPO, "models", "generated", "transformer")) or _cur(cfg, key)
    if key == "port":
        return _list_serial_ports() or _cur(cfg, key)
    if key == "target":
        return ["stm32n6"]
    if key == "net_name":
        return ["network"]
    if key == "c_api":
        return ["st-ai"]
    if key == "host":
        return ["http://localhost:11434"]
    if key == "sec_gguf":
        return _rel_files(os.path.join(_REPO, "models"), ".gguf") or _cur(cfg, key)
    if key == "sec_registry":
        return _top_files(os.path.join(_REPO, "models"), ".json") or _cur(cfg, key)
    if key == "sec_out":
        return _rel_dirs(os.path.join(_REPO, "models", "forensics")) or _cur(cfg, key)
    if key == "out_dir":
        return _rel_dirs(os.path.join(_REPO, "models", "generated", "convolutional")) or _cur(cfg, key)
    if key == "npu_dir":
        return _rel_dirs(os.path.join(_REPO, "models", "npu_export")) or _cur(cfg, key)
    if key == "sec_assets":
        return _rel_dirs(os.path.join(_REPO, "models")) or _cur(cfg, key)
    # free numeric/text values (version, epochs, lr, opset, baud, boot_timeout, embed_dim, ...)
    return _cur(cfg, key)


def _complete_candidates(buf, commands, cfg, security_subcmds, grammar_names):
    """Context-aware Tab candidates from the current line buffer."""
    toks = buf.split()
    first_word = (not toks) or (len(toks) == 1 and not buf.endswith(" "))
    if first_word:
        return list(commands) + list(grammar_names)
    head = toks[0].lower()
    if head == "/grammar":
        return _list_grammar_files()
    if head in ("/set", "/get"):
        completing_value = len(toks) >= 2 and (len(toks) > 2 or buf.endswith(" "))
        if completing_value:
            # Tab on a value = intention to SET (value assistance); GET is a read command.
            return _value_candidates(toks[1], cfg) if head == "/set" else []
        return sorted(cfg)
    if head == "/security":
        return list(security_subcmds)
    return []


def _install_completer(commands, cfg=None, security_subcmds=(), grammar_names=()):
    """In-depth Tab completion + persistent command history.
       first token -> commands (+ grammar rule names in host mode); /set·/get -> config keys
       (grammar filenames after 'grammar'); /security -> subcommands."""
    try:
        import readline, atexit
        readline.set_completer_delims(" \t\n")   # only whitespace breaks, so '/set' etc. complete
        histfile = os.path.join(os.path.expanduser("~"), ".model_runner_history")
        try:
            readline.read_history_file(histfile)
        except OSError:
            pass
        readline.set_history_length(2000)
        atexit.register(readline.write_history_file, histfile)

        def _c(text, state):
            cands = _complete_candidates(readline.get_line_buffer(), commands,
                                         cfg or {}, security_subcmds, grammar_names)
            opts = [c for c in cands if c.startswith(text)]
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

    def close(self):
        try:
            self.f.close()
        except Exception:                                       # noqa: BLE001
            pass


def _ollama_models():
    """List locally available Ollama model names (listing assistance for host mode)."""
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return [ln.split()[0] for ln in out.stdout.splitlines()[1:] if ln.split()]
    except Exception:                                           # noqa: BLE001
        return []


def _ask(prompt, options=None, default=None):
    """Prompt for one value with optional listing assistance; Enter accepts the default."""
    if options:
        print("    available: " + ", ".join(options))
    suffix = (" [%s]" % default) if default not in (None, "") else ""
    try:
        ans = input("    %s%s: " % (prompt, suffix)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return ans or default


def prompt_setup(config=None, logger=None):
    """Interactive setup of the essential model-management values when the runner is started
    with no arguments: mode, model-or-grammar, name — with listing assistance. Everything else
    keeps its config-file default and stays settable later via /set · /get. Returns (mode, cfg)."""
    log = logger or _logger()
    provided = dict(config or {})
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({k: v for k, v in provided.items() if k in cfg and v is not None})
    print("== runner setup — Enter accepts [default]; CLI-provided values kept; all values /set-able later ==")
    # mode (skip if already provided on the CLI)
    if str(provided.get("mode", "")).lower() in ("device", "host"):
        mode = str(provided["mode"]).lower()
    else:
        mode = (_ask("mode", options=["device", "host"], default=cfg.get("mode") or "device") or "device").lower()
        if mode not in ("device", "host"):
            mode = "device"
    cfg["mode"] = mode
    # grammar (the model-or-grammar essential)
    if "grammar" not in provided:
        g = _ask("grammar", options=_list_grammar_files(), default=cfg.get("grammar"))
        if g:
            cfg["grammar"] = g
    _apply_grammar_derived(cfg, log)   # tokenizer auto-set from the grammar meta (still /set-able)
    # mode-specific essential
    if mode == "host":
        if "model" not in provided:
            m = _ask("model (Ollama)", options=(_ollama_models() or None), default=cfg.get("model"))
            if m:
                cfg["model"] = m
    else:
        if "port" not in provided:
            ports = _list_serial_ports()
            dflt = cfg.get("port")
            if ports and dflt not in ports:
                dflt = ports[0]
            p = _ask("port (device serial)", options=(ports or None), default=dflt)
            if p:
                cfg["port"] = p
    # name
    if "name" not in provided:
        n = _ask("name", default=cfg.get("name"))
        if n:
            cfg["name"] = n
    return mode, cfg


def _print_grammar(grammar_file):
    try:
        with open(grammar_file, "r", encoding="utf-8") as fh:
            sys.stdout.write(fh.read())
    except OSError as e:                                         # noqa: BLE001
        print("(grammar unavailable: %s)" % e)


def _grammar_meta(grammar_path):
    """Parse '# Key : value' meta directives from a grammar file (lightweight, no ML)."""
    meta = {}
    if not grammar_path:
        return meta
    try:
        with open(grammar_path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s.startswith("#") or ":" not in s:
                    continue
                head, val = s[1:].split(":", 1)
                key = head.strip()
                if key and " " not in key:
                    meta[key.lower()] = val.strip()
    except OSError:
        pass
    return meta


def _grammar_training(grammar_path):
    """Training file(s) declared by the grammar's '# Training :' meta (comma-separated list)."""
    raw = _grammar_meta(grammar_path).get("training", "")
    return [f.strip() for f in raw.split(",") if f.strip() and f.strip().lower() != "none"]


def _apply_grammar_derived(cfg, log):
    """On grammar selection, derive grammar-bound config from the grammar meta: the tokenizer is
    auto-set (still /set-overridable), and the training file(s) are surfaced. Mirrors how training
    is bound to the grammar — the association lives in the grammar, not in scattered config."""
    gpath = os.path.join(_GRAMMAR_DIR, cfg["grammar"]) if cfg.get("grammar") else None
    if not gpath:
        return
    meta = _grammar_meta(gpath)
    tok = meta.get("tokenizer")
    if tok and tok.lower() != "none":
        cfg["tokenizer"] = tok
        log.log("info", "SYSTEM", "tokenizer (from grammar meta): %s" % tok)
    tr = _grammar_training(gpath)
    if tr:
        log.log("info", "SYSTEM", "training (from grammar meta): %s" % ", ".join(tr))


def _load_command_vocab(grammar_path, log):
    """Auto-load the command-vocabulary (token -> command) from the training file(s) the grammar
    declares in its meta, so procedure-grammar tokens are executable in host mode. Atoms compose
    with the structural TRAINING_DIR; returns a commands dict for GrammarRunner."""
    cmds = {}
    for f in _grammar_training(grammar_path):
        p = f if (os.path.isabs(f) or os.sep in f) else os.path.join(_TRAINING_DIR, f)
        if not os.path.isfile(p):
            log.log("warning", "RUNNER", "grammar training file not found: %s" % p)
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (OSError, ValueError) as e:
            log.log("warning", "RUNNER", "could not read training file %s: %s" % (p, e))
            continue
        if isinstance(obj, dict) and obj.get("_type") == "command_vocabulary":
            n = 0
            for k, v in obj.items():
                if k == "_exec" or (not k.startswith("_") and isinstance(v, str)):
                    cmds[k] = v
                    n += 0 if k.startswith("_") else 1
            log.log("info", "RUNNER", "auto-loaded training '%s' (%d token(s))" % (os.path.basename(p), n))
    return cmds


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
# Default VALUES for runner/CLI-manageable keys live in the external config file
# (scripts/model_runner_config.json) — the single source of truth, shared with the
# worker scripts. NOTHING user-managed is hardcoded here. Architecture keys
# (embed_dim/seq_len/kernel) are intentionally left without a default — their value
# lives in the worker (model_create_npu_tcn.py); /set still overrides them.
_CONFIG_FILE = os.path.join(_REPO, "scripts", "model_runner_config.json")

_RUNTIME_KEYS = ("mode", "grammar", "port", "baud", "boot_timeout", "model", "host")
_ARCH_KEYS = ("embed_dim", "seq_len", "kernel")           # value lives in the worker, override-only
_BUILDER_OPS_KEYS = ("name", "version", "tokenizer", "epochs", "lr", "out_dir",
                     "npu_dir", "stedgeai", "target", "net_name", "c_api", "opset")
_BUILDER_DISPLAY = _BUILDER_OPS_KEYS + _ARCH_KEYS
_SECURITY_KEYS = ("sec_gguf", "sec_out", "sec_registry", "sec_assets", "sec_dynamic")
_ALL_KEYS = _RUNTIME_KEYS + _BUILDER_OPS_KEYS + _ARCH_KEYS + _SECURITY_KEYS


def _load_defaults():
    """Load manageable default VALUES from the config file (single source of truth)."""
    cfg = {k: None for k in _ALL_KEYS}
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as fh:
            cfg.update({k: v for k, v in json.load(fh).items() if k in cfg})
    except (OSError, ValueError):
        pass
    return cfg


DEFAULT_CONFIG = _load_defaults()

def _as_bool(s):
    return str(s).strip().lower() in ("1", "true", "yes", "on")


_CFG_TYPES = {"baud": int, "boot_timeout": int, "embed_dim": int, "seq_len": int,
              "kernel": int, "epochs": int, "lr": float, "opset": int,
              "sec_dynamic": _as_bool}
_MODE_KEYS = {"device": ("port", "baud", "boot_timeout"), "host": ("model", "host")}
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
    # training file(s) are derived from the grammar's '# Training :' meta — not a settable key
    _tr = _grammar_training(os.path.join(_GRAMMAR_DIR, cfg["grammar"])) if cfg.get("grammar") else []
    print("    %-13s %s" % ("training",
          (", ".join(_tr) + "   (from grammar meta)") if _tr else "(none — inline/expression grammar)"))
    print("  -- builder (model_create_npu_tcn.py) --")
    for k in _BUILDER_DISPLAY:
        v = cfg.get(k)
        if v is not None:
            shown = v
        elif k in _ARCH_KEYS:
            shown = "(architecture default — in worker)"
        else:
            shown = "(from config file)"
        print("    %-13s %s" % (k, shown))
    print("  -- security (model_security_re.py) --")
    for k in _SECURITY_KEYS:
        v = cfg.get(k)
        print("    %-13s %s" % (k, v if v is not None else "(unset)"))


def _set_config(log, cfg, args):
    if len(args) < 2:
        log.log("warning", "CONFIG", "usage: /set <key> <value>   (keys: %s)" % ", ".join(sorted(cfg)))
        return
    key = args[0].lower()
    val = " ".join(args[1:])
    if key not in cfg:
        log.log("warning", "CONFIG", "unknown key '%s' (known: %s)" % (key, ", ".join(sorted(cfg))))
        return
    if key == "mode":
        val = val.strip().lower()
        if val not in ("host", "device"):
            log.log("error", "CONFIG", "mode must be 'host' or 'device'")
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
        rc = subprocess.run(argv, cwd=_REPO).returncode   # repo root: relative model paths resolve
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

    # acquisition target: explicit passthrough wins; else config sec_gguf, else Ollama model
    if "--ollama" not in args and "--gguf" not in args:
        if cfg.get("sec_gguf"):
            args = ["--gguf", str(cfg["sec_gguf"])] + args
        elif cfg.get("model"):
            args = ["--ollama", str(cfg["model"])] + args
        else:
            log.log("warning", "SECURITY",
                    "no target — /set model <name> (Ollama) or /set sec_gguf <path>, or pass --ollama/--gguf")
            return

    # config-managed options, forwarded only where valid for the subcommand
    if cfg.get("sec_out") and "--out" not in args:
        args += ["--out", str(cfg["sec_out"])]
    if cfg.get("sec_dynamic") and subcmd != "threat" and "--dynamic" not in args:
        args += ["--dynamic"]
    if subcmd in ("analyze", "integrity"):
        if cfg.get("sec_registry") and "--registry" not in args:
            args += ["--registry", str(cfg["sec_registry"])]
        if cfg.get("sec_assets") and "--assets" not in args:
            args += ["--assets", str(cfg["sec_assets"])]

    argv = [sys.executable, _SECURITY_SCRIPT, subcmd] + args
    log.log("info", "SECURITY", "%s: %s" % (subcmd, " ".join(argv[1:])))
    try:
        rc = subprocess.run(argv, cwd=_REPO).returncode   # repo root: relative model paths resolve
        log.log("ok" if rc == 0 else "error", "SECURITY", "model_security_re exit code %d" % rc)
    except Exception as e:                                       # noqa: BLE001
        log.log("error", "SECURITY", "failed to run model_security_re: %s" % e)


def _list_serial_ports():
    import glob
    return sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def _resolve_device_port(cfg, log):
    """Pick the device serial port: the configured one if present; else auto-detect
    (/dev/ttyACM*, /dev/ttyUSB*) — one match is used, several prompt with a list, none prompts."""
    port = cfg.get("port")
    if port and os.path.exists(port):
        return port
    found = _list_serial_ports()
    if len(found) == 1:
        log.log("info", "RUNNER", "auto-detected device port: %s" % found[0])
        return found[0]
    if len(found) > 1:
        return _ask("device port", options=found, default=(port if port in found else found[0]))
    return _ask("device port (none auto-detected)", default=port)


def _setup_backend(mode, cfg, log):
    """(Re)build the backend for the active mode. Returns (device, host_ctx) or None on failure."""
    grammar_file = os.path.join(_GRAMMAR_DIR, cfg["grammar"]) if cfg.get("grammar") else None
    if mode == "device":
        port = _resolve_device_port(cfg, log)
        if not port:
            log.log("error", "RUNNER", "no device serial port available")
            return None
        cfg["port"] = port
        log.log("info", "RUNNER", "device mode - connecting to autonomous STM32N6 on %s ..." % port)
        try:
            device = DeviceTerminal(port, baud=int(cfg["baud"]), logger=log,
                                    boot_timeout=int(cfg["boot_timeout"]))
        except Exception as e:                                   # noqa: BLE001
            log.log("error", "RUNNER", "cannot open %s: %s" % (port, e))
            return None
        log.log("ok", "RUNNER", "device ready - it parses + evaluates + runs the NPU on-chip")
        return (device, None)

    if not cfg.get("model"):
        log.log("error", "RUNNER", "model is required for host mode (/set model <name>)")
        return None
    if not grammar_file:
        log.log("error", "RUNNER", "grammar is required for host mode (/set grammar <file>)")
        return None
    sys.path.insert(0, os.path.dirname(_HERE))
    from classes.class_model_grammar import ModelGrammar, GrammarRunner   # lazy (pulls gguf chain)
    gf = ModelGrammar.load_file(grammar_file, logger=log)
    if not gf:
        log.log("error", "RUNNER", "could not load grammar '%s'" % grammar_file)
        return None
    gname = gf["name"]
    tree = gf["tree"]
    playbook = tree[gname] if isinstance(tree.get(gname), dict) else tree
    rules = list(playbook.keys())
    start_rule = rules[0] if rules else "expr"
    qfn = ollama_query_fn(cfg["model"], cfg.get("host"))
    commands = _load_command_vocab(grammar_file, log)   # auto-loaded from the grammar meta

    def make_runner():
        return GrammarRunner(grammar_name=gname, query_fn=qfn,
                             fallback_playbook=playbook, logger=log, commands=commands)

    host_ctx = {"gname": gname, "playbook": playbook, "rules": rules,
                "start_rule": start_rule, "make_runner": make_runner,
                "GrammarRunner": GrammarRunner}
    log.log("info", "RUNNER", "host mode - GrammarRunner local, oracle = Ollama '%s'" % cfg["model"])
    return (None, host_ctx)


def _repl(current_mode, cfg, device, host_ctx, log):
    """Run the REPL for one mode. Returns the mode to switch to, or None to quit."""
    base = ["/?", "/help", "/bye", "/mode", "/grammar", "/config", "/get", "/set",
            "/create", "/export", "/security", "exit", "quit"]
    if host_ctx:
        base = base + ["/rules"]
    _install_completer(base,
                       cfg=cfg,
                       security_subcmds=list(_SECURITY_SUBCMDS),
                       grammar_names=(host_ctx["rules"] if host_ctx else []))
    grammar_file = os.path.join(_GRAMMAR_DIR, cfg["grammar"]) if cfg.get("grammar") else None
    log.log("info", "SYSTEM",
            "mode=%s — type an expression (e.g. 3 + 4), /? for help, /bye to quit" % current_mode)

    while True:
        try:
            ui = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not ui:
            continue
        parts = ui.split()
        cmd = parts[0].lower()
        low = ui.lower()

        if low in ("/bye", "exit", "quit"):
            log.log("ok", "SYSTEM", "closing runner"
                    + (" - the device keeps running standalone" if current_mode == "device" else ""))
            return None
        if cmd in ("/?", "/help"):
            _print_help(current_mode); continue
        if cmd == "/mode":
            if len(parts) >= 2 and parts[1].lower() in ("host", "device"):
                new = parts[1].lower()
                if new != current_mode:
                    log.log("info", "SYSTEM", "switching mode: %s -> %s" % (current_mode, new))
                    cfg["mode"] = new
                    return new
            _show_mode(log, current_mode, cfg)
            continue
        if cmd in ("/config", "/get"):
            _show_config(cfg, current_mode, parts[1:]); continue
        if cmd == "/set":
            _set_config(log, cfg, parts[1:])
            key = parts[1].lower() if len(parts) >= 2 else ""
            if len(parts) >= 3 and key == "mode" \
                    and cfg.get("mode") in ("host", "device") and cfg["mode"] != current_mode:
                log.log("info", "SYSTEM", "switching mode: %s -> %s" % (current_mode, cfg["mode"]))
                return cfg["mode"]
            if len(parts) >= 3 and key == "grammar":
                _apply_grammar_derived(cfg, log)   # new grammar -> auto-set tokenizer + training
                if host_ctx:                       # reload host backend with the new grammar
                    return current_mode
            continue
        if cmd == "/grammar":
            if len(parts) >= 2 and parts[1].strip():       # /grammar <file> -> switch grammar
                cfg["grammar"] = parts[1].strip()
                log.log("ok", "SYSTEM", "grammar set to %s" % cfg["grammar"])
                _apply_grammar_derived(cfg, log)   # auto-set tokenizer + surface training from meta
                if host_ctx:        # host backend depends on grammar + tokenizer + training -> reload
                    return current_mode
                continue
            if host_ctx:                                    # /grammar -> print active grammar
                for r in host_ctx["rules"]:
                    print("  <%s> ::= %s" % (r, host_ctx["playbook"][r]))
            elif grammar_file:
                _print_grammar(grammar_file)
            else:
                print("(no grammar configured)")
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
        if current_mode == "device":
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


# ----------------------------------------------------------------------------
# unified runner — one client; mode is live-switchable (/set mode | /mode <m>)
# ----------------------------------------------------------------------------
def run(mode=None, config=None, logger=None):
    log = logger or _logger()
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if k in cfg and v is not None})
    if mode:
        cfg["mode"] = mode
    if not cfg.get("mode"):
        cfg["mode"] = "device"

    prev_mode = None
    while True:
        current_mode = cfg["mode"]
        backend = _setup_backend(current_mode, cfg, log)
        if backend is None:
            if prev_mode is None:
                return 1
            log.log("warning", "RUNNER",
                    "could not enter %s mode — staying in %s" % (current_mode, prev_mode))
            cfg["mode"] = prev_mode
            continue
        device, host_ctx = backend
        prev_mode = current_mode
        nxt = _repl(current_mode, cfg, device, host_ctx, log)
        if device is not None:
            device.close()
        if nxt is None:
            break
        cfg["mode"] = nxt
    return 0


# ----------------------------------------------------------------------------
# backward-compatible thin wrappers (older callers / direct imports)
# ----------------------------------------------------------------------------
def run_device(port=None, grammar=None):
    return run("device", {"port": port, "grammar": grammar})


def run_host(grammar=None, model=None, host=None):
    return run("host", {"grammar": grammar, "model": model, "host": host})


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
