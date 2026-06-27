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


def run_device(port=("/dev/ttyACM0"), grammar_file=_DEFAULT_GRAMMAR):
    log = _logger()
    log.log("info", "RUNNER", "device mode - connecting to autonomous STM32N6 on %s ..." % port)
    try:
        term = DeviceTerminal(port, logger=log)
    except Exception as e:                                       # noqa: BLE001
        log.log("error", "RUNNER", "cannot open %s: %s" % (port, e))
        return 1
    log.log("ok", "RUNNER", "device ready - it parses + evaluates + runs the NPU on-chip")
    log.log("info", "SYSTEM", "type an expression (e.g. 3 + 4), /? for help, /bye to quit")

    _install_completer(["/?", "/help", "/bye", "/mode", "/grammar", "exit", "quit"])
    while True:
        try:
            ui = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not ui:
            continue
        low = ui.lower()
        if low in ("/bye", "exit", "quit"):
            log.log("ok", "SYSTEM", "closing runner - the device keeps running standalone")
            break
        if ui in ("/?", "/help"):
            print("  /?            this help")
            print("  /mode         show the active mode")
            print("  /grammar      print the BNF grammar (for reference; the device has its own)")
            print("  /bye          quit the runner (device stays autonomous)")
            print("  <expression>  pushed to the device; it computes on-chip and returns the output")
            continue
        if low == "/mode":
            log.log("info", "SYSTEM", "mode=device (autonomous)  port=%s" % port)
            continue
        if low == "/grammar":
            _print_grammar(grammar_file)
            continue
        # push the prompt; the device does everything and we print what it returns
        out = term.eval_expr(ui).strip("\r\n")
        if out:
            print(out)
    return 0


def _print_grammar(grammar_file):
    try:
        with open(grammar_file, "r", encoding="utf-8") as fh:
            sys.stdout.write(fh.read())
    except OSError as e:                                         # noqa: BLE001
        print("(grammar unavailable: %s)" % e)


# ----------------------------------------------------------------------------
# HOST mode - the host runs GrammarRunner locally, Ollama as the grammar oracle
# ----------------------------------------------------------------------------
def ollama_query_fn(model, host=None):
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def q(prompt):
        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode()).get("response", "").strip()
    return q


def run_host(grammar_file=_DEFAULT_GRAMMAR, model=None, host=None):
    log = _logger()
    if not model:
        log.log("error", "RUNNER", "--model is required for host mode")
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
    qfn = ollama_query_fn(model, host)

    log.log("info", "RUNNER", "host mode - GrammarRunner local, oracle = Ollama '%s'" % model)
    log.log("info", "SYSTEM", "type an expression (e.g. 3 + 4), /? for help, /bye to quit")
    _install_completer(["/?", "/help", "/bye", "/mode", "/grammar", "/rules"] + rules)

    def make_runner():
        return GrammarRunner(grammar_name=gname, query_fn=qfn,
                             fallback_playbook=playbook, logger=log)

    while True:
        try:
            ui = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not ui:
            continue
        low = ui.lower()
        if low in ("/bye", "exit", "quit"):
            log.log("ok", "SYSTEM", "closing runner")
            break
        if ui in ("/?", "/help"):
            print("  /?            this help")
            print("  /rules        recall every grammar rule via the Ollama oracle")
            print("  /grammar      print the loaded BNF grammar")
            print("  /mode         show the active mode")
            print("  /bye          quit")
            print("  <expression>  parsed + evaluated locally; rules recalled from the model")
            continue
        if low == "/mode":
            log.log("info", "SYSTEM", "mode=host  grammar=%s  model=%s" % (gname, model))
            continue
        if low == "/grammar":
            for r in rules:
                print("  <%s> ::= %s" % (r, playbook[r]))
            continue
        if low == "/rules":
            r = make_runner()
            for name in rules:
                r._query_rule(name)
            continue
        if GrammarRunner.probe(gname, ui, playbook, start_rule=start_rule):
            make_runner().run(ui, start_rule=start_rule)
        else:
            log.log("warning", "RUNNER", "not a valid %s expression: '%s' (try: 3 + 4)" % (gname, ui))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Unified grammar runner (autonomous device / host Ollama).")
    ap.add_argument("--mode", choices=["host", "device"], default="device",
                    help="device: autonomous STM32N6 over serial · host: GrammarRunner + Ollama")
    ap.add_argument("--grammar", default=_DEFAULT_GRAMMAR, help="BNF/EBNF playbook grammar file")
    ap.add_argument("--port", default="/dev/ttyACM0", help="device serial port (device mode)")
    ap.add_argument("--model", default=None, help="Ollama model name (host mode)")
    ap.add_argument("--host", default=None, help="Ollama host URL (host mode)")
    a = ap.parse_args()
    if a.mode == "device":
        sys.exit(run_device(a.port, a.grammar))
    sys.exit(run_host(a.grammar, a.model, a.host))
