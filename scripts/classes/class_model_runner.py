"""class_model_runner.py - Unified runner (host / device) for grammar-LM oracles.

Pioneer "nocode" grammar-LM line (dates from file/git history):
  2026-06-06  fsm_language_model.py            genesis - FSM language model
  2026-06-14  model_create_hugging_face.py     nocode model-from-grammar generation
  2026-06-16  model_create_hf_cl.py            interactive grammar client
  2026-06-27  class_model_runner.py            NPU-as-oracle unified runner (this file)

One interactive client - prompt, history, TAB completion, /commands, auto-detection -
over a PLUGGABLE oracle backend selected by mode:

  mode=host    query_fn -> Ollama chat model        (get_ollama_answer)
  mode=device  query_fn -> serial /dev/ttyACM0 -> STM32N6 Neural-ART NPU oracle

Both backends plug into the SAME GrammarRunner.query_fn seam, so the on-device NPU is
a drop-in replacement for the chat model. The CPU runner parses + evaluates from the
returned BNF rule bodies; the oracle only recalls the grammar.

Device protocol (line based, 115200 8N1) - see run-23 FSBL main.c [NPU-ORACLE]:
  host   -> "<rule_name>\n"      one of the grammar's rule names (e.g. expr)
  device -> "ANS:<rule_body>\n"  the NPU-recalled BNF rule body
            "ERR:unknown rule '<x>'\n" on miss

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
sys.path.insert(0, os.path.dirname(_HERE))          # scripts/  so 'classes' resolves
from classes.class_model_grammar import ModelGrammar, GrammarRunner   # noqa: E402
from classes.class_terminal_logs import TerminalLogger                # noqa: E402

_DEFAULT_GRAMMAR = os.path.join(os.path.dirname(_HERE), "..",
                                "models", "grammars", "playbook_model_calculator.txt")


# ----------------------------------------------------------------------------
# Backends - query_fn factories (the pluggable oracle)
# ----------------------------------------------------------------------------
def ollama_query_fn(model, host=None):
    """Host backend: query an Ollama chat model for one grammar rule."""
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def q(prompt):
        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode()).get("response", "").strip()
    return q


class SerialOracle:
    """Device backend: talk to the STM32N6 NPU oracle over the ST-Link VCP."""

    def __init__(self, port="/dev/ttyACM0", baud=115200, logger=None, ready_timeout=15):
        self.port = port
        self.logger = logger
        subprocess.run(["stty", "-F", port, str(baud), "raw", "-echo", "-echoe", "-echok"],
                       check=False)
        self.f = open(port, "r+b", buffering=0)
        if not self._drain_until("[NPU-ORACLE] ready", ready_timeout) and logger:
            logger.log("warning", "DEVICE",
                       "did not see the ready banner (device may already be serving) - continuing")

    def _readline(self, timeout):
        buf = bytearray()
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.f], [], [], max(0.0, end - time.time()))
            if not r:
                break
            c = self.f.read(1)
            if not c:
                continue
            if c in (b"\n", b"\r"):
                if buf:
                    return buf.decode(errors="replace")
                continue
            buf += c
        return buf.decode(errors="replace") if buf else None

    def _drain_until(self, marker, timeout):
        end = time.time() + timeout
        while time.time() < end:
            ln = self._readline(max(0.5, end - time.time()))
            if ln is None:
                continue
            if self.logger:
                self.logger.log("debug", "DEVICE", ln)
            if marker in ln:
                return True
        return False

    def query(self, prompt):
        """prompt is '<grammar> <rule>' (GrammarRunner convention); the device wants the rule."""
        rule = prompt.split()[-1]
        self.f.write((rule + "\n").encode())
        for _ in range(64):
            ln = self._readline(15)
            if ln is None:
                break
            if ln.startswith("ANS:"):
                return ln[4:].strip()
            if ln.startswith("ERR:"):
                return ""
        return ""


def serial_query_fn(port="/dev/ttyACM0", baud=115200, logger=None):
    return SerialOracle(port, baud, logger).query


# ----------------------------------------------------------------------------
# Interactive runner with command assistance
# ----------------------------------------------------------------------------
def run_interactive(grammar_file=_DEFAULT_GRAMMAR, mode="device",
                    model=None, port="/dev/ttyACM0", host=None):
    logger = TerminalLogger()

    gf = ModelGrammar.load_file(grammar_file, logger=logger)
    if not gf:
        logger.log("error", "RUNNER", "could not load grammar '%s'" % grammar_file)
        return 1
    gname = gf["name"]
    # load_file's "tree" is nested {grammar_name: {rule: body}} (ModelGrammar.to_playbook_tree);
    # GrammarRunner wants the flat {rule: body} subtree.
    tree = gf["tree"]
    if isinstance(tree.get(gname), dict):
        playbook = tree[gname]
    else:
        playbook = tree
    rules = list(playbook.keys())
    start_rule = rules[0] if rules else "expr"

    if mode == "device":
        logger.log("info", "RUNNER", "connecting to device NPU oracle on %s ..." % port)
        try:
            qfn = serial_query_fn(port, logger=logger)
        except Exception as e:                                   # noqa: BLE001
            logger.log("error", "RUNNER", "cannot open %s: %s" % (port, e))
            return 1
        logger.log("ok", "RUNNER", "device oracle ready")
    else:
        if not model:
            logger.log("error", "RUNNER", "--model is required for host mode")
            return 1
        logger.log("info", "RUNNER", "using host Ollama model '%s'" % model)
        qfn = ollama_query_fn(model, host)

    def make_runner():
        return GrammarRunner(grammar_name=gname, query_fn=qfn,
                             fallback_playbook=playbook, logger=logger)

    # readline: history + TAB completion (commands + rule names)
    CMDS = ["/help", "/?", "/bye", "/mode", "/rules", "/grammar", "exit", "quit"]
    try:
        import readline
        def _completer(text, state):
            opts = [c for c in CMDS + rules if c.startswith(text)]
            return opts[state] if state < len(opts) else None
        readline.set_completer(_completer)
        readline.parse_and_bind("tab: complete")
    except Exception:                                            # noqa: BLE001
        pass

    backend = (port if mode == "device" else model)
    logger.log("info", "SYSTEM",
               "===== unified grammar runner - mode=%s grammar=%s (%s) =====" % (mode, gname, backend))
    logger.log("info", "SYSTEM",
               "type an expression (e.g. 3 + 4), /? for help, TAB to complete, /bye to quit")

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
            logger.log("ok", "SYSTEM", "closing runner - goodbye")
            break
        if ui in ("/?", "/help"):
            print("  /?            this help")
            print("  /rules        recall every grammar rule via the oracle (shows the dialog)")
            print("  /grammar      print the loaded playbook grammar")
            print("  /mode         show the active oracle backend")
            print("  /bye          quit")
            print("  <expression>  auto-detected + evaluated via the %s oracle (e.g. 12 * (3 + 4))" % mode)
            print("  TAB           complete commands + rule names")
            continue
        if low == "/mode":
            logger.log("info", "SYSTEM", "mode=%s  grammar=%s  backend=%s" % (mode, gname, backend))
            continue
        if low == "/grammar":
            for r in rules:
                print("  <%s> ::= %s" % (r, playbook[r]))
            continue
        if low == "/rules":
            r = make_runner()
            for name in rules:
                r._query_rule(name)          # triggers + logs the oracle dialog
            continue

        # auto-detection: does the input parse against the grammar? (playbook-only, silent)
        if GrammarRunner.probe(gname, ui, playbook, start_rule=start_rule):
            make_runner().run(ui, start_rule=start_rule)
        else:
            logger.log("warning", "RUNNER",
                       "not a valid %s expression: '%s' (try: 3 + 4)" % (gname, ui))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Unified grammar runner (host Ollama / device NPU).")
    ap.add_argument("--mode", choices=["host", "device"], default="device",
                    help="oracle backend: host (Ollama) or device (STM32N6 NPU over serial)")
    ap.add_argument("--grammar", default=_DEFAULT_GRAMMAR, help="BNF/EBNF playbook grammar file")
    ap.add_argument("--port", default="/dev/ttyACM0", help="device serial port (device mode)")
    ap.add_argument("--model", default=None, help="Ollama model name (host mode)")
    ap.add_argument("--host", default=None, help="Ollama host URL (host mode)")
    a = ap.parse_args()
    sys.exit(run_interactive(a.grammar, a.mode, a.model, a.port, a.host))
