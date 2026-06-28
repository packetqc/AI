#!/usr/bin/env python3
"""model_runner.py — official entry point for the unified grammar runner.

This is the supported CLI for running a grammar-LM solution. It is a thin wrapper
over the runner class library (``scripts/classes/class_model_runner.py``) — all the
logic lives in that library; this script only parses arguments and dispatches.

Two execution modes that differ by WHERE the CPU logic runs:

  --mode device   the STM32N6 is AUTONOMOUS. The runner is a thin serial terminal:
                  it pushes the prompt over the ST-Link VCP and collects the output;
                  the device's own C++ GrammarRunner tokenizes, parses, evaluates and
                  drives its Neural-ART NPU entirely on-chip. NO ML dependencies —
                  run it with the system python.

  --mode host     the HOST runs GrammarRunner locally and queries an Ollama chat model
                  as the grammar oracle, parsing + evaluating on the host. Needs the
                  project venv (the ML chain is imported lazily only for this mode).

Usage:
  python3 scripts/model_runner.py --mode device --port /dev/ttyACM0
  python3 scripts/model_runner.py --mode host   --model model_calculator_test_npu
"""
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# The runner class library is the single source of truth for the runner behaviour.
from classes.class_model_runner import run_device, run_host, _DEFAULT_GRAMMAR


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="model_runner.py",
        description="Official unified grammar runner (autonomous device / host Ollama).",
    )
    ap.add_argument("--mode", choices=["host", "device"], default="device",
                    help="device: autonomous STM32N6 over serial · host: GrammarRunner + Ollama")
    ap.add_argument("--grammar", default=_DEFAULT_GRAMMAR,
                    help="BNF/EBNF playbook grammar file")
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="device serial port (device mode)")
    ap.add_argument("--model", default=None,
                    help="Ollama model name (host mode)")
    ap.add_argument("--host", default=None,
                    help="Ollama host URL (host mode)")
    a = ap.parse_args(argv)

    if a.mode == "device":
        return run_device(a.port, a.grammar)
    return run_host(a.grammar, a.model, a.host)


if __name__ == "__main__":
    sys.exit(main())
