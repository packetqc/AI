#!/usr/bin/env python3
"""nocode_runner.py — CLI for the grammar-agnostic "nocode" runner (host CPU path).

A replicat-and-adapted of model_runner.py.  Where model_runner executes a grammar's logic from
hardcoded Python (evaluate) or a side-car dict (procedure commands), nocode_runner executes the
logic the MODEL emits — governed by the exec policy ladder.

  python3 scripts/nocode_runner.py --mode host --grammar playbook_model_calculator.txt \
      --model model_calculator_version_1 --policy vocab_verified

  # then:  3 + 4   ->   the evaluator is sourced from the model, not from class_model_grammar.py
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from classes import class_nocode_runner as runner            # noqa: E402
from classes.class_nocode_grammar import CodeExecPolicy      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="host", choices=("host", "device"),
                    help="host = CPU logic from the model (default); device = deferred")
    ap.add_argument("--grammar", nargs="+", metavar="FILE",
                    help="one or more grammar files (bare name or path). Multiple are merged so a "
                         "composing grammar can call others, e.g. --grammar playbook_combo.txt "
                         "playbook_fibonacci.txt playbook_greeting.txt")
    ap.add_argument("--model", help="Ollama model name (the grammar oracle)")
    ap.add_argument("--host", help="Ollama host URL (default: env/localhost)")
    ap.add_argument("--policy", default=CodeExecPolicy.DEFAULT, choices=CodeExecPolicy.ALL,
                    help="exec policy: token_select | vocab_verified | generative (default: %(default)s)")
    args = ap.parse_args()

    config = {"mode": args.mode, "grammar": args.grammar, "model": args.model,
              "host": args.host, "policy": args.policy}
    runner.run(mode=args.mode, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
