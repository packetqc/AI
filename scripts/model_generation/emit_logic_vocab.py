#!/usr/bin/env python3
"""emit_logic_vocab.py — CLI for LogicTransposer.

Transpose working CPU-side grammar logic into a model-trainable function-vocabulary JSON.

  python3 scripts/model_generation/emit_logic_vocab.py --grammar calculator
  python3 scripts/model_generation/emit_logic_vocab.py --grammar calculator --selftest
  python3 scripts/model_generation/emit_logic_vocab.py --list

The emitted file (models/training/train_<grammar>_functions.json) is consumed by the model
create pipeline (trained as (token -> body) anchors) and by nocode_runner.py (executed per the
active exec policy).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from classes.class_logic_transposer import LogicTransposer   # noqa: E402


def _selftest_calculator(transposer):
    """Prove the transposed evaluate() reproduces the original arithmetic — fully offline.

    Builds parse-tree dicts in the shape GrammarRunner.parse() produces and checks the
    standalone, model-carryable body computes the same results as the in-class method.
    """
    vocab = transposer.analyze_grammar("calculator")
    body = vocab["evaluate"]
    ns = {"_log": lambda *a, **k: None}          # logging shim, as nocode_runner will provide
    exec(compile(body, "<transposed:calculator.evaluate>", "exec"), ns)   # noqa: S102
    evaluate = ns["evaluate"]

    def term(v):
        return {"terminal": v}

    def number(*digits):
        return {"rule": "number", "children": [term(d) for d in digits]}

    # 3 + 4  ->  expr( number(3) "+" number(4) )
    t_add = {"rule": "expr", "children": [number("3"), term("+"), number("4")]}
    # 12 - 5 ->  number(1,2) "-" number(5)
    t_sub = {"rule": "expr", "children": [number("1", "2"), term("-"), number("5")]}
    # 6 * 7
    t_mul = {"rule": "term", "children": [number("6"), term("*"), number("7")]}
    # 8 / 2
    t_div = {"rule": "term", "children": [number("8"), term("/"), number("2")]}

    cases = [(t_add, 7), (t_sub, 7), (t_mul, 42), (t_div, 4)]
    ok = True
    for tree, expect in cases:
        got = evaluate(tree)
        flag = "OK " if got == expect else "FAIL"
        if got != expect:
            ok = False
        print("  [%s] evaluate(...) = %r  (expected %r)" % (flag, got, expect))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grammar", help="grammar name to transpose (see --list)")
    ap.add_argument("--out", help="output path (default: models/training/train_<g>_functions.json)")
    ap.add_argument("--selftest", action="store_true",
                    help="verify the transposed logic offline before writing")
    ap.add_argument("--list", action="store_true", help="list grammars with a transposition spec")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the emitted vocabulary instead of writing it")
    ap.add_argument("--decompose", action="store_true",
                    help="emit the SMALL per-operation tokens (decomposed) instead of the whole function")
    ap.add_argument("--review", action="store_true",
                    help="code-review the vocabulary: flag tokens that are too big to be precise")
    args = ap.parse_args()

    transposer = LogicTransposer()
    kind = "decompose" if args.decompose else "analyze"

    if args.list:
        print("Grammars with a transposition spec:")
        for g in LogicTransposer.known_grammars():
            print("  - " + g + ("  [decomposable]" if g in LogicTransposer.decomposable_grammars() else ""))
        return 0

    if not args.grammar:
        ap.error("--grammar is required (or use --list)")

    if args.selftest:
        if args.grammar != "calculator":
            print("selftest currently implemented for 'calculator' only")
            return 2
        print("=== selftest: transposed calculator.evaluate ===")
        ok = _selftest_calculator(transposer)
        print("=== selftest %s ===" % ("PASSED" if ok else "FAILED"))
        if not ok:
            return 1

    vocab = transposer.decompose_grammar(args.grammar) if kind == "decompose" \
        else transposer.analyze_grammar(args.grammar)

    if args.review:
        print("=== code review (%s, budget %d chars / %d lines) ===" %
              (kind, LogicTransposer.REVIEW_MAX_CHARS, LogicTransposer.REVIEW_MAX_LINES))
        all_ok = True
        for token, ok, detail in transposer.review(vocab):
            all_ok = all_ok and ok
            print("  [%s] %-10s %s" % ("OK " if ok else "BIG", token, detail))
        print("=== review %s ===" % ("PASSED" if all_ok else "FLAGGED — decompose oversized tokens"))

    if args.show:
        import json
        print(json.dumps(vocab, indent=2, ensure_ascii=False))
        return 0

    if args.review and not args.decompose and not args.out:
        return 0   # review-only on the analyze form: don't overwrite the trained file

    path = transposer.emit(args.grammar, out_path=args.out, kind=kind)
    print("wrote: " + os.path.relpath(path, _REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
