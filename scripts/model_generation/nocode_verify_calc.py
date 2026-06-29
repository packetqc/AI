#!/usr/bin/env python3
"""nocode_verify_calc.py — live regression for the decomposed calculator nocode loop.

Runs the calculator through NoCodeGrammarRunner (mode=evaluate_ops) against a trained model across
all three exec policies, and probes that the model emits each operation token's body.  Exits non-zero
if any policy fails — so it doubles as a CI gate for the "nocode destiny" calculator proof.

  ./venv/bin/python scripts/model_generation/nocode_verify_calc.py [model_name]
"""
import sys, json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from classes.class_model_grammar import ModelGrammar
from classes.class_nocode_grammar import NoCodeGrammarRunner, CodeExecPolicy
from classes import class_model_runner as base


class _Quiet:
    def log(self, *a, **k):
        pass


CASES = [("3 + 4", 7), ("12 - 5", 7), ("6 * 7", 42), ("8 / 2", 4), ("2 + 3 * 4", 14), ("9 - 2 - 3", 4)]
OP_TOKENS = ["op_add", "op_sub", "op_mul", "op_div", "number"]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "model_calculator_nocode_v1"
    os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
    quiet = _Quiet()

    gf = ModelGrammar.load_file(os.path.join(_REPO, "models/grammars/playbook_model_calculator.txt"), logger=quiet)
    gname = gf["name"]; tree = gf["tree"]
    playbook = tree[gname] if isinstance(tree.get(gname), dict) else tree
    start = next(iter(playbook))
    fv = json.load(open(os.path.join(_REPO, "models/training/train_calculator_functions.json")))
    commands = {k: v for k, v in fv.items() if not k.startswith("_")}
    commands["_exec"] = fv.get("_exec", "python")
    qfn = base.ollama_query_fn(model)

    print("=== model emission per operation token (model=%s) ===" % model)
    for tok in OP_TOKENS:
        em = qfn("calculator " + tok).strip()
        vb = commands.get(tok, "")
        match = ("\n".join(l.rstrip() for l in em.splitlines()) ==
                 "\n".join(l.rstrip() for l in vb.splitlines()))
        print("  %-7s -> %-45r exact==vocab: %s" % (tok, em[:45], match))

    all_ok = True
    for policy in CodeExecPolicy.ALL:
        npass = 0
        for expr, expect in CASES:
            r = NoCodeGrammarRunner(grammar_name=gname, query_fn=qfn, fallback_playbook=playbook,
                                    logger=quiet, commands=commands, policy=policy,
                                    mode="evaluate_ops", eval_token="evaluate")
            try:
                res = r.run(expr, start_rule=start)
            except Exception as e:                                  # noqa: BLE001
                res = "EXC:" + str(e)
            npass += (res == expect)
        ok = npass == len(CASES)
        all_ok = all_ok and ok
        print("  [%s] policy %-14s %d/%d" % ("OK " if ok else "FAIL", policy, npass, len(CASES)))

    print("=== %s ===" % ("PASSED" if all_ok else "FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
