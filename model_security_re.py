#!/usr/bin/env python3
"""
model_security_re.py — Blackbox security reverse-engineering of a loadable model.

Two tracks (host-only, blackbox):
  static   parse the Ollama-downloaded GGUF file WITHOUT running it
           (format-safety triage + content extraction + exec-capability sweep)
  dynamic  query the model live via Ollama (behavioral probing + grammar
           reconstruction from outputs — output is evidence, never executed)

Trust boundary: uses ONLY the model artifact + live query access. Never the
client/app source, the training files, or the HF source.

Usage:
  python3 model_security_re.py static  --ollama <name> [--out DIR]
  python3 model_security_re.py static  --gguf <path>   [--out DIR]
  python3 model_security_re.py dynamic --ollama <name> [--rules expr term factor number digit]
"""
import argparse, json, os, sys, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "classes"))
from class_model_security import (  # noqa: E402
    OllamaArtifact, GgufStaticAnalyzer, sha256_file,
    render_static_report, discover_symbols,
)

DEF_OUT = "evidence"


def cmd_static(args):
    artifact = None
    if args.ollama:
        artifact = OllamaArtifact.resolve(args.ollama)
        if not artifact.blob_path or not os.path.exists(artifact.blob_path):
            sys.exit(f"could not resolve blob for '{args.ollama}' via `ollama show`")
        out = os.path.join(args.out, args.ollama.replace(":", "_").replace("/", "_"))
        gguf = artifact.stage(out)
        name = args.ollama
    else:
        gguf = args.gguf
        out = os.path.join(args.out, os.path.splitext(os.path.basename(gguf))[0])
        os.makedirs(out, exist_ok=True)
        name = os.path.basename(gguf)

    digest = sha256_file(gguf)
    ana = GgufStaticAnalyzer(gguf)
    report = render_static_report(name, gguf, digest, artifact, ana)

    rpath = os.path.join(out, "static_report.md")
    with open(rpath, "w") as f:
        f.write(report)
    print(report)
    print(f"\n[written] {rpath}")


def _ollama_generate(name, prompt, n=48):
    body = json.dumps({"model": name, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": n}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("response", "")


def _score_against_oracle(recovered, oracle_path):
    """SELF-TEST ONLY: score recovered symbols against a known grammar. Valid only
    because WE built this model — a real engagement has no oracle. Never an input
    to reconstruction."""
    import re
    truth = set(re.findall(r"<([A-Za-z_]\w*)>\s*::=", open(oracle_path).read()))
    hit = sorted(truth & set(recovered))
    print(f"\n[SELF-TEST vs {os.path.basename(oracle_path)} — validation only, NOT a method input]")
    print(f"  oracle nonterminals : {sorted(truth)}")
    print(f"  recovered & matched : {hit}  ({len(hit)}/{len(truth)})")


def cmd_dynamic(args):
    # MODEL-ONLY symbol discovery: if the analyst supplies no seeds, derive them
    # from the artifact's own vocab (BPE merges) — NEVER from host grammar files.
    if args.rules:
        rules, src = args.rules, "analyst-supplied"
    else:
        art = OllamaArtifact.resolve(args.ollama)
        if not art.blob_path or not os.path.exists(art.blob_path):
            sys.exit(f"cannot resolve blob for '{args.ollama}' to discover symbols")
        decoded = GgufStaticAnalyzer(art.blob_path).tokenizer()["decoded"]
        rules, src = discover_symbols(decoded, top=args.k), "model-derived (vocab merges)"
    print(f"# Dynamic analysis — {args.ollama}  (live blackbox probing)")
    print(f"# reconstruction seeds [{src}]: {rules}\n")
    recovered = {}
    # mixed prompt battery: BNF-form + prose (prose escapes the temp-0 attractor)
    for r in rules:
        for p in (f"<{r}> ::=", f"A {r} is", r):
            resp = _ollama_generate(args.ollama, p).strip().replace("\n", " ")
            print(f'  {p!r:26} -> {resp[:84]}')
            recovered.setdefault(r, []).append(resp)
        print()
    print("Outputs are evidence only — nothing emitted by the model is executed.")
    if args.oracle:
        _score_against_oracle(recovered, args.oracle)


def main():
    ap = argparse.ArgumentParser(description="Blackbox model security RE")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("static", help="artifact-only analysis (no model load)")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--ollama", help="ollama model name (resolved via `ollama show`)")
    g.add_argument("--gguf", help="direct path to a .gguf file")
    s.add_argument("--out", default=DEF_OUT, help="evidence output dir")
    s.set_defaults(func=cmd_static)

    d = sub.add_parser("dynamic", help="live behavioral probing (model-only seeds)")
    d.add_argument("--ollama", required=True)
    d.add_argument("--rules", nargs="+", default=None,
                   help="optional analyst seeds; default = discover from the model's vocab")
    d.add_argument("--k", type=int, default=12, help="number of model-derived symbols to probe")
    d.add_argument("--oracle", default=None,
                   help="SELF-TEST ONLY: grammar file to score recall against (never a method input)")
    d.set_defaults(func=cmd_dynamic)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
