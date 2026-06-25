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
    OllamaArtifact, GgufStaticAnalyzer, ExecCapabilityDetector,
    sha256_file, render_static_report,
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
    det = ExecCapabilityDetector()
    findings = det.scan_vocab(ana.tokenizer()["decoded"])
    report = render_static_report(name, gguf, digest, artifact, ana, findings)

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


def cmd_dynamic(args):
    rules = args.rules
    print(f"# Dynamic analysis — {args.ollama}  (live blackbox probing)\n")
    recovered = {}
    # mixed prompt battery: BNF-form + prose (prose escapes the temp-0 attractor)
    for r in rules:
        for p in (f"<{r}> ::=", f"A {r} is", r):
            resp = _ollama_generate(args.ollama, p).strip().replace("\n", " ")
            print(f'  {p!r:24} -> {resp[:90]}')
            recovered.setdefault(r, []).append(resp)
        print()
    print("Outputs are evidence only — nothing emitted by the model is executed.")


def main():
    ap = argparse.ArgumentParser(description="Blackbox model security RE")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("static", help="artifact-only analysis (no model load)")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--ollama", help="ollama model name (resolved via `ollama show`)")
    g.add_argument("--gguf", help="direct path to a .gguf file")
    s.add_argument("--out", default=DEF_OUT, help="evidence output dir")
    s.set_defaults(func=cmd_static)

    d = sub.add_parser("dynamic", help="live behavioral probing")
    d.add_argument("--ollama", required=True)
    d.add_argument("--rules", nargs="+",
                   default=["expr", "term", "factor", "number", "digit"])
    d.set_defaults(func=cmd_dynamic)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
