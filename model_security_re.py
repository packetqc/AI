#!/usr/bin/env python3
"""
model_security_re.py — Blackbox security reverse-engineering of a loadable model.

Two tracks (host-only, blackbox):
  static   parse the Ollama-downloaded GGUF file WITHOUT running it
           (format-safety triage + content extraction + exec-capability sweep)
  dynamic  query the model live via Ollama (behavioral probing + grammar
           reconstruction from outputs — output is evidence, never executed)
  analyze  run BOTH into one forensic case folder + write CASE.md

Trust boundary: uses ONLY the model artifact + live query access. Never the
client/app source, the training files, or the HF source.

All evidence + extractions are written to a dated forensic case folder:
  forensics/<YYYY-MM-DD>/<model>/
    <model>.gguf            staged blob (chain-of-custody)
    static_report.md        human-readable static analysis
    extracted_content.json  machine-readable extraction (metadata/vocab/merges/tensors/sweep)
    dynamic_report.md        live probe transcript
    recovered_grammar.bnf    best-effort grammar reconstructed from the model
    CASE.md                 case summary + file index (analyze)

Usage:
  python3 model_security_re.py static  --ollama <name>  [--out forensics]
  python3 model_security_re.py static  --gguf <path>    [--out forensics]
  python3 model_security_re.py dynamic --ollama <name>  [--k N] [--oracle FILE]
  python3 model_security_re.py analyze --ollama <name>  [--k N] [--oracle FILE]
"""
import argparse, datetime, json, os, re, sys, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "classes"))
from class_model_security import (  # noqa: E402
    OllamaArtifact, GgufStaticAnalyzer, ExecCapabilityDetector,
    sha256_file, render_static_report, discover_symbols,
)

DEF_OUT = "forensics"


# ---------------------------------------------------------------------------
# forensic case folder
# ---------------------------------------------------------------------------
def case_dir(out_root, model_name):
    """forensics/<UTC-date>/<sanitized-model>/ — created if missing."""
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    safe = re.sub(r"[^\w.-]", "_", model_name)
    d = os.path.join(out_root, day, safe)
    os.makedirs(d, exist_ok=True)
    return d


def _content_json(name, gguf, sha, artifact, ana):
    tok = ana.tokenizer()
    md = ana.metadata()
    det = ExecCapabilityDetector()
    high, low_counts, enc = det.scan_vocab(tok["decoded"])
    template = det.scan_template(md, artifact.template if artifact else None)
    return {
        "name": name, "file": os.path.basename(gguf), "size_bytes": ana.size, "sha256": sha,
        "architecture": md.get("general.architecture"),
        "model_class": det.model_class(tok["n"]), "vocab_size": tok["n"],
        "ollama_template": (artifact.template if artifact else None),
        "metadata": {k: str(v) for k, v in md.items()},
        "special_tokens": tok["specials"],
        "digit_terminals": sorted({d.strip() for d in tok["decoded"] if d.strip() in list("0123456789")}),
        "operator_terminals": sorted({d.strip() for d in tok["decoded"] if d.strip() in ["+", "-", "*", "/", "(", ")", "=", "."]}),
        "discovered_symbols": discover_symbols(tok["decoded"], top=24),
        "tokens": tok["decoded"],
        "token_types": tok["types"],
        "merges": tok["merges"],
        "tensors": ana.tensors(),
        "exec_sweep": {
            "high": [f.__dict__ for f in high],
            "low_word_counts": dict(low_counts),
            "encoded": [f.__dict__ for f in enc],
            "template_pattern_C": (template.__dict__ if template else None),
            "verdict": det.verdict(tok["n"], high, low_counts, template, enc),
        },
    }


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------
def run_static(name, gguf, artifact, case):
    digest = sha256_file(gguf)
    ana = GgufStaticAnalyzer(gguf)
    report = render_static_report(name, gguf, digest, artifact, ana)
    with open(os.path.join(case, "static_report.md"), "w") as f:
        f.write(report)
    with open(os.path.join(case, "extracted_content.json"), "w") as f:
        json.dump(_content_json(name, gguf, digest, artifact, ana), f, indent=2, default=str)
    return ana, digest


def cmd_static(args):
    if args.ollama:
        artifact = OllamaArtifact.resolve(args.ollama)
        if not artifact.blob_path or not os.path.exists(artifact.blob_path):
            sys.exit(f"could not resolve blob for '{args.ollama}' via `ollama show`")
        case = case_dir(args.out, args.ollama)
        gguf = artifact.stage(case)
        name = args.ollama
    else:
        artifact, gguf = None, args.gguf
        case = case_dir(args.out, os.path.splitext(os.path.basename(gguf))[0])
        name = os.path.basename(gguf)
    run_static(name, gguf, artifact, case)
    print(open(os.path.join(case, "static_report.md")).read())
    _list_case(case)


# ---------------------------------------------------------------------------
# dynamic
# ---------------------------------------------------------------------------
def _ollama_generate(name, prompt, n=48):
    body = json.dumps({"model": name, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": n}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("response", "")


def _score_against_oracle(recovered, oracle_path):
    truth = set(re.findall(r"<([A-Za-z_]\w*)>\s*::=", open(oracle_path).read()))
    hit = sorted(truth & set(recovered))
    return (f"[SELF-TEST vs {os.path.basename(oracle_path)} — validation only, NOT a method input]\n"
            f"  oracle nonterminals : {sorted(truth)}\n"
            f"  recovered & matched : {hit}  ({len(hit)}/{len(truth)})")


def run_dynamic(name, case, k, rules=None, oracle=None):
    if rules:
        seeds, src = rules, "analyst-supplied"
    else:
        art = OllamaArtifact.resolve(name)
        if not art.blob_path or not os.path.exists(art.blob_path):
            sys.exit(f"cannot resolve blob for '{name}' to discover symbols")
        decoded = GgufStaticAnalyzer(art.blob_path).tokenizer()["decoded"]
        seeds, src = discover_symbols(decoded, top=k), "model-derived (vocab merges)"

    lines = [f"# Dynamic analysis — {name}  (live blackbox probing)",
             f"# reconstruction seeds [{src}]: {seeds}", ""]
    recovered, rule_body, rule_alts = {}, {}, {}
    _stop = {"defined", "is", "as", "can", "be", "or", "the", "routes", "an", "a"}

    def _informative(resp):
        # rank by distinct meaningful tokens (len>=3, not a stopword) — this beats
        # the junk '", ", defined,' responses and prefers real symbol/command bodies
        toks = {w for w in re.findall(r"[A-Za-z_]{3,}", resp.lower()) if w not in _stop}
        return (len(toks), len(resp))

    for r in seeds:
        probes = []
        for p in (f"<{r}> ::=", f"A {r} is", r):
            resp = _ollama_generate(name, p).strip().replace("\n", " ")
            lines.append(f"  {p!r:26} -> {resp[:120]}")
            probes.append(resp)
            recovered.setdefault(r, []).append(resp)
        lines.append("")
        rule_body[r] = max(probes, key=_informative, default="")
        rule_alts[r] = probes
    lines.append("Outputs are evidence only — nothing emitted by the model is executed.")
    score = _score_against_oracle(recovered, oracle) if oracle else None
    if score:
        lines += ["", score]
    with open(os.path.join(case, "dynamic_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    bnf = ["# Recovered grammar (blackbox, best-effort) — seeds are MODEL-derived, not host-sourced",
           f"# model: {name}   reconstructed: {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
           "# primary body = most-informative probe response; ## alts = the other probes", ""]
    for r, b in rule_body.items():
        bnf.append(f"<{r}> ::= {b}")
        for alt in rule_alts[r]:
            if alt != b:
                bnf.append(f"    ## alt: {alt[:100]}")
        bnf.append("")
    with open(os.path.join(case, "recovered_grammar.bnf"), "w") as f:
        f.write("\n".join(bnf) + "\n")
    return seeds, rule_body, score


def cmd_dynamic(args):
    case = case_dir(args.out, args.ollama)
    run_dynamic(args.ollama, case, args.k, args.rules, args.oracle)
    print(open(os.path.join(case, "dynamic_report.md")).read())
    _list_case(case)


# ---------------------------------------------------------------------------
# analyze — both tracks into one case folder + CASE.md
# ---------------------------------------------------------------------------
def cmd_analyze(args):
    artifact = OllamaArtifact.resolve(args.ollama)
    if not artifact.blob_path or not os.path.exists(artifact.blob_path):
        sys.exit(f"could not resolve blob for '{args.ollama}' via `ollama show`")
    case = case_dir(args.out, args.ollama)
    gguf = artifact.stage(case)
    ana, digest = run_static(args.ollama, gguf, artifact, case)
    content = json.load(open(os.path.join(case, "extracted_content.json")))
    seeds, rule_body, score = run_dynamic(args.ollama, case, args.k, args.rules, args.oracle)

    case_md = [
        f"# Forensic case — `{args.ollama}`",
        f"- date (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"- artifact: `{os.path.basename(gguf)}`  ({ana.size:,} bytes)",
        f"- sha256: `{digest}`",
        f"- model-class: **{content['model_class']}**  ·  vocab {content['vocab_size']:,}",
        f"- static verdict: **{content['exec_sweep']['verdict']}**",
        f"- model-derived symbols: {content['discovered_symbols'][:16]}",
        f"- dynamic seeds probed: {seeds}",
        f"- rules reconstructed: {sum(1 for b in rule_body.values() if b)}/{len(rule_body)}",
        "",
        "## Evidence files",
        "- `" + os.path.basename(gguf) + "` — staged blob (chain-of-custody)",
        "- `static_report.md` — human-readable static analysis",
        "- `extracted_content.json` — full machine-readable extraction",
        "- `dynamic_report.md` — live probe transcript",
        "- `recovered_grammar.bnf` — grammar reconstructed from the model",
    ]
    if score:
        case_md += ["", "## Self-test", "```", score, "```"]
    with open(os.path.join(case, "CASE.md"), "w") as f:
        f.write("\n".join(case_md) + "\n")
    print("\n".join(case_md))
    _list_case(case)


def _list_case(case):
    print(f"\n[forensic case folder] {case}")
    for fn in sorted(os.listdir(case)):
        sz = os.path.getsize(os.path.join(case, fn))
        print(f"  - {fn}  ({sz:,} B)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Blackbox model security RE (forensic file output)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("static", help="artifact-only analysis (no model load)")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--ollama", help="ollama model name (resolved via `ollama show`)")
    g.add_argument("--gguf", help="direct path to a .gguf file")
    s.add_argument("--out", default=DEF_OUT, help="forensic root (default: forensics/)")
    s.set_defaults(func=cmd_static)

    d = sub.add_parser("dynamic", help="live behavioral probing (model-only seeds)")
    d.add_argument("--ollama", required=True)
    d.add_argument("--out", default=DEF_OUT)
    d.add_argument("--rules", nargs="+", default=None,
                   help="optional analyst seeds; default = discover from the model's vocab")
    d.add_argument("--k", type=int, default=12, help="number of model-derived symbols to probe")
    d.add_argument("--oracle", default=None,
                   help="SELF-TEST ONLY: grammar file to score recall against (never a method input)")
    d.set_defaults(func=cmd_dynamic)

    a = sub.add_parser("analyze", help="static + dynamic into one forensic case folder")
    a.add_argument("--ollama", required=True)
    a.add_argument("--out", default=DEF_OUT)
    a.add_argument("--rules", nargs="+", default=None)
    a.add_argument("--k", type=int, default=12)
    a.add_argument("--oracle", default=None)
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
