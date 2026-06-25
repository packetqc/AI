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
    sha256_file, render_static_report, discover_symbols, token_inventory,
    RECON_LEXICON, binary_forensics, binary_forensics_verdict,
    render_tokens_decoded, decode_tokens,
)

DEF_YAR = os.path.join(os.path.dirname(__file__), "model_security_rules", "recon_c2.yar")

# heuristic: does a model response contain a literal OS-command string?
_LITERAL_CMD = re.compile(
    r"\b(nmap|ss|ip|arp|ping|curl|wget|uname|netstat|ps|nc|ssh|dig|host|masscan)\b[ \-]"
    r"|[|;>&`$]"                      # shell metacharacters
    r"|(?:^|\s)-[A-Za-z]{1,3}\b"      # flags  -sV -O -p
    r"|/(?:usr|bin|etc|sbin|proc)/",  # paths
    re.I)
_CMD_PROMPTS = ("{a} runs the command:", "execute {a}:", "the command for {a} is", "{a} command:")

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
    syms = discover_symbols(tok["decoded"], top=24)
    high, low_counts, enc = det.scan_vocab(tok["decoded"])
    template = det.scan_template(md, artifact.template if artifact else None)
    symfind = det.scan_symbols(syms)
    cats, _ = decode_tokens(tok["tokens"], tok["decoded"], tok["types"])
    return {
        "name": name, "file": os.path.basename(gguf), "size_bytes": ana.size, "sha256": sha,
        "architecture": md.get("general.architecture"),
        "model_class": det.model_class(tok["n"]), "vocab_size": tok["n"],
        "ollama_template": (artifact.template if artifact else None),
        "metadata": {k: str(v) for k, v in md.items()},
        "special_tokens": tok["specials"],
        "digit_terminals": sorted({d.strip() for d in tok["decoded"] if d.strip() in list("0123456789")}),
        "operator_terminals": sorted({d.strip() for d in tok["decoded"] if d.strip() in ["+", "-", "*", "/", "(", ")", "=", "."]}),
        "discovered_symbols": syms,
        "token_inventory": token_inventory(tok["decoded"], syms),
        "token_decode_categories": {k: len(v) for k, v in cats.items()},
        "tokens": tok["decoded"],
        "token_types": tok["types"],
        "merges": tok["merges"],
        "tensors": ana.tensors(),
        "exec_sweep": {
            "high": [f.__dict__ for f in high],
            "low_word_counts": dict(low_counts),
            "encoded": [f.__dict__ for f in enc],
            "template_pattern_C": (template.__dict__ if template else None),
            "action_grammar_pattern_D": (symfind.__dict__ if symfind else None),
            "verdict": det.verdict(tok["n"], high, low_counts, template, enc, symfind),
        },
    }


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------
def render_binary_forensics(name, bf):
    L = [f"# Binary forensics — `{name}`",
         "**Method:** raw-blob `strings` + `binwalk` + YARA — no GGUF parse, no model load.",
         f"- YARA rules: `{bf.get('yara_rules')}`", ""]
    s = bf["strings"]
    L.append(f"## strings ({s['total']:,} total)")
    L.append(f"- network endpoints — URLs: {s['urls'] or '—'} · IPs: {s['ips'] or '—'} · .onion: {s['onion'] or '—'}")
    L.append(f"- shell/exec strings: {s['shell'] or '—'}")
    L.append(f"- recon-related strings ({len(s['recon'])}): {s['recon'][:50]}")
    L.append("\n## binwalk signatures")
    L += ([f"- {row}" for row in bf["binwalk"][:30]] or
          ["- none (no embedded/compressed signatures detected)"])
    L.append("\n## YARA matches (engineer-editable rules)")
    if bf.get("yara_error"):
        L.append(f"- (yara error: {bf['yara_error']})")
    elif bf["yara"]:
        for y in bf["yara"]:
            L.append(f"- **[{y['severity']}]** `{y['rule']}` — {y['description']}  ({y['match_count']} hits)")
    else:
        L.append("- no rule matched")
    L.append(f"\n**VERDICT:** {binary_forensics_verdict(bf)}")
    return "\n".join(L) + "\n"


def run_static(name, gguf, artifact, case, yar=None):
    digest = sha256_file(gguf)
    ana = GgufStaticAnalyzer(gguf)
    with open(os.path.join(case, "static_report.md"), "w") as f:
        f.write(render_static_report(name, gguf, digest, artifact, ana))
    with open(os.path.join(case, "tokens_decoded.md"), "w") as f:
        f.write(render_tokens_decoded(name, ana.tokenizer()))
    bf = binary_forensics(gguf, yar)
    with open(os.path.join(case, "binary_forensics.md"), "w") as f:
        f.write(render_binary_forensics(name, bf))
    content = _content_json(name, gguf, digest, artifact, ana)
    content["binary_forensics"] = bf
    with open(os.path.join(case, "extracted_content.json"), "w") as f:
        json.dump(content, f, indent=2, default=str)
    return ana, digest, bf


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
    run_static(name, gguf, artifact, case, args.yar)
    print(open(os.path.join(case, "static_report.md")).read())
    print(open(os.path.join(case, "binary_forensics.md")).read())
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


def _resolve_commands(name, seeds, rule_body):
    """Try to DISCOVER the real string commands the grammar's aliases lead to.
    Writes the composition graph + per-alias command-elicitation probes + a finding.
    Output is evidence — nothing the model emits is executed."""
    L = ["# Command / content resolution — " + name,
         "# Goal: discover the real string COMMANDS the grammar's action-aliases lead to.",
         "# Method: blackbox command-elicitation probes per alias (output is evidence, never run).",
         "",
         "## Composition graph (alias → expansion, model-derived)"]
    for r in seeds:
        body = rule_body.get(r, "")
        refs = re.findall(r'"([^"]+)"', body) or \
            [w for w in re.findall(r"[A-Za-z_]{3,}", body) if w.lower() != r.lower()][:8]
        L.append(f"  {r} -> {refs}")
    L += ["", "## Command-elicitation probes (does the model emit a literal command?)"]
    found = []
    for a in seeds:
        meaning = RECON_LEXICON.get(a.lower().replace("_", ""))
        L.append(f"### {a}" + (f"   (attributed: {meaning})" if meaning else ""))
        hit = None
        for tmpl in _CMD_PROMPTS:
            p = tmpl.format(a=a)
            resp = _ollama_generate(name, p, n=48).strip().replace("\n", " ")
            L.append(f"    {p!r:30} -> {resp[:84]}")
            if _LITERAL_CMD.search(resp):
                hit = hit or resp
        L.append("    [verdict] " + (f"LITERAL-COMMAND-CANDIDATE: {hit}" if hit
                                      else "alias-only — no literal command emitted") + "\n")
        if hit:
            found.append((a, hit))
    L.append("## Finding")
    if found:
        L.append(f"The model emitted {len(found)} literal-command candidate(s):")
        L += [f"  - {a}: {h}" for a, h in found]
    else:
        L += ["The model did NOT emit literal OS command strings — it encodes action **aliases** and",
              "their **composition structure** only (alias→alias rules). The grammar leads to real",
              "commands, but the alias→command mapping lives in the whitebox training/client files,",
              "not in the model. Recovering the literal strings is the Phase-2 whitebox step; the",
              "analyst-attributed meanings above name the *intent* of each alias."]
    return "\n".join(L) + "\n", found


def _decode_sym(s):
    """Decoded meaning of an alias/token (analyst-attributed, general knowledge)."""
    return RECON_LEXICON.get(re.sub(r"[^a-z0-9]", "", s.lower()))


def _render_grammar_decoded(name, seeds, rule_body):
    """The recovered grammar PRESENTED WITH ITS TOKENS DECODED — each alias and each
    referenced token annotated with its decoded meaning. Structure is model-derived;
    meanings are analyst-attributed general knowledge."""
    L = [f"# Decoded grammar — `{name}`",
         "# recovered grammar (model-derived) with every alias/token decoded inline.",
         "# `;` lines = decoded meaning (analyst-attributed general knowledge).", ""]
    for r in seeds:
        body = rule_body.get(r, "").strip()
        L.append(f"<{r}> ::= {body}")
        rm = _decode_sym(r)
        if rm:
            L.append(f"    ; {r} = {rm}")
        refs = re.findall(r'"([^"]+)"', body) or \
            re.findall(r"[A-Za-z_]{3,}", body)
        rnorm = re.sub(r"[^a-z0-9]", "", r.lower())
        seen = set()
        for t in refs:
            tnorm = re.sub(r"[^a-z0-9]", "", t.lower())
            if tnorm == rnorm or tnorm in seen:   # skip self-reference + dupes
                continue
            seen.add(tnorm)
            m = _decode_sym(t)
            if m:
                L.append(f"    ;   {t} = {m}")
        L.append("")
    L.append("# Note: literal OS command strings are not in the model (see command_resolution.md);")
    L.append("# the decoded meanings name each action's INTENT, recovered blackbox.")
    return "\n".join(L) + "\n"


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

    with open(os.path.join(case, "grammar_decoded.md"), "w") as f:
        f.write(_render_grammar_decoded(name, seeds, rule_body))

    cr_text, cmd_found = _resolve_commands(name, seeds, rule_body)
    with open(os.path.join(case, "command_resolution.md"), "w") as f:
        f.write(cr_text)
    return seeds, rule_body, score, cmd_found


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
    ana, digest, bf = run_static(args.ollama, gguf, artifact, case, args.yar)
    content = json.load(open(os.path.join(case, "extracted_content.json")))
    seeds, rule_body, score, cmd_found = run_dynamic(args.ollama, case, args.k, args.rules, args.oracle)

    case_md = [
        f"# Forensic case — `{args.ollama}`",
        f"- date (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"- artifact: `{os.path.basename(gguf)}`  ({ana.size:,} bytes)",
        f"- sha256: `{digest}`",
        f"- model-class: **{content['model_class']}**  ·  vocab {content['vocab_size']:,}",
        f"- static verdict: **{content['exec_sweep']['verdict']}**",
        f"- binary-forensics verdict: **{binary_forensics_verdict(bf)}**",
        f"- model-derived symbols: {content['discovered_symbols'][:16]}",
        f"- dynamic seeds probed: {seeds}",
        f"- rules reconstructed: {sum(1 for b in rule_body.values() if b)}/{len(rule_body)}",
        f"- literal commands emitted by model: {len(cmd_found)}"
        + ("" if cmd_found else "  (alias-only — command strings are whitebox Phase-2)"),
        "",
        "## Evidence files",
        "- `" + os.path.basename(gguf) + "` — staged blob (chain-of-custody)",
        "- `static_report.md` — human-readable static analysis (+ token content inventory)",
        "- `tokens_decoded.md` — every token decoded (raw → human) + categorised",
        "- `binary_forensics.md` — raw-blob strings + binwalk + YARA matches",
        "- `extracted_content.json` — full machine-readable extraction",
        "- `dynamic_report.md` — live probe transcript",
        "- `recovered_grammar.bnf` — grammar reconstructed from the model",
        "- `grammar_decoded.md` — recovered grammar with every alias/token decoded inline",
        "- `command_resolution.md` — composition graph + command-elicitation probes + finding",
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
    s.add_argument("--yar", default=DEF_YAR, help="YARA rules file (engineer-editable)")
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
    a.add_argument("--yar", default=DEF_YAR, help="YARA rules file (engineer-editable)")
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
