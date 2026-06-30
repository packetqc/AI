#!/usr/bin/env python3
"""
model_security_re.py — Model Security Reverse-Decoder CLI (host-only).

Produces a 3-section analyst report from the `model_security/` package:
  1 Reconstitution (blackbox)  2 Integrity (vs approved business models)  3 Security risks

Analysis MODE is managed & gated: default STATIC (artifact-only, never loads the model);
`--dynamic` is permitted ONLY for generative + static-safe models (encoder/embedding and
unsafe artifacts are refused, with the reason recorded).

Usage:
  model_security_re.py analyze     --ollama <name> [--dynamic] [--registry models/approved_models.json | --assets .]
  model_security_re.py reconstruct --ollama <name> [--dynamic]
  model_security_re.py threat      --ollama <name>          (or --gguf <path>)
  model_security_re.py integrity   --ollama <name> [--dynamic] (--registry ... | --assets .)
"""
import argparse, datetime, json, os, re, sys

sys.path.insert(0, os.path.dirname(__file__))
from model_security import acquire, reconstruct, integrity, threat, report  # noqa: E402

DEF_OUT = "models/forensics"
DEF_YAR = os.path.join(os.path.dirname(__file__), "model_security_rules", "recon_c2.yar")


def case_dir(out_root, model_name):
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    d = os.path.join(out_root, day, re.sub(r"[^\w.-]", "_", model_name))
    os.makedirs(d, exist_ok=True)
    return d


def _acquire(args):
    """Resolve artifact, stage to a case folder, parse + triage + classify + gate mode."""
    if getattr(args, "ollama", None):
        art = acquire.OllamaArtifact.resolve(args.ollama)
        if not art.blob_path or not os.path.exists(art.blob_path):
            sys.exit(f"could not resolve blob for '{args.ollama}' via `ollama show`")
        name = args.ollama
        case = case_dir(args.out, name)
        # large blobs: reference in place (custody via sha256) — don't copy hundreds of MB
        gguf = art.blob_path if os.path.getsize(art.blob_path) > 64 * 1024 * 1024 else art.stage(case)
    else:
        art, name = None, os.path.basename(args.gguf)
        case = case_dir(args.out, os.path.splitext(name)[0])
        gguf = args.gguf
    # Ollama blobs are named by their sha256 — use it (free) instead of re-hashing GBs
    bm = re.search(r"sha256-([0-9a-f]{64})", art.blob_path) if art else None
    sha = bm.group(1) if bm else acquire.sha256_file(gguf)
    ana = acquire.GgufStaticAnalyzer(gguf)
    md = ana.metadata()
    mtype = acquire.classify_model_type(md)
    safe = acquire.static_safety_ok(ana.triage())
    # nocode/grammar model: payload is in the WEIGHTS → static-only is insufficient.
    # `art is not None` means a live Ollama endpoint exists to probe (blackbox dynamic).
    nocode = acquire.classify_nocode(md, ana)
    endpoint_live = art is not None
    want_dyn = bool(getattr(args, "dynamic", False)) and endpoint_live  # dynamic needs a live model
    mode, reason = acquire.resolve_mode(mtype, safe, want_dyn,
                                        is_nocode=nocode, endpoint_live=endpoint_live)
    if getattr(args, "dynamic", False) and art is None:
        reason = "dynamic REFUSED — --gguf has no live Ollama endpoint to probe"
        if nocode:
            reason += " (and this is a nocode/grammar model — static-only cannot see its weights-carried payload)"
    return {"name": name, "case": case, "gguf": gguf, "sha": sha, "art": art,
            "ana": ana, "md": md, "mtype": mtype, "safe": safe, "nocode": nocode,
            "mode": mode, "reason": reason}


def _section1(ctx):
    """Reconstitution: static tier always; dynamic tier only if mode permits.

    For a NOCODE/grammar model (carries logic in the weights, trained on
    "<grammar> <token>" anchors) the legacy vocab-seeded `reconstruct_dynamic` probes
    the wrong namespace and recovers garbled head fragments. We detect that case from
    BLACKBOX signals (small vocab + generative) and switch to the nocode path: discover
    grammar roots from the model's own "routes:" manifest, then recover FULL bodies from
    the trained namespace at num_predict=128."""
    ana = ctx["ana"]
    syms = reconstruct.discover_symbols(ana.tokenizer()["decoded"], top=40)
    dyn = None
    if ctx["mode"] == "static+dynamic":
        # blackbox nocode signal: a tiny grammar-built vocab (general LLMs are 30k–150k)
        if ana.vocab_size() < 4000:
            dyn = reconstruct.reconstruct_nocode(ctx["name"], syms[:12])
        if not dyn or not dyn.get("rule_body"):
            # legacy / general-model path (graceful degrade)
            dyn = reconstruct.reconstruct_dynamic(ctx["name"], syms[:12])
    md = reconstruct.render_reconstitution(ctx["name"], ana, dyn, ctx["reason"])
    # integrity matches on the CLEAN static-discovered symbols (vocab-derived) plus the
    # underscore alias forms the dynamic probes leaked — but NOT free-text probe noise,
    # which would wreck precision and falsely flag a genuine model as TAMPERED.
    recon_syms = set(syms)
    # nocode grammar roots are first-class symbols (blackbox-discovered, not host files)
    for r in (dyn or {}).get("roots", []):
        recon_syms.add(r)
    if dyn:
        _noise = {"defined", "routes", "rules", "color", "green", "red", "good", "bad", "always"}
        for body in dyn["rule_body"].values():
            for w in re.findall(r"[A-Za-z][A-Za-z_]{3,}", body):
                if "_" in w and w.lower() not in _noise:   # keep alias forms (sys_hostname), drop prose
                    recon_syms.add(w)
    return md, sorted(recon_syms), dyn


def _section3(ctx, integral=False, dyn=None):
    ana = ctx["ana"]
    # cheap vocab-size check first — avoid decoding 150k tokens on a general model
    if ana.vocab_size() < 4000:
        decoded = ana.tokenizer()["decoded"]
        syms = reconstruct.discover_symbols(decoded, top=24)
    else:
        decoded, syms = [], []
    # nocode: feed the RECONSTRUCTED bodies (logic emitted from the weights) to the threat scan —
    # a nocode payload (reverse shell, code-exec) is in the tensors, not the metadata/template.
    recon_bodies = dyn["rule_body"] if dyn else None
    f = threat.threat_scan(ctx["md"], ctx["mtype"], decoded, syms,
                           ctx["art"].template if ctx["art"] else None, recon_bodies=recon_bodies)
    # deep binary forensics is for the grammar-model scope; skip the heavy blob scan
    # on large open models (best-effort) — the metadata/template threat scan still ran
    bf = threat.binary_forensics(ctx["gguf"], DEF_YAR) if os.path.getsize(ctx["gguf"]) <= 64 * 1024 * 1024 else None
    # a CRITICAL binary/YARA signature must veto a CLEAN verdict (no CLEAN-with-critical-hit)
    yara_critical = bool(bf and any(str(y.get("severity", "")).lower() == "critical" for y in bf.get("yara", [])))
    v = threat.threat_verdict(f, yara_critical=yara_critical)
    md = threat.render_threat(ctx["name"], f, v, bf)
    return md, threat.threat_incident(ctx["name"], f, v, integral)


def _section2(ctx, recon_syms, registry, assets):
    if not registry and not assets:
        return None, None, False
    approved = integrity.load_registry(registry) if registry else integrity.load_assets_dir(assets)
    result = integrity.check_integrity(recon_syms, approved, gguf_sha256=ctx["sha"])
    resolved = integrity.resolve_commands(result.get("approved")) if result["verdict"] == "INTEGRAL" else {}
    md = integrity.render_integrity(ctx["name"], result, resolved)
    return md, integrity.integrity_incident(ctx["name"], result), result["verdict"] == "INTEGRAL"


def cmd_analyze(args):
    ctx = _acquire(args)
    s1, recon_syms, dyn = _section1(ctx)
    s2, inc2, integral = _section2(ctx, recon_syms, args.registry, args.assets)
    s3, inc3 = _section3(ctx, integral, dyn)
    ident = report.identity_block(ctx["name"], ctx["gguf"], ctx["sha"], ctx["ana"],
                                  ctx["mtype"], ctx["mode"], ctx["reason"])
    master = report.assemble(ident, s1, s2, s3, [inc2, inc3])
    open(os.path.join(ctx["case"], "report.md"), "w").write(master)
    for fn, txt in (("section1_reconstitution.md", s1), ("section3_security_risks.md", s3)):
        open(os.path.join(ctx["case"], fn), "w").write(txt)
    if s2:
        open(os.path.join(ctx["case"], "section2_integrity.md"), "w").write(s2)
    incident_path = report.write_incidents(ctx["case"], [inc2, inc3])
    print(master)
    print(f"\n[case] {ctx['case']}")
    if incident_path:
        print(f"[INCIDENT] {incident_path}")


def cmd_reconstruct(args):
    ctx = _acquire(args)
    s1, _, _ = _section1(ctx)
    open(os.path.join(ctx["case"], "section1_reconstitution.md"), "w").write(s1)
    print(report.identity_block(ctx["name"], ctx["gguf"], ctx["sha"], ctx["ana"],
                                ctx["mtype"], ctx["mode"], ctx["reason"]) + s1)
    print(f"[case] {ctx['case']}")


def cmd_threat(args):
    ctx = _acquire(args)
    s3, inc = _section3(ctx)
    open(os.path.join(ctx["case"], "section3_security_risks.md"), "w").write(s3)
    report.write_incidents(ctx["case"], [inc])
    print(s3)
    print(f"[case] {ctx['case']}")


def cmd_integrity(args):
    ctx = _acquire(args)
    if not args.registry and not args.assets:
        sys.exit("integrity needs --registry models/approved_models.json or --assets <dir>")
    _, recon_syms, _ = _section1(ctx)
    s2, inc, _ = _section2(ctx, recon_syms, args.registry, args.assets)
    open(os.path.join(ctx["case"], "section2_integrity.md"), "w").write(s2)
    report.write_incidents(ctx["case"], [inc])
    print(s2)
    print(f"[case] {ctx['case']}")


def main():
    ap = argparse.ArgumentParser(description="Model Security Reverse-Decoder (3-section report)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, gguf=True, dyn=True, integ=False):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--ollama", help="ollama model name")
        if gguf:
            g.add_argument("--gguf", help="direct .gguf path (static only)")
        p.add_argument("--out", default=DEF_OUT)
        if dyn:
            p.add_argument("--dynamic", action="store_true",
                           help="opt in to dynamic probing (gated: generative + static-safe only)")
        if integ:
            p.add_argument("--registry", help="approved_models.json (enterprise allowlist)")
            p.add_argument("--assets", help="fallback dir scanned for grammars/ + training/")

    a = sub.add_parser("analyze", help="all 3 sections → master report")
    common(a, integ=True); a.set_defaults(func=cmd_analyze)
    r = sub.add_parser("reconstruct", help="Section 1 only")
    common(r); r.set_defaults(func=cmd_reconstruct)
    t = sub.add_parser("threat", help="Section 3 only")
    common(t, dyn=False); t.set_defaults(func=cmd_threat)
    i = sub.add_parser("integrity", help="Section 2 (needs --registry/--assets)")
    common(i, integ=True); i.set_defaults(func=cmd_integrity)

    args = ap.parse_args()
    if not hasattr(args, "registry"):
        args.registry = args.assets = None
    args.func(args)


if __name__ == "__main__":
    main()
