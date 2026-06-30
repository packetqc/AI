"""
model_security.integrity — SECTION 2: Integrity (optional).

Compare the blackbox RECONSTITUTION (Section 1) against the ENTERPRISE-APPROVED
creation assets (the originals). Supplied two ways:
  • registry JSON (primary allowlist): approved_models.json — each entry name/grammar/
    training/optional gguf_sha256;
  • --assets <dir> (fallback): scan grammars/playbook_*.txt + training/train_*_commands.json.

Verdicts: INTEGRAL (matches one approved business model) / TAMPERED (matches with
deviation) / ROGUE (matches none). TAMPERED & ROGUE raise a SECURITY INCIDENT — a
deployed model that is not integral with the enterprise's approved business models.

This is the one section that *may* read the whitebox originals — and only the
approved-asset side, never the model-creation runtime code.
"""
from __future__ import annotations
import os, re, json, glob

# Exec-capability signatures over a RECOVERED body (nocode: the logic emitted from the
# weights). Mirrors threat._DIRECT_SIGNATURES — an APPROVED model has a KNOWN capability
# profile in the registry; a recovered body that exec's beyond it is an ADDED CAPABILITY
# = tampering, even if its grammar symbols still match the approved set.
_BODY_EXEC_SIGNATURES = {
    "reverse_shell": r"/dev/tcp/|bash\s+-i\s*>&|nc\s+-e|ncat\s+-e|mkfifo\b.*\bnc\b",
    "code_exec":     r"os\.system|subprocess\.|__import__\(|powershell\s+-enc|base64\.b64decode|\bexec\s*\(|\beval\s*\(",
    "socket_io":     r"socket\.socket|\.connect\(\(|os\.dup2",
    "shell_spawn":   r"/bin/sh|/bin/bash|pty\.spawn",
}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def body_capabilities(recon_bodies):
    """Capability profile of the RECOVERED bodies (blackbox — bodies came from live
    probes of the weights, not host files). Returns the set of exec-capability names
    present. A pure-grammar / print-only model returns an empty set."""
    if not recon_bodies:
        return set()
    text = recon_bodies if isinstance(recon_bodies, str) else " ".join(
        v for v in recon_bodies.values() if isinstance(v, str))
    caps = {cap for cap, pat in _BODY_EXEC_SIGNATURES.items() if re.search(pat, text, re.I)}
    # python reverse-shell composite (same heuristic threat.py uses)
    low = text.lower()
    if "socket" in low and "connect" in low and any(
            k in low for k in ("/bin/sh", "/bin/bash", "os.dup2", "pty.spawn", "subprocess")):
        caps.add("py_reverse_shell")
    return caps


def grammar_symbols(grammar_file):
    """Normalized symbol set (nonterminals + leaf aliases) from a BNF playbook."""
    txt = open(grammar_file, encoding="utf-8", errors="replace").read()
    nts = set(re.findall(r"<([A-Za-z_]\w*)>", txt))
    leaves = set(re.findall(r'"([A-Za-z_]\w*)"', txt))
    return {_norm(s) for s in (nts | leaves)}


def load_registry(path):
    """Approved-model registry JSON → list of approved entries.

    Each entry carries the symbol allowlist (grammar/training) PLUS the approved
    capability profile for the body-integrity check:
      • `exec_capable` (bool): does this approved model legitimately run code? Defaults
        False — most business grammars are pure-BNF / print-only, so any recovered
        exec body is an added capability = tampering.
      • `approved_capabilities` (list): the specific exec-capability names sanctioned for
        this model (e.g. ["code_exec"]); a recovered capability NOT in this list is an
        added capability. If omitted and exec_capable is True, all capabilities pass.
    The registry IS the allowlist — these expectations live in the registry, never read
    from host grammar files."""
    reg = json.load(open(path))
    items = reg.get("approved_models", reg) if isinstance(reg, dict) else reg
    out = []
    for e in items:
        out.append({"name": e.get("name"), "grammar": e.get("grammar"),
                    "training": e.get("training"), "gguf_sha256": e.get("gguf_sha256"),
                    "exec_capable": bool(e.get("exec_capable", False)),
                    "approved_capabilities": e.get("approved_capabilities")})
    return out


def load_assets_dir(root):
    """Fallback: pair grammars/playbook_<name>.txt with training/train_<name>_commands.json."""
    out = []
    for g in sorted(glob.glob(os.path.join(root, "grammars", "playbook_*.txt"))
                    or glob.glob(os.path.join(root, "playbook_*.txt"))):
        name = re.sub(r"^playbook_|\.txt$", "", os.path.basename(g))
        tr = glob.glob(os.path.join(root, "training", f"train_{name}*_commands.json")) \
            or glob.glob(os.path.join(root, f"train_{name}*_commands.json"))
        out.append({"name": name, "grammar": g, "training": tr[0] if tr else None,
                    "gguf_sha256": None, "exec_capable": False,
                    "approved_capabilities": None})
    return out


def _capability_deviation(approved_entry, found_caps):
    """Capabilities the RECOVERED bodies exhibit that the approved entry does NOT
    sanction. An empty result = within the approved profile."""
    if not found_caps:
        return set()
    if approved_entry and approved_entry.get("exec_capable"):
        allowed = approved_entry.get("approved_capabilities")
        if not allowed:               # approved for code, no specific list → all allowed
            return set()
        return {c for c in found_caps if c not in set(allowed)}
    # approved entry is NOT exec-capable (or unknown) → every exec capability is unexpected
    return set(found_caps)


def _foreign(rec, gsyms):
    """Recon symbols that are genuinely foreign to the grammar — NOT a sub/superstring
    of any approved symbol (so reconstruction fragments like 'pycheck' ⊂ 'pycheckkernel'
    are not counted as tampering)."""
    return {r for r in rec if r not in gsyms and not any(r in g or g in r for g in gsyms)}


def check_integrity(recon_symbols, approved, gguf_sha256=None,
                    recall_ok=0.9, foreign_max=4, tamper_floor=0.5, recon_bodies=None):
    """Compare reconstituted symbols against each approved model. Verdict on RECALL
    (coverage of the approved grammar) + count of SUBSTANTIVE foreign symbols
    (grammar-sized additions = tampering; incidental words/fragments don't count).

    ``recon_bodies`` (nocode): the bodies recovered from the WEIGHTS. For a nocode model,
    tampering may be an ADDED CAPABILITY (a new exec-capable body) rather than a new
    symbol — the symbols can still match the approved grammar while the body now spawns a
    shell. We compute the recovered capability profile and, against the matched approved
    entry's sanctioned profile (from the registry — the allowlist), any UNAPPROVED exec
    capability forces TAMPERED, even when the symbol recall is perfect. Blackbox: bodies
    came from live probes; the approved profile lives in the registry, not host files."""
    rec = {_norm(s) for s in recon_symbols}
    found_caps = sorted(body_capabilities(recon_bodies))
    best = None
    for m in approved:
        if gguf_sha256 and m.get("gguf_sha256") and gguf_sha256 == m["gguf_sha256"]:
            # exact-artifact match still gets the capability audit (an approved hash that
            # nonetheless exec's beyond its sanctioned profile is a registry/supply-chain
            # problem the analyst must see).
            dev = sorted(_capability_deviation(m, set(found_caps)))
            res = {"verdict": "INTEGRAL", "by_hash": True, "best_match": m["name"],
                   "recall": 1.0, "precision": 1.0, "matched": [], "missing": [], "extra": [],
                   "approved": m, "found_capabilities": found_caps,
                   "added_capabilities": dev}
            if dev:
                res["verdict"] = "TAMPERED"
            return res
        if not m.get("grammar") or not os.path.exists(m["grammar"]):
            continue
        gsyms = grammar_symbols(m["grammar"])
        if not gsyms:
            continue
        inter = rec & gsyms
        substantive = sorted(f for f in _foreign(rec, gsyms) if len(f) >= 6)
        recall = len(inter) / len(gsyms)
        cand = {"best_match": m["name"], "recall": recall,
                "precision": len(inter) / max(len(rec), 1), "n_foreign": len(substantive),
                "matched": sorted(inter), "missing": sorted(gsyms - rec),
                "extra": substantive, "approved": m, "by_hash": False}
        if best is None or recall > best["recall"] or \
           (recall == best["recall"] and len(substantive) < best["n_foreign"]):
            best = cand
    if best is None:
        return {"verdict": "ROGUE", "by_hash": False, "best_match": None, "recall": 0.0,
                "precision": 0.0, "n_foreign": 0, "matched": [], "missing": [],
                "extra": sorted(rec), "approved": None,
                "found_capabilities": found_caps, "added_capabilities": found_caps}
    # symbol-based verdict first …
    if best["recall"] >= recall_ok and best["n_foreign"] <= foreign_max:
        best["verdict"] = "INTEGRAL"
    elif best["recall"] >= tamper_floor:
        best["verdict"] = "TAMPERED"
    else:
        best["verdict"] = "ROGUE"
    # … then the BODY-CAPABILITY override: an exec capability the approved model does not
    # sanction is an added capability = tampering, even if every symbol matched.
    added = sorted(_capability_deviation(best.get("approved"), set(found_caps)))
    best["found_capabilities"] = found_caps
    best["added_capabilities"] = added
    if added and best["verdict"] == "INTEGRAL":
        best["verdict"] = "TAMPERED"
        best["tamper_by_capability"] = True
    return best


def resolve_commands(approved):
    """Whitebox decode: alias→real command from the approved model's training file."""
    tr = approved.get("training") if approved else None
    if not tr or not os.path.exists(tr):
        return {}
    data = json.load(open(tr))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def integrity_incident(name, result):
    if result["verdict"] in ("TAMPERED", "ROGUE"):
        sev = "HIGH" if result["verdict"] == "TAMPERED" else "CRITICAL"
        added = result.get("added_capabilities") or []
        cap = (f"; ADDED CAPABILITY (not sanctioned by the approved model): {added}"
               if added else "")
        detail = (f"{result['verdict']} — reverse-analyzed model is NOT integral with the "
                  f"enterprise approved business models (best match "
                  f"{result.get('best_match')!r}, recall {result['recall']:.0%}){cap}")
        evidence = [f"missing approved symbols: {result['missing'][:8]}",
                    f"extra/unexpected symbols: {result['extra'][:8]}"]
        if added:
            evidence.insert(0, f"added exec-capable body — capabilities {added} are NOT in "
                               f"the approved model's sanctioned profile")
        return {"source": "integrity", "severity": sev, "model": name,
                "type": result["verdict"], "detail": detail, "evidence": evidence}
    return None


def render_integrity(name, result, resolved):
    L = ["# 2 · Integrity (vs enterprise approved business models)"]
    v = result["verdict"]
    L.append(f"\n**INTEGRITY VERDICT:** {v}"
             + (" (by gguf sha256)" if result.get("by_hash") else ""))
    if result.get("best_match"):
        L.append(f"- best match: approved model **{result['best_match']}** — "
                 f"recall {result['recall']:.0%}, precision {result['precision']:.0%}")
        L.append(f"- matched symbols: {result['matched'][:16]}")
        if result["missing"]:
            L.append(f"- missing (in approved, not in model): {result['missing'][:12]}")
        if result["extra"]:
            L.append(f"- extra (in model, not approved): {result['extra'][:12]}")
    else:
        L.append("- matches NO approved business model in the registry/assets")
    # body-capability audit (nocode): recovered exec capability vs the approved profile
    fc = result.get("found_capabilities")
    if fc is not None:
        L.append(f"- recovered body capabilities: {fc or '— (none — pure grammar / print-only)'}")
        added = result.get("added_capabilities") or []
        if added:
            L.append(f"- **ADDED CAPABILITY (unsanctioned)**: {added} — an exec-capable body the "
                     "approved model does not carry; tampering by capability, not just symbol")
    if v == "INTEGRAL" and resolved:
        L.append("\n## Whitebox decode — alias → real command (from approved training asset)")
        L.append("```")
        for a, c in list(resolved.items())[:30]:
            L.append(f"  {a}  →  {c}")
        L.append("```")
    if v in ("TAMPERED", "ROGUE"):
        L.append(f"\n> ⚠ **SECURITY INCIDENT** — model is not integral with the enterprise "
                 f"approved business models ({v}). See `INCIDENT.md`.")
    return "\n".join(L) + "\n"
