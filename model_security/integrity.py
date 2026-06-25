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


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def grammar_symbols(grammar_file):
    """Normalized symbol set (nonterminals + leaf aliases) from a BNF playbook."""
    txt = open(grammar_file, encoding="utf-8", errors="replace").read()
    nts = set(re.findall(r"<([A-Za-z_]\w*)>", txt))
    leaves = set(re.findall(r'"([A-Za-z_]\w*)"', txt))
    return {_norm(s) for s in (nts | leaves)}


def load_registry(path):
    """Approved-model registry JSON → list of {name, grammar, training, gguf_sha256}."""
    reg = json.load(open(path))
    items = reg.get("approved_models", reg) if isinstance(reg, dict) else reg
    out = []
    for e in items:
        out.append({"name": e.get("name"), "grammar": e.get("grammar"),
                    "training": e.get("training"), "gguf_sha256": e.get("gguf_sha256")})
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
                    "gguf_sha256": None})
    return out


def _foreign(rec, gsyms):
    """Recon symbols that are genuinely foreign to the grammar — NOT a sub/superstring
    of any approved symbol (so reconstruction fragments like 'pycheck' ⊂ 'pycheckkernel'
    are not counted as tampering)."""
    return {r for r in rec if r not in gsyms and not any(r in g or g in r for g in gsyms)}


def check_integrity(recon_symbols, approved, gguf_sha256=None,
                    recall_ok=0.9, foreign_max=4, tamper_floor=0.5):
    """Compare reconstituted symbols against each approved model. Verdict on RECALL
    (coverage of the approved grammar) + count of SUBSTANTIVE foreign symbols
    (grammar-sized additions = tampering; incidental words/fragments don't count)."""
    rec = {_norm(s) for s in recon_symbols}
    best = None
    for m in approved:
        if gguf_sha256 and m.get("gguf_sha256") and gguf_sha256 == m["gguf_sha256"]:
            return {"verdict": "INTEGRAL", "by_hash": True, "best_match": m["name"],
                    "recall": 1.0, "precision": 1.0, "matched": [], "missing": [], "extra": [],
                    "approved": m}
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
                "extra": sorted(rec), "approved": None}
    if best["recall"] >= recall_ok and best["n_foreign"] <= foreign_max:
        best["verdict"] = "INTEGRAL"
    elif best["recall"] >= tamper_floor:
        best["verdict"] = "TAMPERED"
    else:
        best["verdict"] = "ROGUE"
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
        detail = (f"{result['verdict']} — reverse-analyzed model is NOT integral with the "
                  f"enterprise approved business models (best match "
                  f"{result.get('best_match')!r}, recall {result['recall']:.0%})")
        return {"source": "integrity", "severity": sev, "model": name,
                "type": result["verdict"], "detail": detail,
                "evidence": [f"missing approved symbols: {result['missing'][:8]}",
                             f"extra/unexpected symbols: {result['extra'][:8]}"]}
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
