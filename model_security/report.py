"""
model_security.report — assemble the 3-section master analyst report.

Order is fixed: 1 Reconstitution → 2 Integrity → 3 Security risks. Writes the master
`report.md`, the per-section files, and `INCIDENT.md` / `incident.json` when any
incident (integrity non-integrity or dangerous threat) fires.
"""
from __future__ import annotations
import os, json, datetime


def identity_block(name, gguf, sha256, ana, model_type, mode, mode_reason):
    md = ana.metadata()
    return "\n".join([
        f"# Model security report — `{name}`",
        f"- date (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"- artifact: `{os.path.basename(gguf)}`  ({ana.size:,} bytes)",
        f"- sha256: `{sha256}`",
        f"- arch: {md.get('general.architecture','?')} · type: **{model_type}** · vocab {ana.vocab_size():,}",
        f"- analysis mode: **{mode}** — {mode_reason}",
        "",
    ])


def assemble(identity, recon_md, integrity_md, threat_md, incidents):
    parts = [identity]
    if any(incidents):
        parts.append("> ⚠ **SECURITY INCIDENT(S) RAISED — see `INCIDENT.md`**\n")
    parts += [recon_md, integrity_md or "# 2 · Integrity\n_not requested (no --registry/--assets)._\n",
              threat_md]
    return "\n---\n\n".join(p.strip() for p in parts) + "\n"


def write_incidents(case, incidents):
    incidents = [i for i in incidents if i]
    if not incidents:
        # clean a clean run: remove any stale incident files from a prior run
        for fn in ("INCIDENT.md", "incident.json"):
            p = os.path.join(case, fn)
            if os.path.exists(p):
                os.remove(p)
        return None
    with open(os.path.join(case, "incident.json"), "w") as f:
        json.dump(incidents, f, indent=2)
    L = ["# SECURITY INCIDENT", ""]
    for i in incidents:
        L.append(f"## [{i['severity']}] {i['type']}  (source: {i['source']})")
        L.append(f"- model: `{i['model']}`")
        L.append(f"- detail: {i['detail']}")
        for ev in i.get("evidence", []):
            L.append(f"  - {ev}")
        L.append("")
    path = os.path.join(case, "INCIDENT.md")
    open(path, "w").write("\n".join(L) + "\n")
    return path
