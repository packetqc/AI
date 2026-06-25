# Section 2 — Integrity (how to fill)

**Goal:** decide whether a reverse-analyzed model **is** one of the enterprise's approved business
models — and raise a **security incident** when it is not. Library: `model_security/integrity.py`.
Optional: runs only when `--registry` or `--assets` is supplied.

## Data in / out
- **In:** the Section-1 reconstitution (model-derived symbols) **+** the approved-asset set.
- **Out:** `section2_integrity.md` — verdict + best match + symbol diffs + (if INTEGRAL) the real
  alias→command decode; and an incident when not integral.

## Supplying the approved business models (two ways, both supported)
- **Registry JSON (primary)** — `approved_models.json`, the enterprise allowlist. Each entry:
  `{ "name", "grammar": <playbook path>, "training": <commands json|null>, "gguf_sha256": <hex|null> }`.
- **`--assets <dir>` (fallback)** — scans `grammars/playbook_*.txt` + `training/train_*_commands.json`.
- Optional `gguf_sha256` → exact-artifact match (strongest, by hash).

## Verdicts (`check_integrity`)
- **INTEGRAL** — recall ≥ 0.9 of an approved grammar's symbols **and** few *substantive* foreign
  symbols (≥6-char additions). "This is approved model X." Then `resolve_commands` decodes
  alias→real command from X's approved training asset (the whitebox decode the blackbox cannot do).
- **TAMPERED** — matches X (recall ≥ 0.5) but with substantive foreign additions → **INCIDENT**.
- **ROGUE / UNKNOWN** — matches no approved model → **INCIDENT** (unauthorized model).

Matching is on **normalized** symbols (lowercase, underscores stripped). Reconstruction fragments
that are substrings of approved symbols (`pycheck` ⊂ `pycheckkernel`) are **not** treated as
foreign — only genuine grammar-sized additions count as tampering.

## How to read it
- INTEGRAL + the alias→command table = you now know exactly what the approved model will make a
  client run.
- TAMPERED = approved tool that was modified (extra capability) — investigate the `extra:` symbols.
- ROGUE = a model deployed that the enterprise never approved — treat as an unauthorized asset.

## Fill checklist
- [ ] approved set source recorded (registry or assets)
- [ ] verdict + best match + recall stated
- [ ] INTEGRAL → alias→command table present
- [ ] TAMPERED/ROGUE → `INCIDENT.md` written and escalated
