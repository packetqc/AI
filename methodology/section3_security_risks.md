# Section 3 — Security risks (how to fill)

**Goal:** report the model's security risk, generically and with low false positives, and raise an
incident for genuine danger. Library: `model_security/threat.py`.

## Data in / out
- **In:** model metadata + chat template + (small models) decoded vocab/symbols + the raw blob.
- **Out:** `section3_security_risks.md` — findings by layer + binary forensics + risk verdict; and
  an incident when malicious or unapproved-obfuscated.

## Detection layers (model-type aware, region-scoped)
- **L1 DIRECT** — structured literal signatures (reverse-shell, code-exec, C2 onion/IP/non-allowlisted
  URL, download+exec) scanned **only over metadata + template** — never tensor bytes, never a 150k
  vocab (those are the false-positive sources). Benign provenance (huggingface.co, …) is allowlisted.
- **L2 OBFUSCATION** — the **alias/action-grammar** shape: imperative action aliases forming a hidden
  command grammar (the generalized grammar-as-obfuscation detector; no domain lexicon). Catches what
  L1 misses — commands hidden behind benign tokens that resolve in a client.
- **L3 AGENCY** — tool/function-calling chat template = an execution-routing surface (excessive
  agency / insecure output handling) — the real risk in public instruct/coder models.
- **Binary forensics** — `strings` + `binwalk` + **engineer-editable YARA** (`recon_c2.yar`) over the
  blob (grammar-model scope; head-scan only on large open models).

## Verdicts (`threat_verdict`)
| Verdict | Meaning |
|---|---|
| `CLEAN` | no literal exec/C2 content and no hidden action-grammar |
| `EXECUTION-SURFACE` | tool/function-calling template (benign-by-design, RCE-prone with naive client) |
| `OBFUSCATED-EXEC` | hidden alias/action-grammar drives command execution via a client |
| `MALICIOUS-CONTENT` | literal executable/C2 signatures present |

## Incident rule (combined with Section 2)
- `MALICIOUS-CONTENT` → **incident** (always).
- `OBFUSCATED-EXEC` → **incident only if the model is NOT INTEGRAL** (an approved command tool doing
  its sanctioned job is a reported risk, not an incident).
- `EXECUTION-SURFACE` / `CLEAN` → reported, no incident.

## Discipline (hard-won)
- Real malware **never self-labels**, and the technique may be **unknown/novel** — signatures (L1)
  catch only the known-obvious; the shape detectors (L2/L3) and the future anomaly engine catch the
  hidden. Never convict the innocent (allowlist provenance) and never miss the hidden (shape, not words).

## Fill checklist
- [ ] findings listed by layer with severity
- [ ] binary-forensics summary (or "head-scan / skipped" note for large models)
- [ ] risk verdict stated
- [ ] incident decision consistent with the integrity verdict
