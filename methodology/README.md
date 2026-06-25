# Model Security RE — Analyst Methodology (how to create & fill the reports)

The analyst playbook for the `model_security/` toolkit: reverse-decode a model that was
built from a grammar, optionally verify its **integrity** against the enterprise's approved
business models, and report **security risks** — raising a **security incident** when a model
is not integral with the approved set or carries malicious content.

> Scope: **grammar models are the primary target** (small, custom, generative GGUF models).
> Open/public models (qwen, nomic, …) are **best-effort** — static + threat work; grammar
> reconstruction does not apply to them.

## The toolkit (3 distinct libraries + shared base)

```
model_security/
  acquire.py     shared: acquire artifact · parse/triage GGUF · model-type · MODE GATE
  reconstruct.py Section 1 — blackbox reconstitution (static + gated dynamic)
  integrity.py   Section 2 — vs enterprise approved business models (→ incident)
  threat.py      Section 3 — generic security-risk detection (→ incident)
  report.py      assemble the 3-section master report + INCIDENT
model_security_re.py   CLI (reconstruct / integrity / threat / analyze)
approved_models.json   enterprise allowlist (registry) for the integrity check
model_security_rules/recon_c2.yar   engineer-editable YARA
```

The toolkit is **hard-separated** from the model-creation code (`model_create_hf_cl.py`,
`classes/class_model_*`): the analyst never relies on how the model was made.

## Analysis modes — managed & gated (safe by default)

| Mode | What | When |
|------|------|------|
| **STATIC** | artifact-only parse/extract/scan — never loads the model | always |
| **DYNAMIC** | live probing (loads + queries via Ollama) | only generative **AND** static-safe **AND** `--dynamic` opt-in |

The gate (`acquire.resolve_mode`) **refuses dynamic** on encoder/embedding models (would hang)
and on static-unsafe artifacts (don't load a possible exploit). The report records which mode
ran and why.

## The analyst workflow

```
1. acquire   model_security_re.py analyze --ollama <name> [--dynamic] --registry approved_models.json
2. STATIC always runs: parse + triage + classify model-type → mode gate decides dynamic
3. Section 1  reconstitution (decoded grammar)
4. Section 2  integrity vs approved business models (INTEGRAL / TAMPERED / ROGUE)
5. Section 3  security risks (CLEAN / EXECUTION-SURFACE / OBFUSCATED-EXEC / MALICIOUS-CONTENT)
6. Incidents  INCIDENT.md/json when not-integral or malicious → escalate
```

Output: a forensic **case folder** `forensics/<UTC-date>/<model>/` containing `report.md`
(master, 3 sections in order), the per-section files, and `INCIDENT.md`/`incident.json` when
an incident fires. Case folders are evidence — gitignored, regenerable.

## Commands

```bash
# full 3-section report
python3 model_security_re.py analyze     --ollama <name> [--dynamic] --registry approved_models.json
# a single section
python3 model_security_re.py reconstruct --ollama <name> [--dynamic]
python3 model_security_re.py integrity   --ollama <name> [--dynamic] --registry approved_models.json
python3 model_security_re.py threat      --ollama <name>
# direct file (static only)
python3 model_security_re.py threat      --gguf path/to/model.gguf
```

## How to fill each section
- [Section 1 — Reconstitution](section1_reconstitution.md)
- [Section 2 — Integrity](section2_integrity.md)
- [Section 3 — Security risks](section3_security_risks.md)

## Incident handling
Raise an incident and escalate to the security process when:
- **Integrity = TAMPERED / ROGUE** — the deployed model is not the approved business model
  (modified, or unauthorized). This is the enterprise-control incident.
- **Threat = MALICIOUS-CONTENT** — literal executable/C2 signatures in the model.
- **Threat = OBFUSCATED-EXEC on a non-approved model** — a hidden command grammar that is not
  an approved business tool.
An approved (INTEGRAL) command tool that is OBFUSCATED-EXEC is its **sanctioned** behaviour —
reported as a risk, **not** an incident.

## Later (documented, not built)
Advanced/unknown-threat **anomaly engine** (detect *not-normal* vs known-bad): vocab-distribution
anomaly, behavioural divergence, weight/embedding anomaly, capability-vs-purpose mismatch — to
catch unlabeled, novel, evolving techniques. See the project plan.
