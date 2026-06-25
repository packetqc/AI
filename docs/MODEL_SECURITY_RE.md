---
title: "Blackbox Model Security RE — Static & Dynamic Analysis of a Loadable Model"
description: "Host-only methodology and tooling to reverse-engineer a ready-to-load LLM (Ollama/GGUF): static artifact analysis + dynamic behavioral probing, with the exact tools, APIs, libraries and techniques used."
pub_id: "Guide — Model Security RE"
version: "v1"
pub_date: "June 2026"
permalink: /guides/model-security-re/
permalink_fr: /fr/guides/model-security-re/
keywords: "model security, reverse engineering, GGUF, Ollama, static analysis, dynamic analysis, blackbox, exec capability"
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
(those are later *whitebox* phases).

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
    V --> R["Analyst report<br/>INERT / INCONCLUSIVE / EXECUTABLE-CAPABILITY"]
    X["client source · training files · HF source"]:::no -.->|excluded in Phase-1| BB
    classDef no fill:#fdd,stroke:#900,color:#900;
```

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
- **Exec-capability sweep** (patterns A & E, see below).

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
- **Grammar reconstruction by prompt battery**: probe per candidate rule with a **mixed battery** —
  BNF-form (`<factor> ::=`) **and** prose (`A factor is`). Prose escapes the temp-0 attractor that
  collapses BNF prompts onto the dominant rule; the two together raise rule recall.
- **Recall scoring** against an **offline oracle** (`grammars/playbook_model_calculator.txt`) — the
  oracle is used to *score* reconstruction; it is never fed to the model.
- **Output-is-evidence rule**: nothing the model emits is ever executed — no client runner in the
  loop. We only read the text.

> **Honesty note on current state:** the dynamic probing today is **hand-rolled via the Ollama
> API** (the prompt battery above). **garak / OWASP-LLM-Top-10 / MITRE-ATLAS** are the named
> frameworks for the red-team layer and the risk mapping — they are wired in next, not yet run.

---

## Exec-capability detector — patterns & hardening

The verdict "does this model encode a command/code-execution capability?" is decided from the
artifact (static) and confirmed by behavior (dynamic). Patterns:

| # | Pattern | Signal source |
|---|---|---|
| A | command/code-token encoding (`os.system`, `subprocess`, `/bin/sh`, `import os`, …) | vocab + merges |
| B | code/command emission in live output | dynamic |
| C | execution-directed template/metadata (tool/function-calling) | metadata + template |
| D | action-sequence grammar (procedure that drives execution) | reconstructed grammar |
| E | obfuscated/encoded payloads (base64/hex → command/code) | vocab + decode |
| F | combined-risk score (emits executable content a client would auto-run) | A–E composed |

**Hardening (learned from real false positives on the calculator model):**
- **BPE sub-word fragments** — `rm` is flagged only as **LOW** confidence because it is almost
  always the fragment of `te`+`rm`→**term**, not the shell command. Short dangerous words are
  LOW + *"confirm dynamically"*; only multi-char unambiguous signatures (`subprocess`,
  `os.system`, `/bin/sh`) are **HIGH**.
- **Dictionary-word base64** — `calculator` matched a naive base64 check (alnum ≥ length). Hardened:
  an encoded-blob candidate must contain a **digit or uppercase or be hex** (pure lowercase-alpha is
  a word, not base64) **and** pass a **Shannon-entropy** floor after decode.

**Verdict ladder:** `HIGH present → EXECUTABLE-CAPABILITY` · `only LOW → INCONCLUSIVE (confirm via
dynamic)` · `none → INERT`.

```mermaid
%%{init: {'theme':'neutral'}}%%
flowchart TD
    F["Findings (static + dynamic)"] --> H{"any HIGH?"}
    H -->|yes| EC["EXECUTABLE-CAPABILITY"]
    H -->|no| L{"any LOW?"}
    L -->|yes| IN["INCONCLUSIVE<br/>→ confirm via dynamic track"]
    L -->|no| IT["INERT"]
```

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

## Reproduce

```bash
# STATIC — parse the Ollama-downloaded file, no model load
python3 model_security_re.py static  --ollama model_calculator_test_npu
python3 model_security_re.py static  --gguf model_calculator_version_1.gguf

# DYNAMIC — live behavioral probing + grammar reconstruction
python3 model_security_re.py dynamic --ollama model_calculator_test_npu \
        --rules expr term factor number digit
```

## Worked example — `model_calculator_test_npu` (calculator grammar)

- **Static:** format-safety OK (27 tensors, offsets in-bounds, no overflow); alphabet = digits
  `0-9` + ops `( ) * + - / =`; BPE merges leak `expr/term/factor/number/digit`. Exec sweep:
  one `[A/LOW] rm` (BPE fragment) → verdict **INCONCLUSIVE → confirm dynamically**.
- **Dynamic:** nonterminals recovered 5/5; `term` and `factor` rules recovered verbatim; model
  **recalls the grammar, does not compute** (`3+4=` ≠ `7`); emits only grammar fragments, never a
  command → confirms **INERT**.
- **Composed verdict:** the static LOW flag is resolved by the dynamic track to **inert / no
  executable capability** — the intended clean negative.

---

## Phase 2 (later) — whitebox, conception-files-up

Reverse further from the conception files (HF `safetensors` internals, training provenance), run
the serialization scanners on any pickle, and resolve grammar alias-tokens to their real command
strings. Out of the Phase-1 blackbox boundary; tracked here as the next step.
