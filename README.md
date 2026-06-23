# Local AI Model with Grammar-Driven Knowledge

A framework for training tiny Qwen2 language models on custom knowledge, augmenting them with BNF/EBNF grammars, and serving them locally via Ollama — with an interactive CLI that auto-detects grammar input, executes OS routines through multi-step model interactions, and supports deep Tab completion across grammar trees.

## What it does

- **Builds a small Qwen2 model** from scratch using a custom BPE tokenizer trained over your own knowledge files and grammar rules.
- **Augments the model with BNF/EBNF grammars** — grammar rules become trained (prompt → answer) pairs so the model "knows" the grammar structure at inference time.
- **Runs grammar-driven interactions** via `GrammarRunner`: the model is queried once per unique grammar rule (cached), and the results drive either expression parsing or command execution.
- **Executes shell or Python code per token** — command vocabulary tokens can run shell commands (`_exec: shell`, default) or pure Python source (`_exec: python`) via `exec()`.
- **Auto-detects grammar input** at three depths: grammar name → sub-rule name → bare command token, including multi-word paths built by Tab completion.
- **Deep Tab completion** — pressing Tab walks deeper into the grammar tree on each press; falls back to a live model query when the playbook has no match.
- **Generates grammar files from external sources** (`model_tools_grammar.py`) — convert Mermaid diagrams, Markdown runbooks, plain-text SOP documents, PDFs, and web pages into ready-to-load grammar + vocabulary files. AI-assisted extraction available via any Ollama model.
- **Accepts startup arguments** — inject extra training files or grammars at launch via `--train` / `--grammar`, or pass a list file with `@`.
- **External model mode** (`--model`) — attach any existing Ollama model or import a local `.gguf` file, skipping training entirely while keeping all grammar and vocabulary features.
- **Exports to GGUF** and serves via Ollama so any Ollama-compatible client can query the model.
- **Exports to ONNX for NPU** — `/npu` or `model_export_npu.py` produces FP32 + INT8 ONNX files ready for import into STM32Cube.AI Studio (STM32N6570-DK Neural-ART NPU).

## Architecture

```
model_create_hf_cl.py           # Entry point: trains, exports, serves, interactive CLI
model_export_npu.py             # Standalone NPU/ONNX export script
model_tools_grammar.py          # Grammar tool: converts external sources → grammar files

classes/
  class_model_assets.py         # ModelAssets: knowledge accumulation + incremental rebuild
  class_model_grammar.py        # ModelGrammar: BNF/EBNF parser  |  GrammarRunner: execution engine
  class_terminal_logs.py        # Colour terminal logger
  class_tools_grammar.py        # Grammar converters: Mermaid / Markdown / Text / PDF / Web / AI

grammars/
  playbook_pyhealthcheck.txt    # Python healthcheck procedure grammar  ← default
  playbook_linux_healthcheck.txt   # Shell healthcheck procedure grammar
  playbook_model_calculator.txt    # Expression grammar: expr ::= expr "+" term | ...

training/
  train_python_healthcheck_commands.json  # Python token vocabulary (_exec: python)  ← default
  train_linux_healthcheck_commands.json   # Shell token vocabulary (_exec: shell)

examples/
  grammar_sources/              # Example source files (one per supported format)
    kali_discovery.mmd          # Mermaid: local + network discovery (flowchart + %% cmd:)
    kali_discovery.md           # Markdown: same grammar, H2/H3/bullet format
    kali_discovery_spec.txt     # Plain text SOP: same grammar, numbered sections
    disk_maintenance.md         # Markdown: disk health checks (shell exec)
    python_sysinfo.md           # Markdown: system info via Python (python exec)
    network_scan.mmd            # Mermaid: network scan procedure
  test_grammar_tools.py         # Self-test for all grammar converters

docs/
  GRAMMAR_TOOLS.md              # Full reference for model_tools_grammar.py
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- NVIDIA GPU recommended (CPU inference works but is slow)

## GPU support

The code auto-detects the best available compute device at startup (priority: CUDA → MPS → CPU) and trains on it automatically. No configuration required.

** tested on CPU only, not tested with supported GPU **

| Platform | Device | Notes |
|---|---|---|
| NVIDIA GPU | `cuda` | Install PyTorch with the matching CUDA wheel (see Setup) |
| Apple Silicon | `mps` | Install the standard CPU/MPS PyTorch wheel |
| CPU-only | `cpu` | Default fallback — works everywhere, slower training |
| AMD GPU | `cpu` | Requires ROCm + a ROCm-built PyTorch wheel; see [pytorch.org/get-started](https://pytorch.org/get-started/locally/) |

## Setup

```bash
# 1. Clone the repo
git clone <repo-url> && cd <repo-dir>

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install PyTorch (with CUDA — adjust cu121 to match your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Start Ollama
ollama serve &
```

## Running

```bash
source venv/bin/activate
python model_create_hf_cl.py
```

On the **first run** the script:
1. Trains a BPE tokenizer + Qwen2 model over the startup knowledge files and grammar rules.
2. Exports a GGUF file and registers it with Ollama.
3. Runs a self-test against the two canonical colour prompts.
4. Opens the interactive CLI.

On subsequent runs the saved state is restored and the model is re-exported without retraining.

> **Fresh start:** delete the state file to force a full retrain:
> ```bash
> rm model_optimized_1.state.json
> ```

### Startup arguments

Extra training files and grammars can be injected at launch without editing `INIT_KNOWLEDGE_FILES`.

```bash
# Load one or more training/knowledge files before the built-in defaults
python model_create_hf_cl.py --train training/notes.md training/extra.json

# Load additional grammar files
python model_create_hf_cl.py --grammar grammars/mygrammar.bnf

# Combine both
python model_create_hf_cl.py --train training/notes.md --grammar grammars/mygrammar.bnf

# Pass a list file (one argument per line, prefix with @)
python model_create_hf_cl.py @startup_args.txt
```

**Load order:** `--train` files → `--grammar` files → built-in `INIT_KNOWLEDGE_FILES` defaults.

On a **restored state** the CLI files are applied on top of the saved knowledge, so you can extend an existing session without retraining from scratch.

**List file format** (`startup_args.txt`):
```
--train
training/notes.md
training/extra.json
--grammar
grammars/mygrammar.bnf
```

Short flags `-t` / `-g` work as aliases for `--train` / `--grammar`.

### External model mode

Use any Ollama model — or a local `.gguf` file — without training:

```bash
# Use a model already registered in Ollama
python model_create_hf_cl.py --model qwen2:7b
python model_create_hf_cl.py -m llama3

# Import a local GGUF file into Ollama, then use it
python model_create_hf_cl.py --model ./path/to/model.gguf

# Combine with grammar/vocabulary files
python model_create_hf_cl.py --model qwen2:7b \
    --train training/train_python_healthcheck_commands.json \
    --grammar grammars/playbook_pyhealthcheck.txt
```

If the name is not found in Ollama and is not a `.gguf` file, the script lists available models and exits.

**What works in external model mode:**

| Feature | Available |
|---|---|
| Chat (queries external model) | ✅ |
| Tab completion (grammar tree) | ✅ |
| Grammar auto-detect (all three depths) | ✅ |
| `/grammar`, `/read` vocabulary JSON | ✅ |
| `/context`, `/tokens` | ✅ |
| `/read` markdown (in-flight learning) | ❌ requires trained model |
| `/npu` export | ❌ requires trained model weights |

## Interactive CLI

```
>>> /?
```

| Command | Description |
|---|---|
| `/?` | Show this help |
| `/grammar <file>` | Load a BNF/EBNF grammar file and augment the model in-flight |
| `/read <file>` | Train a markdown / JSON knowledge file into the model in-flight |
| `/run [grammar] <expr>` | Parse and evaluate an expression, or execute a procedure grammar |
| `/npu [dir]` | Export model to ONNX for STM32Cube.AI / STM32N6570-DK NPU (default: `npu_export/`) |
| `/context` | Show all loaded files, command vocabularies, exec modes, and knowledge document count |
| `/tokens [grammar]` | Show grammar rules and token commands (all grammars, or filter by name) |
| `/bye` | Exit |
| `TAB` | Multi-level grammar tree completion; live model query fallback |

### Tab completion

Tab walks one level deeper into the grammar tree on each press:

```
>>> <TAB>
pyhealthcheck  /?  /read  /grammar  /run  /npu  /context  /tokens  /bye

>>> pyhealthcheck <TAB>
py_system_status  py_resource_status  py_network_status

>>> pyhealthcheck py_system_status <TAB>
py_check_kernel  py_check_uptime  py_check_services

>>> pyhealthcheck py_system_status py_check<TAB>
py_check_kernel  py_check_uptime  py_check_services
```

When the playbook has no match at that depth, Tab falls back to a live Ollama query so the model's trained knowledge fills in the gaps.

CLI command arguments also complete:
```
>>> /run <TAB>          → grammar names
>>> /tokens <TAB>       → grammar names
```

### Auto-detect modes

Three detection depths, checked in order:

**1. Grammar name** — type the grammar root name to run the full procedure:
```
>>> pyhealthcheck
Auto-detected 'pyhealthcheck' procedure — executing grammar...
[exec/py] py_check_kernel
--- kernel / runtime ---
Kernel : 6.x.x-amd64
Python : 3.x.x
...
```

**2. Multi-word path** — type grammar name + any sub-rule or token (what Tab builds up); the last word is the target:
```
>>> pyhealthcheck py_system_status
Auto-detected path 'pyhealthcheck → py_system_status' — executing sub-procedure...

>>> pyhealthcheck py_system_status py_check_uptime
Auto-detected path 'pyhealthcheck → py_system_status → py_check_uptime' — running command token...
```

**3. Bare rule or token name** — type a single rule name or command token from any loaded grammar:
```
>>> py_system_status
Auto-detected 'py_system_status' rule in 'pyhealthcheck' — executing sub-procedure...

>>> py_check_kernel
Auto-detected command token 'py_check_kernel' in 'pyhealthcheck' — running command...
```

**4. Expression parse** — type any expression and the parser tries it against all loaded grammars:
```
>>> 3 + 4 * 2
Auto-detected 'calculator' expression — running grammar...
Result: 11
```

Falls through to normal chat if none of the four modes match.

### /context

Shows all runtime state — what's loaded, in what mode, how many tokens:

```
>>> /context

=== Context ===

Startup files (INIT_KNOWLEDGE_FILES):
  training/train_python_healthcheck_commands.json
  grammars/playbook_pyhealthcheck.txt

Command vocabularies:
  pyhealthcheck  [exec=python]  8 token(s)
    py_check_kernel  py_check_uptime  py_check_cpu  py_check_memory
    py_check_disk  py_check_processes  py_check_services  py_check_ports

Playbook grammars:
  pyhealthcheck  12 rule(s)

Knowledge memory: 2 prose document(s)
```

### /tokens

Shows grammar rules and the full source of each command token:

```
>>> /tokens pyhealthcheck

=== Grammar: pyhealthcheck  [exec=python] ===

Rules:
  <pyhealthcheck      >  ::=  py_system_status py_resource_status py_network_status
  <py_system_status   >  ::=  py_check_kernel py_check_uptime py_check_services
  ...

Tokens:
  py_check_kernel:
    import platform, sys
    print('--- kernel / runtime ---')
    print('Kernel :', platform.release())
    ...
```

Without an argument, `/tokens` dumps all loaded grammars.

## Command execution modes

Grammar token values can be shell commands or pure Python source. The mode is set by `"_exec"` in the command vocabulary JSON.

### Shell mode (default)

```json
{
  "_type": "command_vocabulary",
  "_grammar": "mycheck",
  "step_one": "echo hello && date",
  "step_two": "df -h"
}
```

Tokens are executed via `subprocess.run(cmd, shell=True)`.

### Python mode

```json
{
  "_type": "command_vocabulary",
  "_grammar": "mycheck",
  "_exec": "python",
  "step_one": "import platform\nprint(platform.release())",
  "step_two": "import shutil\nu = shutil.disk_usage('/')\nprint(f'{u.free // 1073741824} GB free')"
}
```

Token values are pure Python source. Use `\n` in JSON for newlines. Full stdlib available, including `import`, `for`, `try/except`, and `subprocess`. Output goes directly to the terminal via `exec()` — no buffering.

**Current default** (`INIT_KNOWLEDGE_FILES`):

| File | Role |
|---|---|
| `training/train_python_healthcheck_commands.json` | 8 Python tokens (`_exec: python`) |
| `grammars/playbook_pyhealthcheck.txt` | BNF tree: pyhealthcheck → system / resource / network |

Trigger: type `pyhealthcheck` at the prompt.

## Grammar Tools

`model_tools_grammar.py` generates the vocabulary JSON and BNF grammar file
from external source documents so you do not have to write them by hand.

**Full documentation:** [docs/GRAMMAR_TOOLS.md](docs/GRAMMAR_TOOLS.md)

### Supported source formats

| Format | Example | Converter |
|---|---|---|
| Mermaid flowchart `.mmd` | `kali_discovery.mmd` | Structure from edges; commands via `%% cmd:` annotations |
| Markdown `.md` | `kali_discovery.md` | H1 = name, H2/H3 = rules, bullets = tokens + commands |
| Plain text spec `.txt` | `kali_discovery_spec.txt` | Numbered sections; `Step N: name — command` pattern |
| PDF `.pdf` | any procedure PDF | Heuristic heading/bullet extraction (`pip install pypdf`) |
| Web page `http(s)://` | any URL | HTML `<h1>`–`<h3>` and `<li>` structure |
| AI-assisted | any of the above | Ollama model extracts grammar semantically (`--ai-model`) |

### Workflow

```bash
# 1. Convert a source document (format auto-detected)
python model_tools_grammar.py examples/grammar_sources/kali_discovery.md --summary

# 2. Check the generated files
cat training/train_kali_discovery_commands.json
cat grammars/playbook_kali_discovery.txt

# 3. Load into the model at startup
python model_create_hf_cl.py \
    --train   training/train_kali_discovery_commands.json \
    --grammar grammars/playbook_kali_discovery.txt

# 4. Use at the CLI prompt
>>> kali_discovery
>>> kali_discovery <TAB>     → local_discovery  network_discovery
```

### AI-assisted extraction

For dense technical documents where structure is implicit, pass any Ollama
model with `--ai-model`. The model interprets the document intent rather than
relying on formatting patterns:

```bash
python model_tools_grammar.py  security_assessment.pdf  --ai-model qwen2:7b
python model_tools_grammar.py  https://wiki.corp/runbook --ai-model llama3
python model_tools_grammar.py  procedure_manual.txt      --ai-model mistral --summary
```

### Same procedure, three formats

The `kali_discovery` example ships in Mermaid, Markdown, and plain text — all
three produce identical output (10 rules, 21 tokens):

```bash
python model_tools_grammar.py examples/grammar_sources/kali_discovery.mmd   --dry-run
python model_tools_grammar.py examples/grammar_sources/kali_discovery.md    --dry-run
python model_tools_grammar.py examples/grammar_sources/kali_discovery_spec.txt --dry-run
```

### Self-test

```bash
python examples/test_grammar_tools.py           # 5 tests, no Ollama required
python examples/test_grammar_tools.py --verbose
python examples/test_grammar_tools.py --ai-model qwen2:7b   # also test AI path
```

---

## Adding a new grammar

### 1. Expression grammar (parse mode)

Write a BNF file and load it:
```
/grammar my_grammar.txt
```
Then query: `/run my_grammar some input`.

### 2. Procedure grammar — shell tokens

```json
{
  "_type": "command_vocabulary",
  "_grammar": "mycheck",
  "step_one": "echo hello",
  "step_two": "date"
}
```

### 3. Procedure grammar — Python tokens

```json
{
  "_type": "command_vocabulary",
  "_grammar": "mycheck",
  "_exec": "python",
  "step_one": "print('hello')",
  "step_two": "import datetime\nprint(datetime.datetime.now())"
}
```

Write the BNF grammar (`grammars/playbook_mycheck.txt`):
```
<mycheck>  ::= <step_one> <step_two>
<step_one> ::= "step_one"
<step_two> ::= "step_two"
```

> **Naming rule:** the last `_`-separated segment of the filename must match `"_grammar"` in the JSON.
> `playbook_mycheck.txt` → grammar name `mycheck` ✓

Load both at launch (command vocabulary JSON **before** grammar BNF):
```bash
python model_create_hf_cl.py --train training/mycheck_commands.json --grammar grammars/playbook_mycheck.txt
```

Or in-flight:
```
/read training/mycheck_commands.json
/grammar grammars/playbook_mycheck.txt
```

Then type `mycheck` to run it automatically, or use Tab to navigate:
```
>>> mycheck <TAB>       → step_one  step_two
>>> mycheck step_one    → runs step_one directly
```

To make them the permanent default, set both in `INIT_KNOWLEDGE_FILES` inside `model_create_hf_cl.py`.

## How GrammarRunner works

```
User types: pyhealthcheck
  └─ auto-detect mode 1: grammar name + commands dict present → execute mode
       └─ GrammarRunner.execute("pyhealthcheck")
            ├─ [model #1]  pyhealthcheck pyhealthcheck → py_system_status py_resource_status py_network_status
            ├─ [model #2]  pyhealthcheck py_system_status → py_check_kernel py_check_uptime py_check_services
            ├─ [model #3]  pyhealthcheck py_check_kernel → "py_check_kernel"
            │    └─ [exec/py] py_check_kernel  →  exec("import platform...")
            ├─ [model #4]  pyhealthcheck py_check_uptime → "py_check_uptime"
            │    └─ [exec/py] py_check_uptime  →  exec("import os; ...")
            └─ ... (one model query per unique rule, cached)

User types: pyhealthcheck py_system_status py_check_uptime
  └─ auto-detect mode 2: first word = grammar name, last word = target token
       └─ GrammarRunner._run_os_command("py_check_uptime", commands["py_check_uptime"])
            └─ exec("import os; ...")
```

The playbook is always authoritative: even if the tiny model mis-answers a rule query, the stored BNF body replaces the response. Left-recursive grammars (like the calculator's `expr ::= expr "+" term | term`) are handled via iterative extension.

## NPU export for STM32N6570-DK

The `/npu` command (or standalone `model_export_npu.py`) exports the trained model to ONNX format for use with STM32Cube.AI Studio targeting the STM32N6570-DK (Cortex-M55 CPU, INT8 weights).

```bash
# From the interactive session (requires locally trained model, not --model mode)
>>> /npu npu_export/

# Or standalone (requires model_create_hf_cl.py to have run at least once)
python model_export_npu.py [output_dir]
```

**Output files:**

```
npu_export/
  model_npu.onnx           FP32 ONNX (opset 17)
  model_npu_qdq.onnx       INT8 QDQ — use this in STM32Cube.AI Studio
  model_npu_int8.onnx      INT8 dynamic-quantized (alternative)
  model_info.json          arch, I/O specs, full import instructions
  tokenizer/               tokenizer files for host-side pre/post-processing
  generated_cpu/           ready-to-use network.c / network_data.c for STM32CubeIDE
  validation_data/         validation.npz — valid token ID samples for Studio
```

**Import into STM32Cube.AI Studio** — see [docs/STM32_NPU_DEPLOYMENT.md](docs/STM32_NPU_DEPLOYMENT.md) for full instructions. Quick summary:
1. Import `npu_export/model_npu_qdq.onnx`
2. **Disable** the Neural ART NPU toggle (use Cortex-M55 CPU path)
3. Set validation dataset to `npu_export/validation_data/validation.npz`
4. Click Generate Project

> **Note:** The Neural ART NPU in STEdgeAI v4.0.0 only supports CNN architectures. Transformer models (RoPE, GQA, RMSNorm) run on the Cortex-M55 CPU with INT8 weights (1.32 MiB, 4× smaller than FP32).

**Host-side inference flow:**
```
input text → tokenise (tokenizer/) → input_ids [1, seq_len] int64
→ model inference (Cortex-M55, INT8) → logits [1, seq_len, vocab_size]
→ argmax last position → token id → decode → output text
```

## License

MIT — see [LICENSE](LICENSE).
