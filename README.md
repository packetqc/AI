# Local AI Model with Grammar-Driven Knowledge

A framework for training tiny Qwen2 language models on custom knowledge, augmenting them with BNF/EBNF grammars, and serving them locally via Ollama — with an interactive CLI that can parse expressions and execute OS routines through multi-step model interactions.

## What it does

- **Builds a small Qwen2 model** from scratch using a custom BPE tokenizer trained over your own knowledge files and grammar rules.
- **Augments the model with BNF/EBNF grammars** — grammar rules become trained (prompt → answer) pairs so the model "knows" the grammar structure at inference time.
- **Runs grammar-driven interactions** via `GrammarRunner`: the model is queried once per unique grammar rule (cached), and the results drive either expression parsing or command execution.
- **Executes shell or Python code per token** — command vocabulary tokens can run shell commands (`_exec: shell`, default) or pure Python source (`_exec: python`) via `exec()`.
- **Auto-detects grammar input** — type `pyhealthcheck` and the Python healthcheck routine runs automatically; type `1+1` and the calculator grammar parses it.
- **Accepts startup arguments** — inject extra training files or grammars at launch via `--train` / `--grammar`, or pass a list file with `@`.
- **Exports to GGUF** and serves via Ollama so any Ollama-compatible client can query the model.
- **Exports to ONNX for NPU** — `/npu` or `model_export_npu.py` produces FP32 + INT8 ONNX files ready for import into STM32Cube.AI Studio (STM32N6570-DK Neural-ART NPU).

## Architecture

```
model_create_hf_cl.py           # Entry point: trains, exports, serves, interactive CLI
model_export_npu.py             # Standalone NPU/ONNX export script

classes/
  class_model_assets.py         # ModelAssets: knowledge accumulation + incremental rebuild
  class_model_grammar.py        # ModelGrammar: BNF/EBNF parser  |  GrammarRunner: execution engine
  class_terminal_logs.py        # Colour terminal logger

grammars/
  playbook_python_healthcheck.txt  # Python healthcheck procedure grammar  ← default
  playbook_linux_healthcheck.txt   # Shell healthcheck procedure grammar
  playbook_model_calculator.txt    # Expression grammar: expr ::= expr "+" term | ...

training/
  train_python_healthcheck_commands.json  # Python token vocabulary (_exec: python)  ← default
  train_linux_healthcheck_commands.json   # Shell token vocabulary (_exec: shell)
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
| `/bye` | Exit |

### Auto-detect modes

**Procedure execution** — type a grammar name that has a command vocabulary:
```
>>> pyhealthcheck
Auto-detected 'pyhealthcheck' procedure — executing grammar...
[exec/py] py_check_kernel
--- kernel / runtime ---
Kernel : 6.x.x-amd64
Python : 3.x.x
...
```

**Expression parsing** — type any arithmetic expression directly:
```
>>> 3 + 4 * 2
Auto-detected 'calculator' expression — running grammar...
Result: 11
```

Both modes use `GrammarRunner`: the model is queried once per unique grammar rule (one Ollama roundtrip per rule, cached for the lifetime of the interaction).

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

Token values are pure Python source. Use `\n` in JSON for newlines. Full stdlib available, including `import`, `for`, `try/except`, and `subprocess`. Stdout is captured and printed exactly like shell output.

**Current default** (`INIT_KNOWLEDGE_FILES`):

| File | Role |
|---|---|
| `training/train_python_healthcheck_commands.json` | 8 Python tokens (`_exec: python`) |
| `grammars/playbook_python_healthcheck.txt` | BNF tree: pyhealthcheck → system / resource / network |

Trigger: type `pyhealthcheck` at the prompt.

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

Load both at launch (command vocabulary JSON **before** grammar BNF):
```bash
python model_create_hf_cl.py --train training/mycheck_commands.json --grammar grammars/playbook_mycheck.txt
```

Or in-flight:
```
/read training/mycheck_commands.json
/grammar grammars/playbook_mycheck.txt
```

Then type `mycheck` to run it automatically.

To make them the permanent default, set both in `INIT_KNOWLEDGE_FILES` inside `model_create_hf_cl.py`.

## How GrammarRunner works

```
User types: pyhealthcheck
  └─ auto-detect: grammar name + commands dict present → execute mode
       └─ GrammarRunner.execute("pyhealthcheck")
            ├─ [model #1]  pyhealthcheck pyhealthcheck → py_system_status py_resource_status py_network_status
            ├─ [model #2]  pyhealthcheck py_system_status → py_check_kernel py_check_uptime py_check_services
            ├─ [model #3]  pyhealthcheck py_check_kernel → "py_check_kernel"
            │    └─ [exec/py] py_check_kernel  →  exec("import platform...")
            ├─ [model #4]  pyhealthcheck py_check_uptime → "py_check_uptime"
            │    └─ [exec/py] py_check_uptime  →  exec("import os; ...")
            └─ ... (one model query per unique rule, cached)
```

Left-recursive grammars (like the calculator's `expr ::= expr "+" term | term`) are handled automatically via iterative extension (operator-precedence climbing generalised to any BNF).

## NPU export for STM32N6570-DK

The `/npu` command (or standalone `model_export_npu.py`) exports the trained model to ONNX format for use with STM32Cube.AI Studio targeting the STM32N6570-DK's Neural-ART NPU accelerator.

```bash
# From the interactive session
>>> /npu npu_export/

# Or standalone (requires model_create_hf_cl.py to have run at least once)
python model_export_npu.py [output_dir]
```

**Output files:**

```
npu_export/
  model_npu.onnx           FP32 ONNX (opset 17)
  model_npu_int8.onnx      INT8 dynamic-quantized  ← recommended for NPU
  model_info.json          arch, I/O specs, import notes
  tokenizer/               tokenizer files for host-side pre/post-processing
```

**Import into STM32Cube.AI Studio:**
1. `File > Import Model` → select `npu_export/model_npu_int8.onnx`
2. Run the NPU profiler to see which layers land on the Neural-ART accelerator
3. Layers using RoPE or GQA (Qwen2-specific ops) may fall back to Cortex-M55 software

**Host-side inference flow:**
```
input text → tokenise (tokenizer/) → input_ids tensor
→ model inference (NPU) → logits[0, -1, :]
→ argmax → token id → decode → output text
```

## License

MIT — see [LICENSE](LICENSE).
