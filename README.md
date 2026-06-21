# Local AI Model with Grammar-Driven Knowledge

A framework for training tiny Qwen2 language models on custom knowledge, augmenting them with BNF/EBNF grammars, and serving them locally via Ollama — with an interactive CLI that can parse expressions and execute OS routines through multi-step model interactions.

## What it does

- **Builds a small Qwen2 model** from scratch using a custom BPE tokenizer trained over your own knowledge files and grammar rules.
- **Augments the model with BNF/EBNF grammars** — grammar rules become trained (prompt → answer) pairs so the model "knows" the grammar structure at inference time.
- **Runs grammar-driven interactions** via `GrammarRunner`: the model is queried once per unique grammar rule (cached), and the results drive either expression parsing or OS command execution.
- **Auto-detects grammar input** — type `1+1` and the calculator grammar runs automatically; type `healthcheck` and the OS routine executes.
- **Exports to GGUF** and serves via Ollama so any Ollama-compatible client can query the model.

## Architecture

```
model_create_hf_cl.py       # Entry point: trains, exports, serves, interactive CLI
class_model_assets.py       # ModelAssets: knowledge accumulation + incremental rebuild
class_model_grammar.py      # ModelGrammar: BNF/EBNF parser  |  GrammarRunner: execution engine
class_terminal_logs.py      # Colour terminal logger
```

**Grammar files** (loaded at startup or via `/grammar`):

| File | Purpose |
|---|---|
| `playbook_model_calculator.txt` | Expression grammar: `expr ::= expr "+" term \| ...` |
| `playbook_linux_healthcheck.txt` | Procedure grammar: healthcheck → system → OS commands |
| `train_linux_healthcheck_commands.json` | Command vocabulary: `{"check_cpu": "top -bn1 ..."}` |

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
| `/bye` | Exit |

### Auto-detect modes

**Expression parsing** — type any arithmetic expression directly:
```
>>> 3 + 4 * 2
Auto-detected 'calculator' expression — running grammar...
Result: 11
```

**Procedure execution** — type a grammar name that has a command vocabulary:
```
>>> healthcheck
Auto-detected 'healthcheck' procedure — executing grammar...
[exec] check_uptime  $ uptime && echo '---' && cat /proc/loadavg
...
```

Both modes use `GrammarRunner`: the model is queried once per unique grammar rule (one Ollama roundtrip per rule, cached for the lifetime of the interaction).

## Adding a new grammar

### 1. Expression grammar (parse mode)

Write a BNF file and load it:
```
/grammar my_grammar.txt
```
Then query: `/run my_grammar some input`.

### 2. Procedure grammar (execute mode)

Write a command vocabulary JSON:
```json
{
  "_type": "command_vocabulary",
  "_grammar": "mycheck",
  "step_one": "echo hello",
  "step_two": "date"
}
```

Write a BNF procedure grammar (`playbook_mycheck.txt`):
```
<mycheck>  ::= <step_one> <step_two>
<step_one> ::= "step_one"
<step_two> ::= "step_two"
```

Load both (order matters — commands JSON before grammar BNF):
```
/read mycheck_commands.json
/grammar playbook_mycheck.txt
```

Then type `mycheck` to run it automatically.

To include them at startup add both to `INIT_KNOWLEDGE_FILES` in `model_create_hf_cl.py`.

## How GrammarRunner works

```
User types: healthcheck
  └─ auto-detect: grammar name + commands dict present → execute mode
       └─ GrammarRunner.execute("healthcheck")
            ├─ [model #1]  healthcheck healthcheck → system_status resource_status network_status
            ├─ [model #2]  healthcheck system_status → check_kernel check_uptime check_services
            ├─ [model #3]  healthcheck check_kernel → "check_kernel"
            │    └─ [exec] check_kernel  $ uname -r
            ├─ [model #4]  healthcheck check_uptime → "check_uptime"
            │    └─ [exec] check_uptime  $ uptime ...
            └─ ... (one model query per unique rule, cached)
```

Left-recursive grammars (like the calculator's `expr ::= expr "+" term | term`) are handled automatically via iterative extension (operator-precedence climbing generalised to any BNF).

## License

MIT — see [LICENSE](LICENSE).
# Local AI Model with Grammar-Driven Knowledge

A framework for training tiny Qwen2 language models on custom knowledge, augmenting them with BNF/EBNF grammars, and serving them locally via Ollama — with an interactive CLI that can parse expressions and execute OS routines through multi-step model interactions.

## What it does

- **Builds a small Qwen2 model** from scratch using a custom BPE tokenizer trained over your own knowledge files and grammar rules.
- **Augments the model with BNF/EBNF grammars** — grammar rules become trained (prompt → answer) pairs so the model "knows" the grammar structure at inference time.
- **Runs grammar-driven interactions** via `GrammarRunner`: the model is queried once per unique grammar rule (cached), and the results drive either expression parsing or OS command execution.
- **Auto-detects grammar input** — type `1+1` and the calculator grammar runs automatically; type `healthcheck` and the OS routine executes.
- **Exports to GGUF** and serves via Ollama so any Ollama-compatible client can query the model.

## Architecture

```
model_create_hf_cl.py       # Entry point: trains, exports, serves, interactive CLI
class_model_assets.py       # ModelAssets: knowledge accumulation + incremental rebuild
class_model_grammar.py      # ModelGrammar: BNF/EBNF parser  |  GrammarRunner: execution engine
class_terminal_logs.py      # Colour terminal logger
```

**Grammar files** (loaded at startup or via `/grammar`):

| File | Purpose |
|---|---|
| `playbook_model_calculator.txt` | Expression grammar: `expr ::= expr "+" term \| ...` |
| `playbook_linux_healthcheck.txt` | Procedure grammar: healthcheck → system → OS commands |
| `train_linux_healthcheck_commands.json` | Command vocabulary: `{"check_cpu": "top -bn1 ..."}` |

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
| `/bye` | Exit |

### Auto-detect modes

**Expression parsing** — type any arithmetic expression directly:
```
>>> 3 + 4 * 2
Auto-detected 'calculator' expression — running grammar...
Result: 11
```

**Procedure execution** — type a grammar name that has a command vocabulary:
```
>>> healthcheck
Auto-detected 'healthcheck' procedure — executing grammar...
[exec] check_uptime  $ uptime && echo '---' && cat /proc/loadavg
...
```

Both modes use `GrammarRunner`: the model is queried once per unique grammar rule (one Ollama roundtrip per rule, cached for the lifetime of the interaction).

## Adding a new grammar

### 1. Expression grammar (parse mode)

Write a BNF file and load it:
```
/grammar my_grammar.txt
```
Then query: `/run my_grammar some input`.

### 2. Procedure grammar (execute mode)

Write a command vocabulary JSON:
```json
{
  "_type": "command_vocabulary",
  "_grammar": "mycheck",
  "step_one": "echo hello",
  "step_two": "date"
}
```

Write a BNF procedure grammar (`playbook_mycheck.txt`):
```
<mycheck>  ::= <step_one> <step_two>
<step_one> ::= "step_one"
<step_two> ::= "step_two"
```

Load both (order matters — commands JSON before grammar BNF):
```
/read mycheck_commands.json
/grammar playbook_mycheck.txt
```

Then type `mycheck` to run it automatically.

To include them at startup add both to `INIT_KNOWLEDGE_FILES` in `model_create_hf_cl.py`.

## How GrammarRunner works

```
User types: healthcheck
  └─ auto-detect: grammar name + commands dict present → execute mode
       └─ GrammarRunner.execute("healthcheck")
            ├─ [model #1]  healthcheck healthcheck → system_status resource_status network_status
            ├─ [model #2]  healthcheck system_status → check_kernel check_uptime check_services
            ├─ [model #3]  healthcheck check_kernel → "check_kernel"
            │    └─ [exec] check_kernel  $ uname -r
            ├─ [model #4]  healthcheck check_uptime → "check_uptime"
            │    └─ [exec] check_uptime  $ uptime ...
            └─ ... (one model query per unique rule, cached)
```

Left-recursive grammars (like the calculator's `expr ::= expr "+" term | term`) are handled automatically via iterative extension (operator-precedence climbing generalised to any BNF).

## License

MIT — see [LICENSE](LICENSE).
