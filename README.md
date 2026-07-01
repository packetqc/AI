# Grammar-Driven Tiny LLMs — Model-Carried Logic

Train tiny language models on your own knowledge, teach them BNF/EBNF grammars, and **let the model carry the logic**: the working code is transposed into trainable tokens, emitted by the model at inference, and executed by a **grammar-agnostic runner** — so a new or changed grammar needs **no per-grammar CPU code**.

The same grammar conceptualization also runs on device — a transformer on the STM32N6 Cortex‑M55 CPU and a Conv1D/TCN natively on the Neural‑ART NPU — plus a blackbox toolkit to security‑audit any loadable model. All of that is in the **[User Guide](docs/USER_GUIDE.md)**.

## Quick start — `nocode_runner.py`

```bash
source venv/bin/activate

# the model supplies the logic; nocode_runner executes it
python3 scripts/nocode_runner.py --mode host \
    --grammar models/grammars/playbook_model_calculator.txt \
    --model model_calculator_nocode_v1 --policy generative
```

```
nocode> 3 + 4         # expression  → evaluate-mode → Result: 7   (op bodies emitted by the model)
nocode> fibonacci     # command name → execute-mode → runs the model-emitted procedure
```

| Command | What it does |
|---|---|
| `nocode_runner.py [--model M] [--grammar F…] [--policy P]` | run a model-carried grammar; omit `--model` → script-named default, omit `--grammar` → auto-load from the model's `state.json` |
| `nocode_runner.py --grammar a.txt b.txt` | load **multiple** grammars (one grammar can call others) |
| in-session: `<expr>` / `<name>` · `/policy` · `/grammar` · `/set` · `/create` · `/?` · `/bye` · `TAB` | evaluate / execute · set policy · switch grammars · config · retrain · help · quit · completion |

**Exec policy** (`--policy` / `/policy`): `token_select` (vocab only) → `vocab_verified` (model emits, verified fallback — default) → `generative` (run the model-emitted body).

> First time here? Install & setup (venv, PyTorch, Ollama) and building a model are in the [User Guide](docs/USER_GUIDE.md#install--info).

## Documentation

Everything else lives in the **[User Guide](docs/USER_GUIDE.md)** — install & setup, the unified runner (`scripts/model_runner.py`), the full nocode track, host→CPU and host→NPU (STM32N6 Neural-ART) deployment, grammar tools, and the blackbox Model Security RE.

Deep-dive references: [Nocode Runner](docs/NOCODE_RUNNER.md) · [STM32 NPU Deployment](docs/STM32_NPU_DEPLOYMENT.md) · [Grammar Tools](docs/GRAMMAR_TOOLS.md) · [Model Security RE](docs/MODEL_SECURITY_RE.md)

## License

MIT — see [LICENSE](LICENSE).
