# llm_uart — on-device Qwen2 inference via NPU

C source files that replicate the interactive inference logic of `model_create_hf_cl.py`
running on the STM32N6570-DK. *(Legacy example — see the note below for the maintained `run-23`
deployment; the `run-22` project it originated from has been removed.)*

> **Note (current deployment):** this folder is the original **Appli**-based C example. The
> maintained deployment now lives in the **FSBL** of `STM32N6/AI_TO_NPU_1/run-23`, where the device
> runs the grammar calculator **autonomously** on-chip (`main.c` `calc>` loop +
> `FSBL/AI/grammar_runner.cpp`, `npu_query.c`); a host **unified runner**
> (`scripts/classes/class_model_runner.py`, `--mode host|device`) is a thin terminal in device mode.
> For the full export→device flow, the epoch-stall fix (ASYNC + `stai_runtime_init`), and the
> NPU-native (Conv1D/TCN) path that compiles 100% on the NPU, see
> [`docs/STM32_NPU_DEPLOYMENT.md`](../../docs/STM32_NPU_DEPLOYMENT.md).

## Files

| File | Purpose |
|---|---|
| `llm_tokenizer.h/.c` | BPE tokenizer (374 vocab, 117 merges) — encode text ↔ token IDs |
| `llm_npu.h/.c` | STAI inference wrapper — init, run, greedy decode, UART loop |
| `llm_test.h/.c` | Calculator expression test harness for `main.c` |

## Adding to the STM32 project (STM32CubeIDE)

### 1. Copy source files

Copy all `.c` and `.h` files from this folder into the generated project:

```
AI_TO_NPU_1/.ai/run/run-22/Appli/Core/Src/   ← add llm_npu.c, llm_tokenizer.c, llm_test.c
AI_TO_NPU_1/.ai/run/run-22/Appli/Core/Inc/   ← add llm_npu.h, llm_tokenizer.h, llm_test.h
```

Or place them in any folder already on the build path.

### 2. Include paths

The files include `app_x-cube-ai.h`, `network.h`, `bsp_ai.h` — these are already on the
AI project include path (`Appli/AI/App/`).

### 3. Wire into main.c — quick test (recommended first step)

```c
/* main.c */
#include "llm_test.h"

/* Inside the infinite loop, or just before it: */
LLM_TestCalc();
```

`LLM_TestCalc()` initialises the NPU on first call, then runs 10 calculator expressions
through the model and prints results over USART1 at 115200 baud.

### 4. Replace validation loop with interactive UART loop

To get the full Python-equivalent interactive experience, edit `app_x-cube-ai.c`:

```c
/* app_x-cube-ai.c — USER CODE blocks */

/* USER CODE BEGIN includes */
#include "llm_npu.h"
/* USER CODE END includes */

void STM32CubeAI_Studio_AI_Init(void)
{
    MX_UARTx_Init();
    aiPreInitialize();
    /* USER CODE BEGIN init */
    LLM_NPU_Init();          /* replaces aiValidationInit() */
    /* USER CODE END init */
}

void STM32CubeAI_Studio_AI_Process(void)
{
    /* USER CODE BEGIN process */
    LLM_Process();           /* blocking: read UART → infer → print */
    /* USER CODE END process */
}
```

> Note: do **not** call `aiValidationInit()` and `LLM_NPU_Init()` both — they both call
> `stai_mnetwork_init` for the same network slot. Use one or the other.

## Model parameters

| Item | Value |
|---|---|
| Input 1 | `input_ids`    int32[1×32] |
| Input 2 | `attention_mask` int32[1×32] |
| Output  | `logits`       int8[1×32×1×374] |
| Output QLinear | scale=0.069436282, zero_point=−52 |
| Vocab size | 374 tokens |
| Seq len | 32 tokens (fixed) |
| Activation RAM | 331 032 bytes |

## Inference pipeline (C equivalent of model_create_hf_cl.py)

```
User text (UART)
    │
    ▼
LLM_Tokenize()        BPE encode → int32[32] ids + mask
    │
    ▼
LLM_NPU_RunOnce()     STAI: fill inputs → run → int8[32×1×374] logits
    │
    ▼
argmax_i8()           argmax over logits[last_pos, 0:374] → token_id
    │                 (valid on raw int8 because scale > 0, monotonic)
    ▼
LLM_TokenStr()        vocab lookup → decoded string
    │
    ▼
HAL_UART_Transmit()   send back to host
    │
    ▼
(loop: append token → next inference step)
```

## Notes

- The model was trained on a calculator BNF grammar corpus (see `model_create_hf_cl.py`).
  Its predictions are grammar continuations, not arithmetic results.
- For a calculator expression like `"1+1"`, the model predicts the next most likely token
  according to its grammar training — useful to verify the NPU path is functional.
- The BPE tokenizer handles ASCII only; non-ASCII bytes are silently skipped.
- Greedy decoding (`LLM_Generate`) uses a 32-token sliding window context.
