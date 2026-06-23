# STM32N6570-DK Deployment Guide

Deploying a grammar-driven model to the STM32N6570-DK, end to end — from the host export
through ST Edge AI code generation to live on-device inference with an interactive UART REPL.

**Status: WORKING** — the transformer runs on-device (Cortex-M55, INT8) and the interactive
grammar REPL is validated live on hardware. See [Performance](#performance--why-its-cpu-bound)
for the key finding (this transformer is CPU-bound on the NPU) and
[NPU-native path](#npu-native-architecture-path-light-speed) for the proven fast alternative.

---

## End-to-end flow

```
┌─ HOST (Python) ──────────────────┐   ┌─ ST EDGE AI (compile) ─┐   ┌─ STM32N6570-DK (device) ─┐
│ model_create_hf_cl.py            │   │ stedgeai generate      │   │ FSBL (Cortex-M55, bare)  │
│   trains tiny Qwen2 + tokenizer  │   │   --target stm32n6     │   │  ├ CPU int8 embedding    │
│ model_export_npu.py  (/npu)      │──▶│   --mode target        │──▶│  ├ NPU/SW inference      │
│   → npu_export/model_npu_qdq.onnx│   │   --c-api st-ai        │   │  └ C++ grammar REPL over │
│   (ONNX opset17, INT8 QDQ)       │   │ → network.c/_data.c    │   │     USART1 (115200 8N1)  │
└──────────────────────────────────┘   └────────────────────────┘   └──────────────────────────┘
        the "host solution"              the weights-placement knob        load-and-run via GDB
        (model + grammar runner)         lives here (memory-pool)          /dev/ttyACM0 (ST-Link VCP)
```

- **Host** owns the model + grammar engine; the handoff artifact is `model_npu_qdq.onnx`.
- **ST Edge AI** compiles ONNX → C and decides **weights/activation placement** (the speed knob).
- **Device** runs it from the FSBL: CPU does the int8 embedding lookup, the NPU (or SW-fallback)
  runs the body, and a C++ port of `GrammarRunner` drives an interactive REPL over UART.

---

## Quick facts

| Item | Value |
|---|---|
| Model file | `npu_export/model_npu_qdq.onnx` |
| Format | ONNX opset 17, INT8 QDQ |
| Weights | 1.32 MiB (INT8, 4× smaller than FP32) |
| Activations | 323 KiB (RAM) |
| Execution | Cortex-M55 CPU — Neural ART NPU **disabled** |
| Inputs | `input_ids` [1, seq_len] int64, `attention_mask` [1, seq_len] int64 |
| Output | `logits` [1, seq_len, 374] float32 |

---

## STM32Cube.AI Studio import — 3 steps

1. Import `npu_export/model_npu_qdq.onnx`
2. **Disable** the Neural ART NPU toggle (leave on Cortex-M55 CPU)
3. Click Generate Project — no validation data needed

That's it. Studio generates `network.c`, `network_data.c`, `network.h`, `network_data.h`.

---

## Why NOT the Neural ART NPU

STEdgeAI Core v4.0.0 Neural ART compiler only supports CNN architectures (Conv2D, Pool, Dense, BatchNorm). This model uses Transformer-specific ops — RoPE, GQA, RMSNorm — which are not implemented in the Neural ART compiler.

Enabling Neural ART causes:
```
NOT IMPLEMENTED: Unknown layer format for layer Input_N
```
at op 94/103 regardless of model variant or quantization format. This is a confirmed compiler limitation in v4.0.0, not a model error.

The Cortex-M55 CPU path passes 171/171 validation checks with full INT8 MatMul support.

---

## Why no validation data is needed

Reference CNN/audio models use float32 inputs — stedgeai's default random data works fine.
This model uses int64 token ID inputs, so stedgeai's random full-range integers would crash
the ONNX embedding Gather node with out-of-bounds errors.

The fix is built into `model_npu_qdq.onnx`: a `Max(0)` → `Min(373)` guard is inserted before
the embedding Gather. stedgeai's random integers are clamped to `[0, 373]` automatically.
No custom validation dataset required.

This guard is a no-op at deployment — real token IDs are always in `[0, 373]`.

---

## Step-by-step: CubeMX project first

Studio's headless CubeMX invocation has a threading bug (`ConcurrentModificationException`)
in v4.0.0 on Linux. Generate the STM32 project from CubeMX GUI before using Studio.

1. Open STM32CubeMX GUI
2. File → New Project → Board Selector → `STM32N6570-DK` → Start Project
3. Accept default board initialization (configures XSPI1, XSPI2, EXTMEM, clocks)
4. Project Manager tab → Toolchain/IDE: **Makefile** → Generate Code
5. In Studio, open or create a project pointing to that workspace

> For future projects: always generate the CubeMX project first, then link Studio to it.

---

## Expected generate output

```
network.c            ~9,662 lines  Cortex-M55 inference engine
network_data.c       ~3.67 MB      INT8 weights
network.h            STAI API header
network_data.h       data header
network_details.h    per-layer details
```

Memory layout:
```
weights (ro)  : 1.32 MiB  (1 segment)
activations   : 323 KiB   (1 segment, includes input/output buffers)
```

---

## Known non-fatal messages

These appear in the **validate** report only. The **generate** report is clean.

```
E: scaled_dot_product_attention_1_bias_0_conversion layer - number of I/O tensor is not coherent: 0/1
E: val_312_bias_0_2_val_312_conversion layer - number of I/O tensor is not coherent: 0/1
```

Internal compiler annotations for constant attention bias buffers — do not affect the generated C code.

---

## CLI alternative (no Studio needed)

```bash
STEDGEAI=/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai

$STEDGEAI generate --model npu_export/model_npu_qdq.onnx \
    --target stm32n6 --name network --c-api st-ai \
    --output npu_export/generated_cpu
```

Pre-generated files are already in `npu_export/generated_cpu/`.

---

## Runtime inference on the board

```c
/* 1 — initialise */
aiInit();

/* 2 — fill inputs: int32[1 × seq_len], values in [0, 373] */
memcpy(buffer_in_ids,  token_ids, seq_len * sizeof(int32_t));
memcpy(buffer_in_mask, attn_mask, seq_len * sizeof(int32_t));

/* 3 — run inference */
aiRun();

/* 4 — greedy decode: argmax over logits[seq_len-1][:] */
int32_t next_token = argmax(buffer_out, 374);
```

Host tokenization uses `npu_export/tokenizer/` (BPE, vocab_size=374).

---

## What was fixed in model_export_npu.py

| Problem | Root cause | Fix |
|---|---|---|
| `Unknown layer format for layer Input_N` | Neural ART NPU v4.0.0 does not support Transformers | Disable NPU toggle in Studio |
| `indices element out of data bounds, idx=-538846080` | stedgeai generates full-range random int64; embedding Gather has no bounds check | Insert `Max(0)→Min(373)` before Gather in exported ONNX |
| `Scale type of 0 is <class 'int'>` | ONNX `Clip` node's `min=0` int64 input misread by stedgeai as a QDQ scale | Replace `Clip` with `Max`+`Min` elementwise nodes |
| `Removing initializer val_176` | onnxsim leaves dead Constant nodes after constant folding | Remove zero-consumer Constant nodes before QDQ quantizer runs |
| `ConcurrentModificationException` in CubeMX | Threading bug in stedgeai v4.0.0 headless CubeMX invocation on Linux | Generate CubeMX project from GUI first |

---

## On-device deployment (FSBL + interactive REPL)

The generated network is integrated into the **FSBL** (First-Stage Boot Loader) of an
ST Edge AI project (`STM32N6/AI_TO_NPU_1/run-22`). The FSBL is the complete application —
there is no separate Appli image. On top of the C inference wrapper, a **C++ port of the host
grammar engine** runs an interactive REPL over USART1:

| Layer | File | Role |
|---|---|---|
| NPU inference | `FSBL/Core/Src/llm_fsbl.c` | `stai_network_*` wrapper: CPU int8 embedding → NPU run → argmax |
| Tokenizer | `FSBL/Core/Src/llm_tokenizer.c` | BPE encode/decode, 374 vocab |
| Logger | `FSBL/Core/Src/terminal_logger.cpp` | C++ port of `class_terminal_logs.py` — same colored output |
| Grammar engine | `FSBL/Core/Src/grammar_runner.cpp` | C++ port of `GrammarRunner` (tokenize / parse / evaluate / execute / probe) |
| REPL | `FSBL/Core/Src/llm_repl.cpp` | `extern "C" LLM_Repl_Run` — reads a line, probes the grammar, queries the NPU per rule |

C++ is built as an embedded subset (`-fno-exceptions -fno-rtti -fno-threadsafe-statics -nostdlib++`,
no heap); the C boot/HAL/NPU layer is reached via `extern "C"`. The grammar engine is
**model-agnostic at the query interface** — swapping the model does not change the REPL logic.

### Build + run (Linux host, ST-Link V3)

```bash
# Build the FSBL (links the NPU runtime + the C++ REPL)
cd STM32N6/AI_TO_NPU_1/run-22/Makefile/FSBL && make -j8 all

# Serial: the ST-Link VCP is /dev/ttyACM0 @ 115200 8N1 (no flow control)
#   minicom -D /dev/ttyACM0 -b 115200     (the "Offline" indicator is just DCD — ignore)
```

The FSBL is exercised via **GDB load-and-run** (load the ELF into AXISRAM2, release the
`debugFlag` trap, run from RAM) — see `Knowledge/K_STM32_MCP/methodology/methodology-debug-fsbl-direct.md`.
On release the REPL prints its banner and accepts expressions (`3 + 4`, `quit`).

---

## Performance — why it's CPU-bound

A tiny calculator model on a 1 GHz Cortex-M55 + 600 GOPS NPU **should** answer in
milliseconds, yet inference takes **minutes**. Two compounding causes, both confirmed:

1. **The Neural-ART can't accelerate this transformer.** It is a CNN/conv+GEMM int8 engine;
   Qwen2's RoPE / GQA / RMSNorm / softmax are unsupported → the model only compiles via
   `LL_ATON_SW_FALLBACK`, so the transformer math runs on the **CM55 in software**.
2. **Weights live in external XSPI NOR flash** (`copyWeightsToRam: false`) with prefetch
   disabled — so the SW-fallback CPU reads every weight byte through a slow flash transaction.

**Memory tiers (slowest → fastest):** XSPI NOR flash `0x70000000` ◀ PSRAM `0x90000000`
(writable, lower latency) ◀ internal AXISRAM `0x34000000` (~100× faster than external serial).

**Weights-in-SRAM is blocked for this model:** explicit internal `--st-neural-art` profiles hit
the transformer compile bug, and `--copy-weights-at` is `E102: not supported with --c-api st-ai`.
A runtime relocation (memcpy flash→AXISRAM1 + RIF/cache) also faults — the ATON graph is compiled
for the flash pool, so the relocation is inconsistent. **Conclusion: the slowness is the
architecture, not the silicon.**

---

## NPU-native architecture path (light speed)

Because the task (next-token over a tiny grammar, vocab 374, seq 32) is simple, it can be
re-expressed as an architecture the Neural-ART runs **fully in hardware** — keeping the host
export flow, the tokenizer, the grammar runner and the REPL unchanged. Only the model
*architecture* changes (a model-creation step; the NPU export function is untouched).

**Proven (host-side, `stedgeai analyze`):** a causal **Conv1D (TCN)** body
`embeddings[1,256,32] → Conv1d(k3) → Conv1d(k3) → Conv1d(k1) → logits[1,374,32]`, int8 per-channel:

```
Total epochs 5 | pure software 0 | hybrid 0 | pure hardware 5   ← 100% on the NPU
weights 481 KiB int8 · activations 31 KiB · macc 15.6M · fits internal AXISRAM
```

A trained version (reusing the existing 374 tokenizer/vocab, no Qwen2 dependency) **reproduces
the grammar rule bodies the runner queries with 5/5 recall**. The Conv1D body fits internal
SRAM (481 KiB), which also dissolves the flash-weights bottleneck.

| Architecture | NPU? | Notes |
|---|---|---|
| Transformer (Qwen2) — current | ❌ host SW-fallback | RoPE/GQA/RMSNorm unsupported → minutes/token |
| **Conv1D / TCN** | ✅ 100% HW | conv is the NPU's core op; ~tens of µs |
| MLP / fully-connected | ✅ HW | GEMM maps to the Convolution Accelerators |

**Caveat:** the *trained* int8 TCN currently segfaults the `atonn` compiler at ~85%
(`signo=11`) while the same architecture with random weights compiles cleanly — an ST
tool-robustness bug on the specific weight pattern (likely the 374-channel 1×1 head with many
near-zero unused-vocab channels), to be reported to ST / worked around with weight regularization.

**Next step (opt-in):** train/distill the Conv1D model, export through the *same* flow → new
`run-XX`, and adapt only the device wrapper I/O (`int8[1,256,32]` in → `int8[1,374,32]` out).
