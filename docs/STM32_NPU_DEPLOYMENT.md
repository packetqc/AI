# STM32N6570-DK Deployment Guide

Deploying the trained Qwen2 model to the STM32N6570-DK using STM32Cube.AI Studio v4.0.0.

**Status: WORKING** — Studio AI successfully completes code generation with `model_npu_qdq.onnx`.

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
