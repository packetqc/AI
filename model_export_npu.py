#!/usr/bin/env python3
"""
model_export_npu.py — Export the trained model to ONNX for STM32Cube.AI / STM32N6570-DK NPU.

Standalone usage:
    python model_export_npu.py [output_dir]          # default: npu_export/

Interactive (from inside model_create_hf_cl.py session):
    /npu [output_dir]

Output files
------------
  npu_export/
    model_npu.onnx           FP32 ONNX (opset 17)
    model_npu_int8.onnx      INT8 dynamic-quantized ONNX  ← recommended for NPU
    model_info.json          arch, input/output specs, import notes
    tokenizer/               tokenizer files for host-side pre/post-processing

STM32Cube.AI import
-------------------
  File > Import Model > select npu_export/model_npu_int8.onnx
  Run the NPU profiler to see which layers land on the Neural-ART accelerator.

Notes
-----
  * Qwen2 uses rotary positional embeddings (RoPE) and grouped-query attention (GQA).
    These are complex ops; unsupported layers fall back to Cortex-M55 software execution.
  * Use the STM32Cube.AI profiler output to identify bottlenecks and decide whether
    to simplify the architecture for full NPU coverage.
  * The INT8 variant is preferred: smaller memory footprint and faster NPU inference.
"""
import os
import sys
import json
import shutil

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2TokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classes.class_terminal_logs import TerminalLogger
from classes.class_model_assets import ModelAssets

_DEFAULT_OUT = "npu_export"


def _discover_model():
    """Return (model_path, state_path) by scanning for *.state.json in cwd.

    Picks the most recently modified state file whose companion model directory
    contains a config.json and at least one weight file — so this works regardless
    of what the model was named.
    """
    import glob
    for sf in sorted(glob.glob("*.state.json"), key=os.path.getmtime, reverse=True):
        mp = sf[: -len(".state.json")]
        if os.path.isfile(os.path.join(mp, "config.json")) and any(
            os.path.isfile(os.path.join(mp, w))
            for w in ("model.safetensors", "pytorch_model.bin")
        ):
            return mp, sf
    return None, None


def export_for_npu(model, tokenizer, config, arch, model_path, output_dir, logger=None):
    """Export *model* to ONNX + INT8-quantized ONNX under *output_dir*.

    Called with an in-memory model (from the interactive /npu command) or with a
    freshly-loaded model (from the standalone __main__ block below).
    """
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as exc:
        msg = "Missing dependency: " + str(exc) + " — run: pip install onnx"
        if logger:
            logger.log("error", "NPU", msg)
        else:
            print("[ERROR]", msg)
        return False

    os.makedirs(output_dir, exist_ok=True)
    fp32_path = os.path.join(output_dir, "model_npu.onnx")
    int8_path = os.path.join(output_dir, "model_npu_int8.onnx")
    info_path = os.path.join(output_dir, "model_info.json")
    tok_dir   = os.path.join(output_dir, "tokenizer")

    def _log(level, msg):
        if logger:
            logger.log(level, "NPU", msg)
        else:
            print("[" + level.upper() + "] NPU:", msg)

    # ── ONNX export ────────────────────────────────────────────────────────────
    _log("info", "Exporting FP32 ONNX (opset 17)...")
    model.eval()
    # Disable KV cache so the graph has clean input_ids + attention_mask only.
    model.config.use_cache = False

    seq_len = 32
    dummy_ids  = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long)

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask),
                fp32_path,
                opset_version=17,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids":      {0: "batch", 1: "seq_len"},
                    "attention_mask": {0: "batch", 1: "seq_len"},
                    "logits":         {0: "batch", 1: "seq_len"},
                },
                do_constant_folding=True,
                dynamo=False,       # use legacy TorchScript exporter — no onnxscript needed
            )
    except Exception as exc:
        _log("error", "ONNX export failed: " + str(exc))
        return False

    # Validate the exported graph.
    try:
        onnx.checker.check_model(fp32_path)
        _log("ok", "FP32 ONNX validated: " + fp32_path)
    except Exception as exc:
        _log("warning", "ONNX validation warning: " + str(exc))

    # ── INT8 dynamic quantisation ──────────────────────────────────────────────
    _log("info", "Quantising to INT8 (dynamic)...")
    try:
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        _log("ok", "INT8 ONNX written: " + int8_path)
    except Exception as exc:
        _log("warning", "INT8 quantisation failed (FP32 file still usable): " + str(exc))

    # ── tokenizer files ────────────────────────────────────────────────────────
    os.makedirs(tok_dir, exist_ok=True)
    _tok_files = (
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "added_tokens.json",
    )
    for fname in _tok_files:
        src = os.path.join(model_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tok_dir, fname))
    _log("ok", "Tokenizer files copied to " + tok_dir)

    # ── model_info.json ────────────────────────────────────────────────────────
    vocab_size = config.vocab_size
    info = {
        "model_type":    "Qwen2ForCausalLM",
        "export_format": "ONNX opset 17",
        "target":        "STM32N6570-DK Neural-ART NPU via STM32Cube.AI",
        "arch":          arch,
        "vocab_size":    vocab_size,
        "inputs": {
            "input_ids":      {"shape": [1, "seq_len"], "dtype": "int64",
                               "description": "Token IDs — tokenise on host before sending"},
            "attention_mask": {"shape": [1, "seq_len"], "dtype": "int64",
                               "description": "1 for real tokens, 0 for padding"},
        },
        "outputs": {
            "logits": {"shape": [1, "seq_len", vocab_size], "dtype": "float32",
                       "description": "Next-token logits — argmax last position for greedy decode"},
        },
        "files": {
            "fp32_onnx":  "model_npu.onnx",
            "int8_onnx":  "model_npu_int8.onnx",
            "tokenizer":  "tokenizer/",
        },
        "stm32cube_ai_notes": [
            "Recommended import: model_npu_int8.onnx (INT8 dynamic quant, smaller + faster on NPU).",
            "Qwen2 uses RoPE (rotary positional embeddings) and GQA (grouped-query attention).",
            "Unsupported ops fall back to Cortex-M55 software — use the STM32Cube.AI profiler",
            "  to identify which layers land on the Neural-ART NPU and which stay on CPU.",
            "Host-side flow: tokenise input → run inference → argmax over logits[:, -1, :].",
        ],
    }
    with open(info_path, "w", encoding="utf-8") as _f:
        json.dump(info, _f, indent=2)
    _log("ok", "model_info.json written: " + info_path)

    _log("ok", "NPU export complete → " + output_dir)
    _log("info", "STM32Cube.AI: File > Import Model > " + int8_path)
    return True


# ── standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    _out = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_OUT
    _logger = TerminalLogger()

    _MODEL_PATH, _STATE_PATH = _discover_model()
    if not _MODEL_PATH:
        _logger.log("error", "NPU",
                    "No model found in current directory. "
                    "Run model_create_hf_cl.py first to train and save the model.")
        sys.exit(1)

    _logger.log("info", "NPU", "Found model: " + _MODEL_PATH + "/  (state: " + _STATE_PATH + ")")
    _logger.log("info", "NPU", "Loading saved model from " + _MODEL_PATH + "/ ...")
    _config    = Qwen2Config.from_pretrained(_MODEL_PATH)
    _model     = Qwen2ForCausalLM.from_pretrained(_MODEL_PATH)
    _tokenizer = Qwen2TokenizerFast.from_pretrained(_MODEL_PATH)

    _state = ModelAssets.load_state(_STATE_PATH)
    _arch  = _state.get("arch", {}) if _state else {}

    export_for_npu(
        model=_model,
        tokenizer=_tokenizer,
        config=_config,
        arch=_arch,
        model_path=_MODEL_PATH,
        output_dir=_out,
        logger=_logger,
    )
