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
import logging
import warnings

import numpy as np
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
    model.eval()
    model.config.use_cache = False  # clean graph: only input_ids + attention_mask

    seq_len    = 32
    dummy_ids  = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long)

    # Wrapper that returns logits only — avoids KV-cache and auxiliary outputs
    # that confuse the TorchScript tracer.  .eval() prevents training-mode warning.
    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, mdl):
            super().__init__()
            self.mdl = mdl
        def forward(self, input_ids, attention_mask):
            return self.mdl(input_ids=input_ids, attention_mask=attention_mask).logits

    _export_model = _LogitsWrapper(model).eval()

    # dynamic_axes used by the TorchScript path
    _dynamic_axes = {
        "input_ids":      {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "logits":         {0: "batch", 1: "seq_len"},
    }

    # dynamic_shapes for the dynamo path:
    #   batch is kept DYNAMIC so ST Edge AI Core can clearly identify dim-0 as BATCH.
    #   seq_len is left out (baked in as dummy seq_len) so Range/Shape/arange constant-fold.
    try:
        from torch.export import Dim as _Dim
        _dynamic_shapes = {
            "input_ids":      {0: _Dim("batch",      min=1, max=8)},
            "attention_mask": {0: _Dim("batch_mask", min=1, max=8)},
        }
    except Exception:
        _dynamic_shapes = None  # fall back to dynamic_axes for dynamo too

    # Stderr line filter — drops glog/absl-formatted lines that Python's
    # warnings module cannot intercept (e.g. torchvision registry notices).
    class _StderrFilter:
        _DROP = ("torchvision is not installed",)
        def __init__(self, stream):
            self._s, self._buf = stream, ""
        def write(self, text):
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if not any(d in line for d in self._DROP):
                    self._s.write(line + "\n")
        def flush(self):
            if self._buf:
                if not any(d in self._buf for d in self._DROP):
                    self._s.write(self._buf)
                self._buf = ""
            self._s.flush()
        def __getattr__(self, name):
            return getattr(self._s, name)

    # Strategy:
    #   1. dynamo static  — no dynamic_shapes; PyTorch constant-folds Range/Shape/arange
    #                       at export time → simplest graph, best ST Edge AI Core compat.
    #   2. dynamo dynamic — dynamic_shapes kept; more flexible but more complex ops.
    #   3/4. TorchScript  — legacy fallback at opset 17/14.
    _has_onnxscript = True
    try:
        import onnxscript  # noqa: F401
    except ImportError:
        _has_onnxscript = False

    _exported = False
    _attempts = []
    if _has_onnxscript:
        _attempts.append(("dynamo opset 18", {"dynamo": True}))
    _attempts += [
        ("TorchScript opset 17", {"dynamo": False, "opset_version": 17}),
        ("TorchScript opset 14", {"dynamo": False, "opset_version": 14}),
    ]

    _warn_filters = [
        ("ignore", ".*training mode.*",               UserWarning),
        ("ignore", ".*dynamic_axes.*dynamo.*",        UserWarning),
        ("ignore", ".*axis name.*will not be used.*", UserWarning),
        ("ignore", ".*LeafSpec.*",                    FutureWarning),
    ]

    for _label, _kwargs in _attempts:
        _log("info", "Exporting FP32 ONNX (" + _label + ")...")
        try:
            _real_stderr = sys.stderr
            sys.stderr = _StderrFilter(_real_stderr)
            try:
                with warnings.catch_warnings():
                    for _action, _msg, _cat in _warn_filters:
                        warnings.filterwarnings(_action, message=_msg, category=_cat)
                    with torch.no_grad():
                        if _kwargs.get("dynamo"):
                            _kw = dict(input_names=["input_ids", "attention_mask"],
                                       output_names=["logits"],
                                       do_constant_folding=True,
                                       dynamo=True)
                            if _dynamic_shapes:
                                _kw["dynamic_shapes"] = _dynamic_shapes
                            else:
                                _kw["dynamic_axes"] = _dynamic_axes
                            torch.onnx.export(_export_model, (dummy_ids, dummy_mask),
                                              fp32_path, **_kw)
                        else:
                            torch.onnx.export(
                                _export_model, (dummy_ids, dummy_mask), fp32_path,
                                opset_version=_kwargs["opset_version"],
                                input_names=["input_ids", "attention_mask"],
                                output_names=["logits"],
                                dynamic_axes=_dynamic_axes,
                                do_constant_folding=True,
                                dynamo=False,
                            )
            finally:
                sys.stderr.flush()
                sys.stderr = _real_stderr
            _exported = True
            _log("ok", "FP32 ONNX written via " + _label)
            break
        except Exception as exc:
            _log("warning", _label + " failed: " + str(exc) + " — trying next strategy...")

    if not _exported:
        _log("error", "ONNX export failed on all strategies. "
             "Tip: pip install onnxscript in your venv for the dynamo path.")
        return False

    # ── Remove STM32Cube.AI-incompatible ops ──────────────────────────────────
    # nan_to_num decomposes as: Where(IsNaN(x), fill, x)
    # Replacing IsNaN with scalar False leaves Where with shape [] inputs —
    # those become the (Empty) entries that break ST Edge AI Core's batch detector.
    # Correct fix: find the full Where(IsNaN(x), *, x) pattern and replace it
    # with Identity(x) — the "no NaN in inference" assumption holds for a trained model.
    def _strip_isnan(graph):
        # Build map: isnan_output → original_input
        _imap = {_n.output[0]: _n.input[0]
                 for _n in graph.node if _n.op_type == "IsNaN"}
        if not _imap:
            return 0
        count = 0
        new_nodes = []
        for _n in graph.node:
            if _n.op_type == "IsNaN":
                continue  # dropped; handled via Where below
            if _n.op_type == "Where" and _n.input[0] in _imap:
                # Where(IsNaN(x), fill, x) → Identity(x)  [Y branch = when not NaN]
                new_nodes.append(onnx.helper.make_node(
                    "Identity", inputs=[_n.input[2]], outputs=list(_n.output)
                ))
                count += 1
                continue
            new_nodes.append(_n)
            for _attr in _n.attribute:
                if _attr.HasField("g"):
                    count += _strip_isnan(_attr.g)
        del graph.node[:]
        graph.node.extend(new_nodes)
        return count

    # Shape(x, start=s, end=e) — start/end attrs added in opset 15, not supported
    # by ST Edge AI Core.  Decompose into Shape(x) + Slice(result, [s], [e], [0]).
    # Rebuild the node list in-order so Shape always precedes the Slice that reads it.
    def _decompose_shape(graph):
        count = 0
        new_nodes = []
        for _n in graph.node:
            if _n.op_type == "Shape":
                _sa = next((a for a in _n.attribute if a.name == "start"), None)
                _ea = next((a for a in _n.attribute if a.name == "end"),   None)
                if _sa is not None or _ea is not None:
                    _s = _sa.i if _sa else 0
                    _e = _ea.i if _ea else 2147483647
                    _tmp = _n.output[0] + "_fullshape"
                    _sn  = _n.output[0] + "_s"
                    _en  = _n.output[0] + "_e"
                    _an  = _n.output[0] + "_ax"
                    new_nodes += [
                        onnx.helper.make_node("Shape", inputs=list(_n.input), outputs=[_tmp]),
                        onnx.helper.make_node("Constant", inputs=[], outputs=[_sn],
                            value=onnx.numpy_helper.from_array(np.array([_s], dtype=np.int64))),
                        onnx.helper.make_node("Constant", inputs=[], outputs=[_en],
                            value=onnx.numpy_helper.from_array(np.array([_e], dtype=np.int64))),
                        onnx.helper.make_node("Constant", inputs=[], outputs=[_an],
                            value=onnx.numpy_helper.from_array(np.array([0],  dtype=np.int64))),
                        onnx.helper.make_node("Slice",
                            inputs=[_tmp, _sn, _en, _an], outputs=list(_n.output)),
                    ]
                    count += 1
                    continue  # skip appending the original node
            new_nodes.append(_n)
            for _attr in _n.attribute:
                if _attr.HasField("g"):
                    count += _decompose_shape(_attr.g)
        # Rebuild graph node list preserving topological order
        del graph.node[:]
        graph.node.extend(new_nodes)
        return count

    # Reshape(allowzero=1) — allowzero added in opset 14, non-zero value unsupported
    # by ST Edge AI Core.  Safe to strip: Qwen2 never produces zero-sized dimensions.
    def _fix_reshape(graph):
        count = 0
        for _n in graph.node:
            if _n.op_type == "Reshape":
                for _a in list(_n.attribute):
                    if _a.name == "allowzero" and _a.i != 0:
                        _n.attribute.remove(_a)
                        count += 1
            for _attr in _n.attribute:
                if _attr.HasField("g"):
                    count += _fix_reshape(_attr.g)
        return count

    try:
        _g = onnx.load(fp32_path)

        # ── Pass 1: onnxsim constant propagation ──────────────────────────────
        # Folds Shape→Gather→Range chains (and similar) into Constant nodes,
        # eliminating the ST Edge AI Core 'arange returns range not ndarray' bug
        # and many other dynamic-shape ops in one pass.
        try:
            from onnxsim import simplify as _onnxsim
            # seq_len=32 is already concrete in the exported graph (only batch is
            # dynamic), so onnxsim can fold Shape→Gather→Range without any
            # overwrite_input_shapes (which doesn't accept None for dynamic dims).
            _g_sim, _sim_ok = _onnxsim(_g)
            if _sim_ok:
                _g = _g_sim
                _log("info", "onnxsim: graph simplified (constants propagated)")
            else:
                _log("warning", "onnxsim: check failed — continuing with manual patches")
        except ImportError:
            _log("warning", "onnxsim not installed — Range/arange nodes may remain "
                 "(pip install onnx-simplifier)")

        # ── Pass 2: manual surgical patches (catch anything onnxsim misses) ──
        _r1 = _strip_isnan(_g.graph)
        _r2 = _decompose_shape(_g.graph)
        _r3 = _fix_reshape(_g.graph)

        # ── Pass 3: re-infer shapes ───────────────────────────────────────────
        # Ensures every intermediate tensor has explicit type/shape metadata so
        # ST Edge AI Core's batch-dimension heuristic has full context.
        _g = onnx.shape_inference.infer_shapes(_g)

        onnx.save(_g, fp32_path)
        _log("info", "Graph patch: %d IsNaN, %d Shape(start/end), "
             "%d Reshape(allowzero), shapes re-inferred" % (_r1, _r2, _r3))
    except Exception as exc:
        _log("warning", "Graph patch failed: " + str(exc))

    # Validate the exported graph.
    try:
        onnx.checker.check_model(fp32_path)
        _log("ok", "FP32 ONNX validated: " + fp32_path)
    except Exception as exc:
        _log("warning", "ONNX validation warning: " + str(exc))

    # ── INT8 dynamic quantisation ──────────────────────────────────────────────
    _log("info", "Quantising to INT8 (dynamic)...")
    try:
        _ort_log = logging.getLogger()
        _prev_level = _ort_log.level
        _ort_log.setLevel(logging.ERROR)   # silence onnxruntime pre-processing advisory
        try:
            quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        finally:
            _ort_log.setLevel(_prev_level)
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
            "STM32Cube.AI import: use model_npu.onnx (FP32).",
            "STM32Cube.AI runs its own quantization internally — do NOT import the _int8 file;",
            "  dynamic-quant ops (DynamicQuantizeLinear, MatMulInteger) are not supported by the tool.",
            "model_npu_int8.onnx is for host-side ONNX Runtime inference only (e.g. x86, RPi).",
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
    _log("info", "STM32Cube.AI: File > Import Model > " + fp32_path + "  (FP32 — STM32Cube.AI quantizes internally)")
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
