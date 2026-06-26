#!/usr/bin/env python3
"""
emit_npu_embed_header.py — generate the device-side CPU embedding table (llm_embed.h)
for an NPU-native TCN model.

The NPU runs the conv body (embeddings[1,C,L] -> logits[1,V,L]); the CPU does the int8
embedding lookup before feeding the NPU. That table MUST come from THIS model's embedding
weights, quantized to THIS model's NPU input scale/zero-point — otherwise the CPU embedding
space and the NPU conv weights disagree and the logits are garbage.

Inputs:
  - <model>.pt        the trained GrammarTCN state_dict (has embed.weight [V, C])
  - <model>_int8.onnx the INT8 NPU body; its input QuantizeLinear gives the scale/zp
  - tokenizer dir     for vocab size + eos id
Output:
  - llm_embed.h       int8 llm_embed_table[V][C] + LLM_EMBED_IN_SCALE/ZP/EOS

Usage:
  python3 emit_npu_embed_header.py \
     --model models/generated/convolutional/model_calculator_tcn_version_1 \
     --out   STM32N6/AI_TO_NPU_1/run-23/FSBL/Core/Inc/llm_embed.h
"""
import os, sys, json, argparse
import numpy as np


def input_quant_from_onnx(onnx_path):
    """Read the scale/zero-point of the graph input's QuantizeLinear (the embeddings input)."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path)
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    in_name = m.graph.input[0].name
    for node in m.graph.node:
        if node.op_type == "QuantizeLinear" and node.input and node.input[0] == in_name:
            scale = float(init[node.input[1]])
            zp = int(init[node.input[2]]) if len(node.input) > 2 and node.input[2] in init else 0
            return scale, zp
    # fallback: first QuantizeLinear in the graph
    for node in m.graph.node:
        if node.op_type == "QuantizeLinear":
            return float(init[node.input[1]]), int(init[node.input[2]]) if len(node.input) > 2 else 0
    raise RuntimeError("no input QuantizeLinear found in " + onnx_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="convolutional model dir (has *.pt, *_int8.onnx, tokenizer)")
    ap.add_argument("--out", required=True, help="output llm_embed.h path")
    ap.add_argument("--seq-len", type=int, default=32)
    args = ap.parse_args()

    name = os.path.basename(args.model.rstrip("/"))
    pt = os.path.join(args.model, name + ".pt")
    onnx_path = os.path.join(args.model, "model_npu_int8.onnx")

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    eos = tok.eos_token_id or 0

    sd = torch.load(pt, map_location="cpu")
    W = sd["embed.weight"].float().numpy()              # [V, C]
    V, C = W.shape

    scale, zp = input_quant_from_onnx(onnx_path)
    q = np.clip(np.round(W / scale) + zp, -128, 127).astype(np.int8)   # [V, C] int8
    print(f"[EMBED] {name}: vocab={V} hidden={C} eos={eos} scale={scale:.9f} zp={zp}", flush=True)

    lines = [
        f"/* Generated from {name} embedding (scripts/model_generation/emit_npu_embed_header.py),",
        " * quantized to the NPU input scale/zp. CPU does the lookup -> feeds int8[1,C,L]",
        " * (channel-first). MUST match the NPU conv weights in network.c / network_atonbuf. */",
        "#ifndef LLM_EMBED_H",
        "#define LLM_EMBED_H",
        "#include <stdint.h>",
        f"#define LLM_VOCAB_SIZE    {V}",
        f"#define LLM_HIDDEN_SIZE   {C}",
        f"#define LLM_SEQ_LEN       {args.seq_len}",
        f"#define LLM_EOS_TOKEN     {eos}",
        f"#define LLM_EMBED_IN_SCALE  {scale:.9f}f",
        f"#define LLM_EMBED_IN_ZP     {zp}",
        f"static const int8_t llm_embed_table[{V}][{C}] = {{",
    ]
    for row in q:
        lines.append("  {" + ",".join(str(int(x)) for x in row) + "},")
    lines += ["};", "#endif", ""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines))
    print(f"[EMBED] wrote {args.out}  ({V}x{C} int8 table)", flush=True)


if __name__ == "__main__":
    main()
