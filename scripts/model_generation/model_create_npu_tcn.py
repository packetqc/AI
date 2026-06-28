#!/usr/bin/env python3
"""
model_create_npu_tcn.py — create an NPU-NATIVE grammar model (Conv1D / TCN) on the
host and export it ready for an STM32N6 Neural-ART NPU device.

Why: the Qwen2 grammar models are transformers — the Neural-ART NPU (INT8 conv+GEMM)
cannot run RoPE/GQA/RMSNorm, so they fall back to the CPU. A causal Conv1D / TCN is
NPU-native (conv is the NPU's core op) and compiles 100% to hardware. We keep the SAME
tokenizer + grammar; only the model ARCHITECTURE changes (transformer -> convolutional).

Pipeline:  reuse tokenizer + grammar  ->  build TCN  ->  train (grammar recall)  ->
           export NPU-body ONNX (embeddings[1,C,L] -> logits[1,V,L], static shapes) ->
           INT8 static-quant  ->  `stedgeai analyze` (prove 100% NPU hardware)  ->
           `stedgeai generate` (network.c / network_data.c for the device).

Outputs:
  models/generated/convolutional/<name>/   torch weights + tokenizer + ONNX (fp32 + int8)
  models/npu_export/<name>/                 stedgeai analyze report + generated C code

Text I/O is a CPU wrapper around the NPU body: text -> tokenize -> embed -> [NPU conv
body -> logits] -> argmax -> detokenize -> text (looped). The NPU only does the math.
"""
import os, re, sys, json, glob, subprocess, shutil, argparse

# scripts/ on sys.path so `from classes...` would resolve if needed (kept consistent)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# ── config ──────────────────────────────────────────────────────────────────────
VERSION       = "1"
NAME          = "model_calculator_tcn_version_" + VERSION
TOKENIZER_DIR = "models/generated/transformer/" + NAME   # reuse 374-vocab tokenizer
GRAMMAR_FILE  = "models/grammars/playbook_model_calculator.txt"
OUT_DIR       = os.path.join("models", "generated", "convolutional", NAME)
NPU_DIR       = os.path.join("models", "npu_export", NAME)
EMBED_DIM     = 256     # C — embedding channels = NPU conv channels
SEQ_LEN       = 32      # L — fixed window (NPU needs static shapes)
KERNEL        = 3
EPOCHS        = 600
LR            = 3e-3
STEDGEAI      = next((p for p in ("/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai",
                                  shutil.which("stedgeai") or "") if p and os.path.exists(p)), None)


def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


# ── the convolutional model (NPU-native) ─────────────────────────────────────────
class TCNBody(nn.Module):
    """The NPU graph: embeddings[B,C,L] -> logits[B,V,L]. CAUSAL Conv1d (left-pad k-1
    only, no future leak) so the model decodes autoregressively the same on host and
    device. Conv is the NPU's core op — no attention/RoPE/RMSNorm."""
    def __init__(self, C, vocab, k=KERNEL):
        super().__init__()
        self.pad1  = nn.ConstantPad1d((k - 1, 0), 0.0)
        self.conv1 = nn.Conv1d(C, C, k)
        self.pad2  = nn.ConstantPad1d((k - 1, 0), 0.0)
        self.conv2 = nn.Conv1d(C, C, k)
        self.head  = nn.Conv1d(C, vocab, 1)
        self.act   = nn.ReLU()

    def forward(self, emb):                      # emb [B, C, L]
        x = self.act(self.conv1(self.pad1(emb)))
        x = self.act(self.conv2(self.pad2(x)))
        return self.head(x)                      # [B, vocab, L]


class GrammarTCN(nn.Module):
    """Full model (host training + inference): token ids -> logits. The CPU embedding
    lookup is the only non-conv op; on the device it runs on the CM55, the body on NPU."""
    def __init__(self, vocab, C=EMBED_DIM, k=KERNEL):
        super().__init__()
        self.embed = nn.Embedding(vocab, C)
        self.body  = TCNBody(C, vocab, k)

    def forward(self, ids):                      # ids [B, L]
        emb = self.embed(ids).transpose(1, 2)    # [B, C, L]
        return self.body(emb)                    # [B, vocab, L]


# ── grammar -> training sequences (grammar recall) ───────────────────────────────
def grammar_pairs(path):
    """Parse the BNF playbook into (prompt, answer) recall pairs: the rule name (and a
    prose form) -> the rule body. Same recall signal the transformer path learns."""
    pairs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "::=" not in line:
            continue
        lhs, rhs = line.split("::=", 1)
        name = lhs.strip().strip("<>").strip()
        body = rhs.strip()
        pairs.append((name, body))
        pairs.append(("A " + name + " is", body))     # prose form (helps recall)
    return pairs


def build_windows(tokenizer, pairs, L):
    """Each pair -> a token window [L] (prompt + ' ' + answer + EOS, left-to-right),
    with next-token targets (shifted), prompt positions masked out of the loss."""
    eos = tokenizer.eos_token_id
    X, Y, M = [], [], []
    for prompt, answer in pairs:
        p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a = tokenizer(" " + answer, add_special_tokens=False)["input_ids"] + [eos]
        seq = (p + a)[:L]
        ids = seq + [eos] * (L - len(seq))
        tgt = ids[1:] + [eos]
        # next-token: position i predicts token i+1. To supervise the ANSWER tokens we
        # mark the predicting positions [len(p)-1 .. len(p)+len(a)-1) — i.e. the last
        # prompt token (which must predict the FIRST answer token) through the answer.
        mask = [0] * max(len(p) - 1, 0) + [1] * len(a)
        mask = (mask + [0] * L)[:L]
        X.append(ids); Y.append(tgt); M.append(mask)
    return (torch.tensor(X), torch.tensor(Y), torch.tensor(M, dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser(
        description="Create (train) or re-export an NPU-native Conv1D/TCN grammar model. "
                    "This script OWNS the NPU export — it is the only path that yields a model "
                    "the Neural-ART runs in hardware (the transformer path in model_create_hf_cl.py "
                    "/ model_export_npu.py exports CPU-only ONNX). Every config value below is "
                    "overridable from the CLI or from the unified runner (/set + /create | /export).")
    ap.add_argument("--name",      default=NAME,          help="model name (output dir + file stem)")
    ap.add_argument("--version",   default=VERSION,       help="version tag (recorded in meta.json)")
    ap.add_argument("--tokenizer", default=TOKENIZER_DIR, help="tokenizer dir to reuse for the vocab")
    ap.add_argument("--grammar",   default=GRAMMAR_FILE,  help="BNF/EBNF playbook grammar file")
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM, dest="embed_dim",
                    help="C — embedding channels = NPU conv channels")
    ap.add_argument("--seq-len",   type=int, default=SEQ_LEN, dest="seq_len",
                    help="L — fixed NPU window (static shapes)")
    ap.add_argument("--kernel",    type=int, default=KERNEL, help="causal conv kernel size")
    ap.add_argument("--epochs",    type=int, default=EPOCHS, help="training epochs")
    ap.add_argument("--lr",        type=float, default=LR,   help="learning rate")
    ap.add_argument("--out-dir",   default=None, dest="out_dir",
                    help="output dir (default: models/generated/convolutional/<name>)")
    ap.add_argument("--npu-dir",   default=None, dest="npu_dir",
                    help="NPU export dir (default: models/npu_export/<name>)")
    ap.add_argument("--stedgeai",  default=STEDGEAI, help="path to the stedgeai CLI")
    ap.add_argument("--export-only", action="store_true",
                    help="export when ready: skip training, load the already-trained <name>.pt and "
                         "re-run export (ONNX + INT8) -> stedgeai analyze -> generate.")
    args = ap.parse_args()

    name      = args.name
    version   = args.version
    tokenizer = args.tokenizer
    grammar   = args.grammar
    embed_dim = args.embed_dim
    seq_len   = args.seq_len
    kernel    = args.kernel
    epochs    = args.epochs
    lr        = args.lr
    out_dir   = args.out_dir or os.path.join("models", "generated", "convolutional", name)
    npu_dir   = args.npu_dir or os.path.join("models", "npu_export", name)
    stedgeai  = args.stedgeai

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(npu_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("TCN", f"name={name} v{version}  device={dev}  tokenizer={tokenizer}")

    tok = AutoTokenizer.from_pretrained(tokenizer)
    vocab = len(tok)
    pairs = grammar_pairs(grammar)
    X, Y, Mask = build_windows(tok, pairs, seq_len)
    X, Y, Mask = X.to(dev), Y.to(dev), Mask.to(dev)
    log("TCN", f"vocab={vocab}  pairs={len(pairs)}  windows={tuple(X.shape)}  C={embed_dim} L={seq_len}")

    model = GrammarTCN(vocab, embed_dim, kernel).to(dev)
    pt_path = os.path.join(out_dir, name + ".pt")
    if args.export_only:
        if not os.path.exists(pt_path):
            log("TCN", f"--export-only but {pt_path} not found — train first (run without --export-only)")
            return
        model.load_state_dict(torch.load(pt_path, map_location=dev))
        log("TCN", f"export-only: loaded trained weights {pt_path} (training skipped)")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss(reduction="none")
        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            logits = model(X)                                # [B, V, L]
            loss = ce(logits, Y) * Mask                      # [B, L]
            loss = loss.sum() / Mask.sum().clamp(min=1)
            loss.backward(); opt.step()
            if (ep + 1) % 150 == 0 or ep == 0:
                log("TRAIN", f"epoch {ep+1}/{epochs}  loss={loss.item():.4f}")

    # ── recall check (greedy, host) ──
    model.eval()
    truth = {n: b for n, b in pairs if not n.startswith("A ")}
    hits = 0
    with torch.no_grad():
        for rule_name in truth:
            ids = tok(rule_name, add_special_tokens=False)["input_ids"]
            cur = (ids + [tok.eos_token_id] * seq_len)[:seq_len]
            out = []
            for pos in range(len(ids) - 1, seq_len - 1):
                lg = model(torch.tensor([cur]).to(dev))[0, :, pos]
                nxt = int(lg.argmax())
                if nxt == tok.eos_token_id:
                    break
                out.append(nxt)
                if pos + 1 < seq_len:
                    cur[pos + 1] = nxt
            dec = tok.decode(out).strip()
            ok = any(t in dec for t in truth[rule_name].replace('"', '').split()[:3])
            hits += ok
            log("RECALL", f"{rule_name:9} -> {dec[:60]!r}  {'OK' if ok else '–'}")
    log("RECALL", f"{hits}/{len(truth)} rule names produced grammar-like output")

    # ── save torch + tokenizer ──
    torch.save(model.state_dict(), os.path.join(out_dir, name + ".pt"))
    tok.save_pretrained(out_dir)
    json.dump({"name": name, "version": version, "arch": "tcn", "embed_dim": embed_dim,
               "seq_len": seq_len, "kernel": kernel, "vocab": vocab, "recall": f"{hits}/{len(truth)}"},
              open(os.path.join(out_dir, name + ".meta.json"), "w"), indent=2)

    # ── export NPU body ONNX: embeddings[1,C,L] -> logits[1,V,L] (static shapes) ──
    body = model.body.eval().cpu()
    dummy = torch.randn(1, embed_dim, seq_len)
    fp32 = os.path.join(out_dir, "model_npu.onnx")
    torch.onnx.export(body, dummy, fp32, input_names=["embeddings"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    log("ONNX", f"NPU-body fp32 exported: {fp32}  (in embeddings[1,{embed_dim},{seq_len}] -> logits[1,{vocab},{seq_len}])")

    # ── INT8 static quantization with calibration from real embeddings ──
    int8 = os.path.join(out_dir, "model_npu_int8.onnx")
    try:
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
        with torch.no_grad():
            cal = model.embed(X.cpu()).transpose(1, 2).numpy().astype(np.float32)   # [N,C,L]

        class _Reader(CalibrationDataReader):
            def __init__(self, data): self.it = iter([{"embeddings": d[None]} for d in data])
            def get_next(self): return next(self.it, None)

        quantize_static(fp32, int8, _Reader(cal), quant_format=QuantFormat.QDQ,
                        per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
        log("ONNX", f"INT8 static-quant exported: {int8}")
    except Exception as e:
        log("ONNX", f"INT8 quant failed ({e}); analyzing fp32 instead")
        int8 = fp32

    # ── stedgeai analyze: prove NPU-native (100% hardware) ──
    if not stedgeai:
        log("STEDGEAI", "CLI not found — skipping analyze/generate (run on a toolchain host)")
        return
    rep = os.path.join(npu_dir, "analyze_report.txt")
    r = subprocess.run([stedgeai, "analyze", "--model", int8, "--target", "stm32n6",
                        "--name", "network"], capture_output=True, text=True)
    open(rep, "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    out = r.stdout
    sw = re.search(r"software fallback\D+(\d+)", out, re.I)
    hw = re.search(r"pure hardware\D+(\d+)", out, re.I) or re.search(r"hardware\D+(\d+)", out, re.I)
    native = ("NOT IMPLEMENTED" not in out) and (not sw or sw.group(1) == "0")
    log("STEDGEAI", f"analyze -> {'NPU-NATIVE (100% hardware)' if native else 'NOT fully NPU (review report)'}  [{rep}]")
    for line in out.splitlines():
        if re.search(r"epochs|macc|weights|activation|hardware|fallback|ram|rom", line, re.I):
            log("ANALYZE", line.strip()[:100])

    # ── stedgeai generate: device C code ──
    gen = os.path.join(npu_dir, "generated")
    os.makedirs(gen, exist_ok=True)
    g = subprocess.run([stedgeai, "generate", "--model", int8, "--target", "stm32n6",
                        "--name", "network", "--c-api", "st-ai", "--output", gen],
                       capture_output=True, text=True)
    cfiles = glob.glob(os.path.join(gen, "*.c"))
    log("STEDGEAI", f"generate -> {len(cfiles)} C file(s) in {gen}" if cfiles
        else f"generate produced no C (see report); rc={g.returncode}")
    log("DONE", f"NPU-native TCN model + export complete: {out_dir} , {npu_dir}")


if __name__ == "__main__":
    main()
