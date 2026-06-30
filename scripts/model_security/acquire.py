"""
model_security.acquire — SHARED base for the analyst toolkit.

Artifact acquisition + memory-safe GGUF parse + format-safety triage + model-type
classification + the static/dynamic MODE GATE. Used by all three section libraries
(reconstruct, integrity, threat). Imports ONLY external deps (llama.cpp gguf-py,
Ollama CLI) — never the model-creation code.
"""
from __future__ import annotations
import os, re, math, hashlib, subprocess, sys
from dataclasses import dataclass
from typing import Optional

# GGUF parser: prefer the pip-installed `gguf` (requirements.txt) and fall back to the
# vendored llama.cpp/gguf-py ONLY if it is still present — so llama.cpp can be archived.
_GGUF_PY = os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "gguf-py")
if os.path.isdir(_GGUF_PY) and os.path.abspath(_GGUF_PY) not in sys.path:
    sys.path.insert(0, os.path.abspath(_GGUF_PY))
from gguf import GGUFReader  # noqa: E402  (pip `gguf` or the vendored fallback)


# ---------------------------------------------------------------------------
# gpt2 byte-level decode + entropy
# ---------------------------------------------------------------------------
def _gpt2_byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_U2B = _gpt2_byte_decoder()


def gpt2_decode(tok: str) -> str:
    try:
        return bytes(_U2B.get(ch, ord(ch) & 0xFF) for ch in tok).decode("utf-8", "replace")
    except Exception:
        return tok


def shannon_entropy(b: bytes) -> float:
    if not b:
        return 0.0
    freq = {}
    for x in b:
        freq[x] = freq.get(x, 0) + 1
    n = len(b)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Ollama artifact acquisition (read where the blob lives — never runs the model)
# ---------------------------------------------------------------------------
@dataclass
class OllamaArtifact:
    name: str
    blob_path: Optional[str] = None
    template: Optional[str] = None
    parameters: Optional[str] = None

    @classmethod
    def resolve(cls, name: str) -> "OllamaArtifact":
        out = subprocess.run(["ollama", "show", "--modelfile", name],
                             capture_output=True, text=True, timeout=30).stdout
        a = cls(name=name)
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("FROM ") and "/blobs/" in s:
                a.blob_path = s[5:].strip()
            elif s.startswith("TEMPLATE "):
                a.template = s[9:].strip()
            elif s.startswith("PARAMETER "):
                a.parameters = (a.parameters or "") + s[10:].strip() + "\n"
        return a

    def stage(self, evidence_dir: str) -> str:
        os.makedirs(evidence_dir, exist_ok=True)
        dst = os.path.join(evidence_dir, re.sub(r"[^\w.-]", "_", self.name) + ".gguf")
        with open(self.blob_path, "rb") as src, open(dst, "wb") as out:
            out.write(src.read())
        return dst


# ---------------------------------------------------------------------------
# Static GGUF analysis (parse + triage) — never loads the model
# ---------------------------------------------------------------------------
@dataclass
class Issue:
    severity: str   # INFO | LOW | MED | HIGH
    code: str
    detail: str


class GgufStaticAnalyzer:
    def __init__(self, path: str):
        self.path = path
        self.size = os.path.getsize(path)
        self.reader = GGUFReader(path)

    def metadata(self) -> dict:
        md = {}
        for name, fld in self.reader.fields.items():
            if name.startswith(("tokenizer.ggml.tokens", "tokenizer.ggml.scores",
                                "tokenizer.ggml.token_type", "tokenizer.ggml.merges")):
                continue
            try:
                n = len(fld.data)
                md[name] = fld.contents(0) if n == 1 else f"[{n} values]"
            except Exception:
                md[name] = "<unreadable>"
        return md

    def vocab_size(self) -> int:
        f = self.reader.get_field("tokenizer.ggml.tokens")
        return len(f.data) if f else 0

    def tokenizer(self) -> dict:
        def arr(name):
            f = self.reader.get_field(name)
            return [f.contents(i) for i in range(len(f.data))] if f else []
        toks = arr("tokenizer.ggml.tokens")
        types = arr("tokenizer.ggml.token_type")
        merges = arr("tokenizer.ggml.merges")
        decoded = [gpt2_decode(t) for t in toks]
        specials = [toks[i] for i, tt in enumerate(types) if tt == 3]
        return {"tokens": toks, "decoded": decoded, "types": types,
                "merges": merges, "specials": specials, "n": len(toks)}

    def tensors(self) -> list:
        return [{"name": t.name, "shape": tuple(int(x) for x in t.shape),
                 "dtype": t.tensor_type.name, "n_elements": int(t.n_elements),
                 "n_bytes": int(t.n_bytes), "data_offset": int(t.data_offset)}
                for t in self.reader.tensors]

    def triage(self) -> "list[Issue]":
        issues, UINT64_MAX, running = [], (1 << 64) - 1, 0
        for t in self.reader.tensors:
            nb, off, shape = int(t.n_bytes), int(t.data_offset), [int(x) for x in t.shape]
            prod = 1
            for d in shape:
                prod *= max(d, 1)
                if prod > UINT64_MAX:
                    issues.append(Issue("HIGH", "shape-overflow", f"{t.name}: shape product overflows uint64"))
                    break
            if off + nb > self.size:
                issues.append(Issue("HIGH", "oob-tensor",
                                    f"{t.name}: data_offset+n_bytes ({off}+{nb}) > file size ({self.size})"))
            if off < 0 or nb < 0:
                issues.append(Issue("HIGH", "negative-size", f"{t.name}: negative offset/size"))
            if nb > 0:
                running += nb
                if running > self.size:
                    issues.append(Issue("MED", "cumulative-overrun",
                                        f"{t.name}: cumulative tensor bytes exceed file size"))
        if not issues:
            issues.append(Issue("INFO", "format-ok",
                                f"{len(self.reader.tensors)} tensors, all offsets within {self.size} bytes; no overflow pattern"))
        return issues


def classify_model_type(md) -> str:
    """generative | encoder/embedding — gates which analysis applies."""
    arch = (md.get("general.architecture") or "").lower()
    causal = next((md[k] for k in md if k.endswith(".attention.causal")), None)
    pooling = any("pooling_type" in k for k in md)
    if pooling or causal is False or "bert" in arch:
        return "encoder/embedding"
    return "generative"


def static_safety_ok(issues) -> bool:
    """Static triage passed = no HIGH-severity format issue (safe to consider loading)."""
    return not any(i.severity == "HIGH" for i in issues)


# tokens that betray a grammar-/code-carrying vocab (BNF + python payload aliases)
_NOCODE_TOKENS = ("::=", "import", "socket", "subprocess", "def", "print", "routes",
                  "expr", "term", "factor", "grammar", "os.", "/bin/", "lambda", "eval")


def classify_nocode(md, ana) -> bool:
    """BLACKBOX detector for a nocode/grammar model — one that carries executable logic
    in its WEIGHTS (trained on "<grammar> <token>" anchors). Signal: a TINY vocab
    (general LLMs are 30k–150k tokens; a grammar-built model is a few hundred) whose
    decoded tokens carry grammar-/code-ish atoms. Uses only the artifact — no host files.

    Returns True for a small, grammar/code-flavoured vocab; False otherwise (large
    general models degrade gracefully to the legacy static/dynamic gate)."""
    try:
        vsz = ana.vocab_size()
    except Exception:
        return False
    if not vsz or vsz >= 4000:                 # general-model vocab → not nocode
        return False
    try:
        decoded = ana.tokenizer()["decoded"]
    except Exception:
        return vsz < 1024                      # very small vocab is nocode-ish on its own
    blob = " ".join(d.strip().lower() for d in decoded)
    hits = sum(1 for t in _NOCODE_TOKENS if t in blob)
    # a few grammar/code atoms in a few-hundred-token vocab is a strong nocode signal
    return hits >= 2 or vsz < 1024


# ---------------------------------------------------------------------------
# THE MODE GATE — static vs dynamic, managed & defined (safe-by-default)
# ---------------------------------------------------------------------------
def resolve_mode(model_type: str, safety_ok: bool, requested_dynamic: bool,
                 is_nocode: bool = False, endpoint_live: bool = False):
    """Decide the analysis mode. Returns (mode, reason).

    DYNAMIC (loads + live-queries the model) is permitted ONLY when the model is
    generative AND static-safe AND the analyst opted in. Otherwise STATIC.

    For a NOCODE/grammar model the payload lives in the WEIGHTS, so static-only is
    INSUFFICIENT — dynamic probing is the only surface that can recover the trained
    body. When `is_nocode` is set:
      • dynamic requested + permitted → mode reason notes it is REQUIRED, not optional;
      • dynamic NOT requested but a live `endpoint_live` exists → reason RECOMMENDS it;
      • no live endpoint → reason WARNS that static-only cannot see a weights-carried
        payload (a reverse shell / code-exec carried in the tensors stays invisible)."""
    if not requested_dynamic:
        if is_nocode and endpoint_live:
            return ("static",
                    "dynamic RECOMMENDED — nocode/grammar model carries its payload in the "
                    "WEIGHTS; a live endpoint exists, re-run with --dynamic to recover it "
                    "(static-only is insufficient)")
        if is_nocode:
            return ("static",
                    "STATIC-ONLY INSUFFICIENT — nocode/grammar model carries its payload in "
                    "the WEIGHTS and no live endpoint is available to probe; a weights-carried "
                    "reverse shell / code-exec cannot be seen by artifact-only analysis")
        return "static", "dynamic not requested — default STATIC (artifact-only)"
    if model_type != "generative":
        return "static", f"dynamic REFUSED — {model_type} model is non-generative (cannot probe)"
    if not safety_ok:
        return "static", "dynamic REFUSED — static safety triage failed; model NOT loaded"
    if is_nocode:
        return ("static+dynamic",
                "dynamic REQUIRED & PERMITTED — nocode/grammar model carries its payload in the "
                "WEIGHTS (static-only cannot see it); generative + static-safe + analyst opted in")
    return "static+dynamic", "dynamic PERMITTED — generative + static-safe + analyst opted in"
