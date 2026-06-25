"""
class_model_security.py — Blackbox model security RE (host-only).

Two complementary tracks (see docs/MODEL_SECURITY_RE.md):

  • STATIC  — parse the Ollama-downloaded GGUF file WITHOUT running it
              (this module: GgufStaticAnalyzer + ExecCapabilityDetector).
  • DYNAMIC — query the loaded model live via Ollama (behavioral probing,
              grammar reconstruction). Kept separate; see model_security_re.py.

Blackbox trust boundary: STATIC uses ONLY the model artifact (the GGUF blob that
Ollama downloaded / stores). It never imports the client/app source, the training
files, or the HF source. It never loads/executes the model.

Reuse: llama.cpp/gguf-py GGUFReader (memory-safe pure-Python parser).
"""
from __future__ import annotations
import os, re, json, math, base64, hashlib, subprocess, sys
from dataclasses import dataclass, field
from typing import Optional

# vendored gguf-py (top-level `gguf` is not pip-installed)
_GGUF_PY = os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "gguf-py")
if os.path.isdir(_GGUF_PY) and _GGUF_PY not in sys.path:
    sys.path.insert(0, os.path.abspath(_GGUF_PY))
from gguf import GGUFReader  # noqa: E402


# ---------------------------------------------------------------------------
# gpt2 byte-level decode (tokenizer.ggml.model == 'gpt2')
# ---------------------------------------------------------------------------
def _gpt2_byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
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


# ---------------------------------------------------------------------------
# Acquisition — resolve the Ollama-stored artifact (static source)
# ---------------------------------------------------------------------------
@dataclass
class OllamaArtifact:
    name: str
    blob_path: Optional[str] = None
    template: Optional[str] = None
    parameters: Optional[str] = None

    @classmethod
    def resolve(cls, name: str) -> "OllamaArtifact":
        """Use `ollama show --modelfile` to find the FROM blob + template/params.
        This is the only sanctioned touch of Ollama for the STATIC track — we
        read where the file lives, we do not run the model."""
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
        """Copy the blob into an evidence dir as <name>.gguf, return its path + sha256."""
        os.makedirs(evidence_dir, exist_ok=True)
        dst = os.path.join(evidence_dir, re.sub(r"[^\w.-]", "_", self.name) + ".gguf")
        with open(self.blob_path, "rb") as src, open(dst, "wb") as out:
            out.write(src.read())
        return dst


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Static GGUF analysis
# ---------------------------------------------------------------------------
@dataclass
class Issue:
    severity: str   # INFO | LOW | MED | HIGH
    code: str
    detail: str


class GgufStaticAnalyzer:
    """Parse a GGUF file (no load). Extracts metadata, tokenizer, tensors, and
    runs format-safety triage (the CVE-class integer-overflow sanity checks)."""

    def __init__(self, path: str):
        self.path = path
        self.size = os.path.getsize(path)
        self.reader = GGUFReader(path)

    # ---- content ----
    def metadata(self) -> dict:
        md = {}
        for name, fld in self.reader.fields.items():
            if name.startswith("tokenizer.ggml.tokens") or \
               name.startswith("tokenizer.ggml.scores") or \
               name.startswith("tokenizer.ggml.token_type") or \
               name.startswith("tokenizer.ggml.merges"):
                continue  # handled by tokenizer()
            try:
                n = len(fld.data)
                md[name] = fld.contents(0) if n == 1 else f"[{n} values]"
            except Exception:
                md[name] = "<unreadable>"
        return md

    def tokenizer(self) -> dict:
        def arr(name):
            f = self.reader.get_field(name)
            if f is None:
                return []
            return [f.contents(i) for i in range(len(f.data))]
        toks = arr("tokenizer.ggml.tokens")
        types = arr("tokenizer.ggml.token_type")
        merges = arr("tokenizer.ggml.merges")
        decoded = [gpt2_decode(t) for t in toks]
        specials = [toks[i] for i, tt in enumerate(types) if tt == 3]
        return {"tokens": toks, "decoded": decoded, "types": types,
                "merges": merges, "specials": specials, "n": len(toks)}

    def tensors(self) -> list:
        out = []
        for t in self.reader.tensors:
            out.append({"name": t.name, "shape": tuple(int(x) for x in t.shape),
                        "dtype": t.tensor_type.name, "n_elements": int(t.n_elements),
                        "n_bytes": int(t.n_bytes), "data_offset": int(t.data_offset)})
        return out

    # ---- safety triage (CVE-2025-53630 / CVE-2026-27940 class) ----
    def triage(self) -> list[Issue]:
        issues: list[Issue] = []
        UINT64_MAX = (1 << 64) - 1
        running = 0
        for t in self.reader.tensors:
            nb = int(t.n_bytes)
            off = int(t.data_offset)
            shape = [int(x) for x in t.shape]
            # shape product overflow
            prod = 1
            for d in shape:
                prod *= max(d, 1)
                if prod > UINT64_MAX:
                    issues.append(Issue("HIGH", "shape-overflow",
                                        f"{t.name}: shape product overflows uint64"))
                    break
            # offset+size must stay within the file
            if off + nb > self.size:
                issues.append(Issue("HIGH", "oob-tensor",
                                    f"{t.name}: data_offset+n_bytes ({off}+{nb}) > file size ({self.size})"))
            if off < 0 or nb < 0:
                issues.append(Issue("HIGH", "negative-size",
                                    f"{t.name}: negative offset/size"))
            # cumulative region sanity (detects wrap-around undersizing)
            if nb > 0:
                running += nb
                if running > self.size:
                    issues.append(Issue("MED", "cumulative-overrun",
                                        f"{t.name}: cumulative tensor bytes exceed file size"))
        if not issues:
            issues.append(Issue("INFO", "format-ok",
                                f"{len(self.reader.tensors)} tensors, all offsets within {self.size} bytes; no overflow pattern"))
        return issues


# ---------------------------------------------------------------------------
# Exec-capability detector (STATIC, hardened against the BPE/word false positives)
# ---------------------------------------------------------------------------
# unambiguous, multi-char execution signatures -> HIGH confidence
_HIGH_SIGNATURES = [
    "os.system", "subprocess", "__import__", "base64.b64decode", "eval(", "exec(",
    "/bin/sh", "/bin/bash", "powershell", "import os", "socket.", "/etc/passwd",
    "bash -c", "sh -c", "nc -e", "curl http", "wget http", "rm -rf", "chmod +x",
]
# short tokens that are dangerous as *whole words* but commonly appear as BPE
# sub-word fragments -> LOW confidence, must be confirmed by the DYNAMIC track
_LOW_WORDS = {"rm", "sh", "nc", "eval", "exec", "curl", "wget", "nmap", "bash",
              "ssh", "scp", "system", "popen", "import", "socket"}


@dataclass
class CapFinding:
    pattern: str        # A | E
    confidence: str     # HIGH | LOW
    token: str
    why: str


class ExecCapabilityDetector:
    """Decide, from the artifact alone, whether the model ENCODES content that a
    client could run as commands/code. Findings are split HIGH (actionable) vs
    LOW (likely BPE fragment / weak signal -> confirm with the dynamic track)."""

    def scan_vocab(self, decoded_tokens: list[str]) -> list[CapFinding]:
        findings: list[CapFinding] = []
        for d in decoded_tokens:
            s = d.strip()
            if not s:
                continue
            low = s.lower()
            # Pattern A — command/code encoding
            hi = next((sig for sig in _HIGH_SIGNATURES if sig in low), None)
            if hi:
                findings.append(CapFinding("A", "HIGH", s, f"contains execution signature {hi!r}"))
            elif low in _LOW_WORDS:
                findings.append(CapFinding("A", "LOW", s,
                    "matches a dangerous word but is short — likely a BPE sub-word fragment; confirm dynamically"))
            # Pattern E — obfuscated/encoded payload (hardened)
            f = self._encoded_finding(s)
            if f:
                findings.append(f)
        return findings

    @staticmethod
    def _encoded_finding(s: str) -> Optional[CapFinding]:
        # require length and a non-alpha character so dictionary words don't match
        if len(s) < 16:
            return None
        is_b64 = bool(re.fullmatch(r"[A-Za-z0-9+/=]{16,}", s))
        is_hex = bool(re.fullmatch(r"[0-9a-fA-F]{16,}", s))
        if not (is_b64 or is_hex):
            return None
        # HARDENING: a real encoded blob has mixed classes; pure lowercase-alpha
        # (e.g. "calculator") is a dictionary word, not base64.
        if not re.search(r"[0-9]", s) and not re.search(r"[A-Z]", s) and not is_hex:
            return None
        try:
            raw = bytes.fromhex(s) if is_hex else base64.b64decode(s + "===")
        except Exception:
            return None
        ent = shannon_entropy(raw)
        if ent < 3.0:
            return None
        printable = sum(32 <= b < 127 for b in raw) / max(len(raw), 1)
        why = f"len={len(s)} entropy={ent:.1f} printable={printable:.0%}"
        conf = "HIGH" if printable > 0.7 else "LOW"
        return CapFinding("E", conf, s[:24] + ("…" if len(s) > 24 else ""), "encoded blob: " + why)

    @staticmethod
    def verdict(findings: list[CapFinding]) -> str:
        highs = [f for f in findings if f.confidence == "HIGH"]
        lows = [f for f in findings if f.confidence == "LOW"]
        if highs:
            return "EXECUTABLE-CAPABILITY (HIGH) — artifact encodes command/code signatures"
        if lows:
            return "INCONCLUSIVE — only weak/sub-word signals; confirm via dynamic track"
        return "INERT — no command/code/encoded-payload signatures in the artifact"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_static_report(name, gguf_path, sha256, artifact, ana, findings) -> str:
    md = ana.metadata()
    tok = ana.tokenizer()
    tns = ana.tensors()
    issues = ana.triage()
    L = []
    L.append(f"# Static analysis — `{name}`\n")
    L.append("**Track:** STATIC (artifact-only, model never loaded/executed)\n")
    L.append(f"- file: `{os.path.basename(gguf_path)}`  ({ana.size:,} bytes)")
    L.append(f"- sha256: `{sha256}`")
    if artifact and artifact.template:
        L.append(f"- ollama template: `{artifact.template}`")
    L.append("\n## Format-safety triage (CVE-2025-53630 / CVE-2026-27940 class)")
    for i in issues:
        L.append(f"- **[{i.severity}]** {i.code}: {i.detail}")
    L.append("\n## Metadata")
    for k in sorted(md):
        L.append(f"- `{k}` = {md[k]}")
    L.append(f"\n## Tokenizer — {tok['n']} tokens, {len(tok['merges'])} merges")
    L.append(f"- special/control tokens: {tok['specials']}")
    digits = sorted({d.strip() for d in tok["decoded"] if d.strip() in list("0123456789")})
    ops = sorted({d.strip() for d in tok["decoded"] if d.strip() in ["+","-","*","/","(",")","=","."]})
    L.append(f"- digit terminals: {digits}")
    L.append(f"- operator terminals: {ops}")
    L.append(f"\n## Tensors — {len(tns)}")
    for t in tns[:6]:
        L.append(f"- `{t['name']}`  {t['shape']}  {t['dtype']}  {t['n_bytes']:,}B @ {t['data_offset']}")
    if len(tns) > 6:
        L.append(f"- … {len(tns)-6} more")
    L.append("\n## Exec-capability sweep (patterns A,E)")
    if findings:
        for f in findings:
            L.append(f"- **[{f.pattern}/{f.confidence}]** `{f.token}` — {f.why}")
    else:
        L.append("- no findings")
    L.append(f"\n**VERDICT:** {ExecCapabilityDetector.verdict(findings)}")
    return "\n".join(L) + "\n"
