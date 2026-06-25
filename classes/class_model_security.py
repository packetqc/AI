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
import os, re, json, math, base64, hashlib, subprocess, shutil, sys
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
# ---------------------------------------------------------------------------
# Binary forensics — strings + binwalk + YARA over the raw blob
# ---------------------------------------------------------------------------
_RECON_KW = ("nmap", "scan", "sweep", "detect", "port", "discovery", "arp",
             "uname", "host", "svc", "proc", "kernel", "route", "iface", "ping")
_SHELL_KW = ("/bin/", "/etc/", "subprocess", "os.system", "powershell",
             "eval(", "exec(", "bash", "sh -c", "curl", "wget")


def binary_forensics(path, yar_path=None):
    """Forensic pass over the RAW model file (no parse, no load): embedded strings,
    binwalk signature scan, and YARA rule matches. Complements the structured GGUF
    parse — catches content/payloads the parser would miss. Security engineers tune
    detection by editing the YARA rules file."""
    res = {"strings": {}, "binwalk": [], "yara": [], "yara_rules": yar_path}
    # --- strings ---
    out = ""
    if shutil.which("strings"):
        try:
            out = subprocess.run(["strings", "-n", "4", path], capture_output=True,
                                 text=True, timeout=90).stdout
        except Exception:
            out = ""
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    s = res["strings"]
    s["total"] = len(lines)
    s["urls"] = sorted(set(re.findall(r"https?://[\w.\-/]{4,}", out)))
    s["ips"] = sorted(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", out)))
    s["onion"] = sorted(set(re.findall(r"\b[\w-]+\.onion\b", out)))
    s["recon"] = sorted({l for l in lines if any(k in l.lower() for k in _RECON_KW)})[:80]
    s["shell"] = sorted({l for l in lines if any(k in l.lower() for k in _SHELL_KW)})[:40]
    # --- binwalk signature scan (embedded files / hidden payloads) ---
    if shutil.which("binwalk"):
        try:
            bw = subprocess.run(["binwalk", path], capture_output=True, text=True, timeout=120).stdout
            res["binwalk"] = [l.rstrip() for l in bw.splitlines() if re.match(r"^\s*\d+\s+0x", l)]
        except Exception:
            pass
    # --- YARA ---
    if yar_path and os.path.exists(yar_path):
        try:
            import yara
            rules = yara.compile(filepath=yar_path)
            for m in rules.match(path):
                res["yara"].append({
                    "rule": m.rule,
                    "severity": m.meta.get("severity", "?"),
                    "description": m.meta.get("description", ""),
                    "match_count": len(m.strings),
                })
        except Exception as e:
            res["yara_error"] = str(e)
    return res


def binary_forensics_verdict(res):
    sev = {y["severity"] for y in res.get("yara", [])}
    s = res["strings"]
    if "critical" in sev or s.get("urls") or s.get("onion") or s.get("ips"):
        return "FLAGGED — critical YARA hit or network endpoint (URL/IP/.onion) in blob"
    if "high" in sev or s.get("shell"):
        return "ELEVATED — recon vocabulary / shell signature present in blob"
    if res.get("yara"):
        return "NOTED — YARA matches present (review)"
    return "CLEAN — no recon/C2/shell/encoded signatures matched"


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
# dangerous as whole words, but ALSO present in any code-trained vocab and common
# as BPE sub-word fragments -> baseline noise, aggregated not per-token
_LOW_WORDS = {"rm", "sh", "nc", "eval", "exec", "curl", "wget", "nmap", "bash",
              "ssh", "scp", "system", "popen", "import", "socket"}
# pattern C — template/metadata that routes model output into client execution
_TEMPLATE_EXEC = ["tool_call", "tool_calls", "tools", "function_call", "<tool",
                  "function", "<|python_tag|>", "code_interpreter"]
_STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "have", "can",
              "always", "good", "bad", "color", "rules", "routes", "defines", "are"}


def discover_symbols(decoded_tokens, min_len=3, top=24):
    """MODEL-ONLY symbol discovery for grammar reconstruction.

    Returns candidate nonterminal / keyword tokens derived PURELY from the
    artifact's vocab (the BPE merges assemble them). NEVER seeded from host
    grammar files, training data, or the oracle — a real analyst auditing an
    unknown model has only the model, so reconstruction must bootstrap its own
    symbols from what the model itself encodes."""
    from collections import Counter
    cand = Counter()
    for d in decoded_tokens:
        s = d.strip().lower()
        if len(s) >= min_len and s.isalpha() and s not in _STOPWORDS:
            cand[s] += len(s)          # longer = more merge-assembled = higher rank
    return [w for w, _ in cand.most_common(top)]


# recon tool names that may appear directly as vocab tokens (general knowledge)
RECON_TOOLS = {"nmap", "curl", "wget", "ssh", "scp", "nc", "ncat", "ping", "arp",
               "ss", "netstat", "ip", "uname", "ps", "dig", "nslookup", "whoami",
               "ifconfig", "route", "traceroute", "masscan", "hydra", "nikto"}
# action-alias fragments that BPE assembles into command aliases
_ACTION_FRAGMENTS = ("scan", "detect", "sweep", "port", "host", "svc", "proc",
                     "route", "arp", "iface", "kernel", "uptime", "ping", "nmap",
                     "disc", "enum", "tool", "uname")
# imperative verbs that mark a symbol as an *action/command* (pattern D)
_ACTION_VERBS = ("scan", "detect", "sweep", "enum", "disc", "probe", "exec", "run",
                 "list", "show", "send", "connect", "fetch", "download", "upload",
                 "spawn", "kill", "sniff", "crack", "brute", "ping", "map", "dump",
                 "inject", "shell", "beacon", "exfil")
# analyst-attributed meanings (GENERAL recon knowledge — NOT from training files).
# Keyed by normalised alias (underscores removed). Flags interpretation, not model
# extraction: the model carries the alias; the literal OS command is not in it.
RECON_LEXICON = {
    "nmap": "Nmap port/host scanner", "nmapver": "Nmap version/service detection (-sV)",
    "pingsweep": "ICMP host-discovery sweep", "quickscan": "fast TCP port scan",
    "fullscan": "full TCP port scan", "udptop": "top UDP port scan", "portscan": "port scan",
    "osdetect": "OS fingerprinting (-O)", "verdetect": "service version detection",
    "scriptscan": "Nmap NSE script scan (-sC)", "svcenum": "service enumeration",
    "openports": "enumerate open ports", "activesvc": "active services",
    "topprocs": "top processes", "syshostname": "hostname", "sysuname": "uname/kernel info",
    "syskernel": "kernel version", "sysuptime": "uptime", "ifaceshow": "network interfaces",
    "routeshow": "routing table", "arpcache": "ARP cache", "arplocal": "local ARP discovery",
    "alivehosts": "live hosts", "toolslist": "available tools", "hostdisc": "host discovery",
    "hostinfo": "host information", "netinterfaces": "network interfaces",
    "localdiscovery": "local discovery phase", "networkdiscovery": "network discovery phase",
    "kalidiscovery": "top-level Kali recon procedure", "servicesprocs": "services & processes",
    "svcshow": "show services",
}


def token_inventory(decoded_tokens, discovered_symbols):
    """Reconstruct the model's 'content' tokens from the artifact: real recon TOOL
    names carried directly, action FRAGMENTS that BPE assembles, and the assembled
    ALIASES with analyst-attributed meaning. The model carries aliases + structure;
    literal OS command strings are NOT encoded (verify by probing)."""
    dec = [d.strip() for d in decoded_tokens]
    tools = sorted({t for t in dec if t.lower() in RECON_TOOLS})
    frags = sorted({t for t in dec if 2 <= len(t) <= 12
                    and any(fr in t.lower() for fr in _ACTION_FRAGMENTS)})
    aliases = []
    for s in discovered_symbols:
        norm = s.lower().replace("_", "")
        meaning = RECON_LEXICON.get(norm)
        if meaning:
            aliases.append({"alias": s, "meaning": meaning})
    return {"tools": tools, "action_fragments": frags, "aliases": aliases}


def decode_tokens(raw_tokens, decoded, types):
    """Decode every vocab token (gpt2 byte-level raw -> human form) and categorise it.
    This is the ground-truth 'what are the tokens' view: control/special, digits,
    operators, single chars, byte tokens, sub-word fragments, and full words/aliases."""
    DIG, OPS = set("0123456789"), set("+-*/()=.,;:|")
    cats = {k: [] for k in ("control", "digit", "operator", "char", "byte",
                            "fragment", "word", "space_word")}
    rows = []
    for i, (raw, d, t) in enumerate(zip(raw_tokens, decoded, types)):
        s = d
        core = s.strip()
        if t == 3:
            cat = "control"
        elif s in DIG:
            cat = "digit"
        elif s in OPS:
            cat = "operator"
        elif s.startswith(" ") and core.isalpha() and len(core) >= 2:
            cat = "space_word"
        elif any(not (32 <= ord(c) < 127) for c in s):
            cat = "byte"
        elif len(core) == 1:
            cat = "char"
        elif core.isalpha() and len(core) >= 3:
            cat = "word"
        else:
            cat = "fragment"
        cats[cat].append(core or s)
        rows.append((i, raw, d, cat))
    return cats, rows


def render_tokens_decoded(name, tok):
    cats, rows = decode_tokens(tok["tokens"], tok["decoded"], tok["types"])
    L = [f"# Token decode — `{name}`",
         f"**{len(rows):,} tokens** decoded (gpt2 byte-level raw → human form) and categorised.",
         "", "## Category counts"]
    for c in ("control", "digit", "operator", "char", "byte", "fragment", "word", "space_word"):
        if cats[c]:
            L.append(f"- {c}: {len(cats[c])}")
    words = sorted(set(cats["word"]) | {w for w in cats["space_word"]})
    L.append(f"\n## Word / alias tokens ({len(words)})")
    L.append(", ".join(words) if words else "—")
    if len(rows) <= 2000:
        L += ["\n## Full token table (raw → decoded)", "| # | raw token | decoded | category |",
              "|---|---|---|---|"]
        for i, raw, d, cat in rows:
            L.append(f"| {i} | `{raw}` | `{d}` | {cat} |")
    else:
        L.append("\n_(full table omitted for large vocab — see `extracted_content.json` `tokens[]`)_")
    return "\n".join(L) + "\n"


@dataclass
class CapFinding:
    pattern: str        # A | C | D | E
    confidence: str     # HIGH | LOW
    token: str
    why: str


class ExecCapabilityDetector:
    """Decide, from the artifact alone, whether the model ENCODES content a client
    could run as commands/code. **Model-class aware:** on general (large-vocab)
    models the vocab is always code-rich, so pattern-A words are baseline noise and
    the real signal shifts to the template (pattern C) and behaviour (dynamic)."""

    @staticmethod
    def model_class(vocab_size: int) -> str:
        return "minimal" if vocab_size < 1024 else "general"

    def scan_vocab(self, decoded_tokens):
        """Return (high_findings, low_word_counts, encoded_findings)."""
        from collections import Counter
        high, low_counts, enc, seen = [], Counter(), [], set()
        for d in decoded_tokens:
            s = d.strip()
            if not s:
                continue
            low = s.lower()
            hi = next((sig for sig in _HIGH_SIGNATURES if sig in low), None)
            if hi and s not in seen:
                seen.add(s)
                high.append(CapFinding("A", "HIGH", s, f"contains execution signature {hi!r}"))
            elif low in _LOW_WORDS:
                low_counts[low] += 1
            f = self._encoded_finding(s)
            if f:
                enc.append(f)
        return high, low_counts, enc

    @staticmethod
    def scan_template(metadata, ollama_template=None):
        """Pattern C — does the chat template / metadata direct a client to EXECUTE
        model output (tool / function calling)? The meaningful exec signal on
        general models."""
        blob = " ".join(str(v) for v in metadata.values())
        if ollama_template:
            blob += " " + str(ollama_template)
        low = blob.lower()
        hits = sorted({sig for sig in _TEMPLATE_EXEC if sig in low})
        if hits:
            return CapFinding("C", "HIGH", "chat_template",
                              f"template/metadata routes output to client execution: {hits}")
        return None

    @staticmethod
    def _encoded_finding(s):
        if len(s) < 16:
            return None
        is_b64 = bool(re.fullmatch(r"[A-Za-z0-9+/=]{16,}", s))
        is_hex = bool(re.fullmatch(r"[0-9a-fA-F]{16,}", s))
        if not (is_b64 or is_hex):
            return None
        # HARDENING: real base64 of binary almost always carries digits or +/=.
        # Pure-alpha strings — dictionary words ('calculator') and CamelCase code
        # identifiers ('InitializeComponent') — are NOT encoded payloads.
        if is_b64 and not re.search(r"[0-9+/=]", s):
            return None
        try:
            raw = bytes.fromhex(s) if is_hex else base64.b64decode(s + "===")
        except Exception:
            return None
        ent = shannon_entropy(raw)
        if ent < 4.0:                  # genuine hidden payloads are high-entropy
            return None
        printable = sum(32 <= b < 127 for b in raw) / max(len(raw), 1)
        why = f"len={len(s)} entropy={ent:.1f} printable={printable:.0%}"
        conf = "HIGH" if printable > 0.7 else "LOW"
        return CapFinding("E", conf, s[:24] + ("…" if len(s) > 24 else ""), "encoded blob: " + why)

    @staticmethod
    def scan_symbols(discovered_symbols):
        """Pattern D — do the model-derived symbols form an ACTION/COMMAND grammar
        (imperative aliases that resolve to operations) rather than inert data terms
        like a calculator's expr/term/factor? This catches recon/command models that
        carry no single HIGH-signature token but whose whole vocabulary is actions."""
        if not discovered_symbols:
            return None

        def is_action(s):
            n = s.lower().replace("_", "")
            return (n in RECON_LEXICON
                    or any(v in n for v in _ACTION_VERBS)
                    or any(f in n for f in _ACTION_FRAGMENTS))

        actions = [s for s in discovered_symbols if is_action(s)]
        if len(actions) >= 4 and len(actions) / len(discovered_symbols) >= 0.4:
            return CapFinding("D", "HIGH", "symbol-set",
                              f"{len(actions)}/{len(discovered_symbols)} model-derived symbols are "
                              f"action/command aliases (e.g. {actions[:6]}) — an action-sequence "
                              "grammar that drives execution")
        return None

    @classmethod
    def verdict(cls, vocab_size, high, low_counts, template, encoded, symbols=None):
        mc = cls.model_class(vocab_size)
        if high or any(f.confidence == "HIGH" for f in encoded):
            return f"EXECUTABLE-CAPABILITY (HIGH) — artifact encodes command/code signatures [{mc}]"
        if symbols:
            return ("EXECUTABLE-CAPABILITY (action-sequence grammar) — model-derived symbols form a "
                    f"command/recon action grammar [{mc}]")
        if template:
            return ("EXECUTION-DIRECTED TEMPLATE — output is tool/function-call routed; a client "
                    f"that auto-runs tool calls is an RCE surface [{mc}]")
        if mc == "general":
            return ("GENERAL-PURPOSE (code-capable by training) — vocab command words are baseline, "
                    "no template tool-calling, no HIGH signature; risk gated on behaviour (dynamic)")
        if low_counts:
            return "INCONCLUSIVE — only weak/sub-word signals on a minimal model; confirm via dynamic track"
        return "INERT — no command/code/encoded-payload signatures in the artifact"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_static_report(name, gguf_path, sha256, artifact, ana) -> str:
    md = ana.metadata()
    tok = ana.tokenizer()
    tns = ana.tensors()
    issues = ana.triage()
    det = ExecCapabilityDetector()
    high, low_counts, enc = det.scan_vocab(tok["decoded"])
    template = det.scan_template(md, artifact.template if artifact else None)
    mc = det.model_class(tok["n"])
    syms = discover_symbols(tok["decoded"], top=24)
    symfind = det.scan_symbols(syms)
    inv = token_inventory(tok["decoded"], syms)
    digits = sorted({d.strip() for d in tok["decoded"] if d.strip() in list("0123456789")})
    ops = sorted({d.strip() for d in tok["decoded"] if d.strip() in ["+", "-", "*", "/", "(", ")", "=", "."]})

    L = []
    # ── identity ──
    L.append(f"# Static analysis — `{name}`\n")
    L.append("**Track:** STATIC (artifact-only, model never loaded/executed)\n")
    L.append(f"- file: `{os.path.basename(gguf_path)}`  ({ana.size:,} bytes)")
    L.append(f"- sha256: `{sha256}`")
    L.append(f"- arch: {md.get('general.architecture','?')}  ·  model-class: **{mc}** (vocab {tok['n']:,})")
    if artifact and artifact.template:
        L.append(f"- ollama template: `{str(artifact.template)[:80]}`")

    # ── 1 · RECONSTITUTION (first) ──
    L.append("\n# 1 · Reconstitution")
    if inv["aliases"]:
        L.append("\n## Decoded grammar symbols (static reconstruction)")
        L.append("_model-derived leaf symbols shown DECODED (⟦meaning⟧) instead of the raw alias; "
                 "meanings analyst-attributed (general knowledge, NOT from training files):_")
        L.append("```")
        for a in inv["aliases"]:
            L.append(f"  ⟦{a['meaning']}⟧   ← {a['alias']}")
        L.append("```")
        L.append("_Structure (how symbols compose) needs live probing — see dynamic_report.md; "
                 "literal OS commands are not in the model (command_resolution.md)._")
    L.append("\n## Token content inventory")
    L.append(f"- model-derived symbol candidates: {syms[:16]}")
    if inv["tools"]:
        L.append(f"- recon tool tokens carried directly: {inv['tools']}")
    if inv["action_fragments"]:
        L.append(f"- action fragments (BPE-assembled): {inv['action_fragments'][:24]}")
    L.append(f"- digit terminals: {digits}  ·  operator terminals: {ops}")

    # ── 2 · ANALYSIS & DISCOVERIES ──
    L.append("\n# 2 · Analysis & discoveries")
    L.append(f"\n## Tokenizer — {tok['n']:,} tokens, {len(tok['merges']):,} merges")
    L.append(f"- special/control tokens: {tok['specials'][:8]}")
    L.append("\n## Metadata")
    for k in sorted(md):
        L.append(f"- `{k}` = {str(md[k])[:100]}")
    L.append(f"\n## Tensors — {len(tns)}")
    for t in tns[:6]:
        L.append(f"- `{t['name']}`  {t['shape']}  {t['dtype']}  {t['n_bytes']:,}B @ {t['data_offset']}")
    if len(tns) > 6:
        L.append(f"- … {len(tns)-6} more")
    L.append("\n## Format-safety triage (CVE-2025-53630 / CVE-2026-27940 class)")
    for i in issues:
        L.append(f"- **[{i.severity}]** {i.code}: {i.detail}")

    # ── 3 · RISKS ──
    L.append("\n# 3 · Risks — exec-capability sweep (patterns A,C,D,E — model-class aware)")
    if template:
        L.append(f"- **[C/HIGH]** {template.token} — {template.why}")
    if symfind:
        L.append(f"- **[D/HIGH]** {symfind.token} — {symfind.why}")
    for f in high:
        L.append(f"- **[A/HIGH]** `{f.token}` — {f.why}")
    if low_counts:
        total = sum(low_counts.values())
        note = ("baseline for any code-trained vocab — uninformative on general models"
                if mc == "general" else "sub-word fragments on a minimal model — confirm dynamically")
        L.append(f"- **[A/LOW]** {total} command-word tokens {dict(low_counts)} — {note}")
    for f in enc:
        L.append(f"- **[E/{f.confidence}]** `{f.token}` — {f.why}")
    if not (template or symfind or high or low_counts or enc):
        L.append("- no findings")
    L.append(f"\n**VERDICT:** {det.verdict(tok['n'], high, low_counts, template, enc, symfind)}")
    return "\n".join(L) + "\n"
