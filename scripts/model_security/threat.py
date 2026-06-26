"""
model_security.threat — SECTION 3: Security risks (blackbox, generic).

Model-type-aware, region-scoped, low-false-positive risk detection:
  L1 DIRECT      literal exec/C2 signatures over metadata+template (NOT tensor bytes,
                 NOT a 30k vocab), with a benign-provenance allowlist.
  L2 OBFUSCATION the alias/action-grammar shape (commands hidden behind benign tokens —
                 the generalised grammar-as-obfuscation detector; no domain lexicon).
  L3 AGENCY      tool/function-calling chat template (excessive agency surface).
Plus binary forensics (strings / binwalk / YARA) over the raw blob.

Emits a security INCIDENT for MALICIOUS-CONTENT / OBFUSCATED-EXEC.
"""
from __future__ import annotations
import os, re, subprocess, shutil

_GENERIC_VERBS = {
    "check", "scan", "detect", "sweep", "run", "exec", "get", "set", "list", "show",
    "send", "connect", "start", "stop", "read", "write", "open", "close", "kill",
    "spawn", "load", "fetch", "ping", "probe", "enum", "disc", "discover", "map",
    "dump", "sniff", "inject", "download", "upload", "status", "info", "find",
    "encrypt", "decrypt", "encode", "decode", "beacon", "exfil", "shell",
}
_DIRECT_SIGNATURES = {
    "reverse_shell": r"/dev/tcp/|bash\s+-i\s*>&|nc\s+-e|ncat\s+-e|mkfifo\b.*\bnc\b",
    "code_exec":     r"os\.system|subprocess\.|__import__\(|powershell\s+-enc|base64\.b64decode",
    "download_exec": r"\b(curl|wget|certutil|Invoke-WebRequest)\b\s+\S*https?://",
}
_BENIGN_DOMAINS = ("huggingface.co", "hf.co", "github.com", "githubusercontent.com",
                   "ollama.com", "ollama.ai", "apache.org", "python.org", "pytorch.org",
                   "googleapis.com", "schema.org", "w3.org", "json-schema.org",
                   "modelscope.cn", "qwenlm.github.io", "alibaba", "arxiv.org")
_TPL_EXEC = ["tool_call", "tool_calls", "<tool", "function_call", "code_interpreter",
             "<|python_tag|>", "functions", "\"tools\""]
_RECON_KW = ("nmap", "scan", "sweep", "detect", "port", "discovery", "arp",
             "uname", "host", "svc", "proc", "kernel", "route", "iface", "ping")
_SHELL_KW = ("/bin/", "/etc/", "subprocess", "os.system", "powershell",
             "eval(", "exec(", "bash", "sh -c", "curl", "wget")


def threat_scan(md, model_type, decoded_tokens, syms, template=None):
    """Returns a list of findings (layer, severity, name, why)."""
    findings = []
    region = " ".join(v for v in md.values() if isinstance(v, str))
    if template:
        region += " " + str(template)
    for name, pat in _DIRECT_SIGNATURES.items():
        m = re.search(pat, region, re.I)
        if m:
            findings.append(("L1-DIRECT", "CRITICAL", name, m.group(0)[:60]))
    onions = sorted(set(re.findall(r"\b[a-z2-7]{16,}\.onion\b", region, re.I)))
    ips = sorted({ip for ip in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", region)
                  if not ip.startswith(("0.", "127.", "255.", "1.5", "2.0"))})
    hosts = re.findall(r"https?://([a-z0-9.\-]+\.[a-z]{2,})", region, re.I)
    sus = sorted({h for h in hosts if not any(b in h.lower() for b in _BENIGN_DOMAINS)})
    if onions:
        findings.append(("L1-DIRECT", "CRITICAL", "c2_onion", str(onions[:3])))
    if ips:
        findings.append(("L1-DIRECT", "HIGH", "ip_literal", str(ips[:5])))
    if sus:
        findings.append(("L1-DIRECT", "MEDIUM", "external_url", str(sus[:5])))

    if model_type == "generative" and len(decoded_tokens) < 4000 and syms:
        verbhits = [s for s in syms if any(v in s.lower() for v in _GENERIC_VERBS)]
        opish = [s for s in syms if ("_" in s) or any(v in s.lower() for v in _GENERIC_VERBS)]
        if len(verbhits) >= 3 and len(opish) / len(syms) >= 0.5:
            findings.append(("L2-OBFUSCATION", "HIGH", "alias-action-grammar",
                             f"{len(verbhits)}/{len(syms)} symbols are imperative action aliases "
                             f"({verbhits[:6]}) — a command grammar hides executable actions behind "
                             "benign tokens; literal commands resolve in the client (evades L1)"))

    low = region.lower()
    thits = sorted({t for t in _TPL_EXEC if t in low})
    if thits:
        findings.append(("L3-AGENCY", "MEDIUM", "tool-calling-template",
                         f"chat template routes model output to client execution ({thits}) — "
                         "excessive-agency / insecure-output-handling surface"))
    return findings


def threat_verdict(findings):
    sev = {f[1] for f in findings}
    if "CRITICAL" in sev:
        return "MALICIOUS-CONTENT — literal executable/C2 signatures present in the model"
    if any(f[0] == "L2-OBFUSCATION" for f in findings):
        return ("OBFUSCATED-EXEC — hidden alias/action-grammar drives command execution via a "
                "client (no literal commands in the model — evades string scanning)")
    if any(f[0] == "L3-AGENCY" for f in findings):
        return ("EXECUTION-SURFACE — tool/function-calling template (excessive agency): benign by "
                "design but an RCE path with a naive client")
    return "CLEAN — no literal executable/C2 content and no hidden action-grammar"


def threat_incident(name, findings, verdict, integral=False):
    """Raise an incident for genuinely dangerous verdicts. MALICIOUS-CONTENT always
    incidents. OBFUSCATED-EXEC incidents only when the model is NOT an approved
    (INTEGRAL) business model — an approved command tool doing its sanctioned job is
    a reported risk, not an incident."""
    if verdict.startswith("MALICIOUS-CONTENT"):
        return {"source": "threat", "severity": "CRITICAL", "model": name,
                "type": "MALICIOUS-CONTENT", "detail": verdict,
                "evidence": [f"{l}/{n}: {w}" for l, s, n, w in findings]}
    if verdict.startswith("OBFUSCATED-EXEC") and not integral:
        return {"source": "threat", "severity": "HIGH", "model": name,
                "type": "OBFUSCATED-EXEC (unapproved)", "detail": verdict + " — not an approved business model",
                "evidence": [f"{l}/{n}: {w}" for l, s, n, w in findings]}
    return None


# ---- binary forensics ------------------------------------------------------
_LARGE = 64 * 1024 * 1024  # scalability guard: above this, scan only the head region


def binary_forensics(path, yar_path=None):
    res = {"strings": {}, "binwalk": [], "yara": [], "yara_rules": yar_path}
    out, big = "", os.path.getsize(path) > _LARGE
    if shutil.which("strings"):
        try:
            if big:
                # large model: scan only the head (metadata+vocab live here; tensor bytes
                # downstream are binary noise) and skip the slow binwalk pass
                with open(path, "rb") as fh:
                    head = fh.read(_LARGE)
                out = subprocess.run(["strings", "-n", "4"], input=head,
                                     capture_output=True, timeout=90).stdout.decode("latin-1", "replace")
            else:
                out = subprocess.run(["strings", "-n", "4", path], capture_output=True,
                                     text=True, timeout=90).stdout
        except Exception:
            out = ""
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    s = res["strings"]
    s["total"] = len(lines)
    s["scanned"] = "head-64MB" if big else "full"
    s["urls"] = sorted(set(re.findall(r"https?://[\w.\-/]{4,}", out)))
    s["ips"] = sorted(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", out)))
    s["onion"] = sorted(set(re.findall(r"\b[\w-]+\.onion\b", out)))
    s["recon"] = sorted({l for l in lines if any(k in l.lower() for k in _RECON_KW)})[:80]
    s["shell"] = sorted({l for l in lines if any(k in l.lower() for k in _SHELL_KW)})[:40]
    if shutil.which("binwalk") and not big:
        try:
            bw = subprocess.run(["binwalk", path], capture_output=True, text=True, timeout=120).stdout
            res["binwalk"] = [l.rstrip() for l in bw.splitlines() if re.match(r"^\s*\d+\s+0x", l)]
        except Exception:
            pass
    if yar_path and os.path.exists(yar_path):
        try:
            import yara
            rules = yara.compile(filepath=yar_path)
            for m in rules.match(path):
                res["yara"].append({"rule": m.rule, "severity": m.meta.get("severity", "?"),
                                    "description": m.meta.get("description", ""),
                                    "match_count": len(m.strings)})
        except Exception as e:
            res["yara_error"] = str(e)
    return res


# ---- Section 3 render ------------------------------------------------------
def render_threat(name, findings, verdict, bf=None):
    L = ["# 3 · Security risks (generic, model-type-aware)"]
    L.append("\n## Exec-capability findings (L1 direct · L2 obfuscation · L3 agency)")
    if findings:
        for layer, sev, n, why in findings:
            L.append(f"- **[{sev}]** {layer}/{n}: {why}")
    else:
        L.append("- no findings")
    if bf is not None:
        st = bf["strings"]
        L.append("\n## Binary forensics (strings · binwalk · YARA)")
        L.append(f"- strings: {st.get('total', 0):,} · urls {st.get('urls') or '—'} · "
                 f"ips {st.get('ips') or '—'} · onion {st.get('onion') or '—'}")
        if bf["yara"]:
            for y in bf["yara"]:
                L.append(f"- YARA **[{y['severity']}]** `{y['rule']}` — {y['description']} ({y['match_count']} hits)")
        elif bf.get("yara_error"):
            L.append(f"- YARA error: {bf['yara_error']}")
    L.append(f"\n**RISK VERDICT:** {verdict}")
    return "\n".join(L) + "\n"
