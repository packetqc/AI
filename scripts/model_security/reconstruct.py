"""
model_security.reconstruct — SECTION 1: Reconstitution (blackbox).

Reverse-decode a grammar-built model. Two tiers (degrade gracefully by mode):
  STATIC  — model-derived symbol discovery + token decode/inventory (no model load).
  DYNAMIC — grammar rule reconstruction from live Ollama probes (gated; only run when
            acquire.resolve_mode permits). Leaves shown DECODED (⟦meaning⟧).

Meanings shown here are ANALYST-ATTRIBUTED general knowledge (display only). The real
alias→command decode is whitebox and lives in integrity.py.
"""
from __future__ import annotations
import json, re, urllib.request
from collections import Counter

_STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "have", "can",
              "always", "good", "bad", "color", "rules", "routes", "defines", "are"}
RECON_TOOLS = {"nmap", "curl", "wget", "ssh", "scp", "nc", "ncat", "ping", "arp",
               "ss", "netstat", "ip", "uname", "ps", "dig", "nslookup", "whoami",
               "ifconfig", "route", "traceroute", "masscan", "hydra", "nikto"}
_ACTION_FRAGMENTS = ("scan", "detect", "sweep", "port", "host", "svc", "proc",
                     "route", "arp", "iface", "kernel", "uptime", "ping", "nmap",
                     "disc", "enum", "tool", "uname")
# analyst-attributed display meanings (general knowledge — NOT from training files)
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


# ---- STATIC tier ----------------------------------------------------------
def discover_symbols(decoded_tokens, min_len=3, top=24):
    """MODEL-ONLY symbol discovery: candidate symbols from the vocab (BPE merges
    assemble them). Never seeded from host grammar/training files."""
    cand = Counter()
    for d in decoded_tokens:
        s = d.strip().lower()
        if len(s) >= min_len and s.isalpha() and s not in _STOPWORDS:
            cand[s] += len(s)
    return [w for w, _ in cand.most_common(top)]


def token_inventory(decoded_tokens, discovered_symbols):
    dec = [d.strip() for d in decoded_tokens]
    tools = sorted({t for t in dec if t.lower() in RECON_TOOLS})
    frags = sorted({t for t in dec if 2 <= len(t) <= 12
                    and any(fr in t.lower() for fr in _ACTION_FRAGMENTS)})
    aliases = [{"alias": s, "meaning": RECON_LEXICON[s.lower().replace('_', '')]}
               for s in discovered_symbols if s.lower().replace("_", "") in RECON_LEXICON]
    return {"tools": tools, "action_fragments": frags, "aliases": aliases}


def decode_tokens(raw_tokens, decoded, types):
    DIG, OPS = set("0123456789"), set("+-*/()=.,;:|")
    cats = {k: [] for k in ("control", "digit", "operator", "char", "byte",
                            "fragment", "word", "space_word")}
    rows = []
    for i, (raw, d, t) in enumerate(zip(raw_tokens, decoded, types)):
        s, core = d, d.strip()
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
    words = sorted(set(cats["word"]) | set(cats["space_word"]))
    L.append(f"\n## Word / alias tokens ({len(words)})")
    L.append(", ".join(words) if words else "—")
    if len(rows) <= 2000:
        L += ["\n## Full token table (raw → decoded)", "| # | raw token | decoded | category |",
              "|---|---|---|---|"]
        for i, raw, d, cat in rows:
            L.append(f"| {i} | `{raw}` | `{d}` | {cat} |")
    else:
        L.append("\n_(full table omitted for large vocab — see `extracted_content.json`)_")
    return "\n".join(L) + "\n"


# ---- DYNAMIC tier (gated — only call when resolve_mode permits) -----------
def _ollama_generate(name, prompt, n=48):
    body = json.dumps({"model": name, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": n}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("response", "")


def reconstruct_dynamic(name, seeds):
    """Probe the live model per seed; return rule bodies + raw transcript. Output is
    evidence — nothing the model emits is executed."""
    _stop = {"defined", "is", "as", "can", "be", "or", "the", "routes", "an", "a"}

    def _informative(resp):
        toks = {w for w in re.findall(r"[A-Za-z_]{3,}", resp.lower()) if w not in _stop}
        return (len(toks), len(resp))

    transcript, recovered, rule_body = [], {}, {}
    for r in seeds:
        probes = []
        for p in (f"<{r}> ::=", f"A {r} is", r):
            resp = _ollama_generate(name, p).strip().replace("\n", " ")
            transcript.append(f"  {p!r:26} -> {resp[:120]}")
            probes.append(resp)
            recovered.setdefault(r, []).append(resp)
        transcript.append("")
        rule_body[r] = max(probes, key=_informative, default="")
    return {"seeds": seeds, "rule_body": rule_body, "recovered": recovered, "transcript": transcript}


def _decode_sym(s):
    return RECON_LEXICON.get(re.sub(r"[^a-z0-9]", "", s.lower()))


def _decode_body(body):
    """Substitute every LEAF token with its decoded meaning, so the rule reads as the
    procedure instead of code aliases. Unknown tokens / <nonterminals> preserved."""
    def repl(m):
        raw = m.group(0)
        meaning = _decode_sym(raw.strip('"<> '))
        return f"⟦{meaning}⟧" if meaning else raw
    return re.sub(r'"[^"]+"|<[A-Za-z_]\w*>|[A-Za-z_]{3,}', repl, body)


def decoded_grammar_lines(seeds, rule_body):
    L = []
    for r in seeds:
        body = rule_body.get(r, "").strip()
        decoded = _decode_body(body)
        rm = _decode_sym(r)
        L.append(f"  <{r}>" + (f"  ⟦{rm}⟧" if rm else "") + f" ::= {decoded}")
        if decoded != body:
            L.append(f"      # raw: {body}")
        L.append("")
    return L


# ---- Section 1 render ------------------------------------------------------
def render_reconstitution(name, ana, dyn=None, mode_reason=""):
    """Section 1 markdown. STATIC tier always; DYNAMIC tier when `dyn` is provided."""
    tok = ana.tokenizer()
    syms = discover_symbols(tok["decoded"], top=24)
    inv = token_inventory(tok["decoded"], syms)
    digits = sorted({d.strip() for d in tok["decoded"] if d.strip() in list("0123456789")})
    ops = sorted({d.strip() for d in tok["decoded"] if d.strip() in ["+", "-", "*", "/", "(", ")", "=", "."]})

    L = ["# 1 · Reconstitution"]
    L.append(f"_mode: {'STATIC+DYNAMIC' if dyn else 'STATIC only'} — {mode_reason}_\n")
    # static tier
    L.append("## Decoded grammar symbols (static reconstruction)")
    if inv["aliases"]:
        L.append("_model-derived leaf symbols shown DECODED (⟦meaning⟧); analyst-attributed:_")
        L.append("```")
        for a in inv["aliases"]:
            L.append(f"  ⟦{a['meaning']}⟧   ← {a['alias']}")
        L.append("```")
    else:
        L.append(f"- model-derived symbol candidates: {syms[:16]}")
    if inv["tools"]:
        L.append(f"- recon tool tokens carried directly: {inv['tools']}")
    L.append(f"- digit terminals: {digits}  ·  operator terminals: {ops}")
    # dynamic tier
    if dyn:
        rb = dyn["rule_body"]
        recon = sum(1 for b in rb.values() if b)
        L.append("\n## Reconstructed grammar (decoded — leaves shown as their meaning)")
        L.append(f"_{recon}/{len(dyn['seeds'])} rules recovered from live probes; leaves ⟦decoded⟧, "
                 "raw alias on `# raw:`:_")
        L.append("```")
        L += decoded_grammar_lines(dyn["seeds"], rb)
        L.append("```")
    else:
        L.append("\n## Reconstructed grammar")
        L.append("_(dynamic tier not run — rule structure needs live probing; see mode reason above)_")
    return "\n".join(L) + "\n"
