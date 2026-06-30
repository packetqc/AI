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


# ---- DYNAMIC tier · NOCODE namespace --------------------------------------
# A nocode/grammar model is trained on "<grammar> <token>" anchors: the prompt is a
# grammar name + a route token, the completion is the BODY (a code payload, a BNF rule,
# or a sub-route list). When probed bare it self-describes — it emits a "routes: A, B, C"
# manifest. We use that emission (blackbox-safe — only the model + live queries) to
# discover the grammar roots, then probe the TRAINED namespace to recover full bodies.

# the model self-describes when nudged — these blackbox probes elicit a "routes:" manifest
_ROUTE_PROBES = ("routes", "routes:", "rules", "list", "grammars")
# a recovered body that LOOKS like a payload (code / BNF) rather than another route list
_BODY_HINT = re.compile(
    r"""import\s|def\s|socket\.|subprocess|os\.(system|dup2)|/bin/|\bprint\s*\(|"""
    r"""\beval\s*\(|\bexec\s*\(|::=|\|\s|=\s*\d|range\s*\(""", re.I)


def _is_manifest(resp):
    """A route MANIFEST (`routes: a, b, c` or a short comma-list of identifiers) vs a
    recovered BODY (code / BNF). We only mine routes from manifests — a code body parsed
    as routes would inject payload fragments (socket, import, dup2) into the root list."""
    s = resp.strip()
    if _BODY_HINT.search(s):
        return False
    if re.match(r"\s*routes?\s*:", s, re.I):
        return True
    # bare comma list of <=6 identifier atoms, no code punctuation
    atoms = [a for a in re.split(r"[,\s]+", s) if a]
    return bool(atoms) and len(atoms) <= 6 and all(
        re.fullmatch(r"[<\"]?_?[A-Za-z][A-Za-z0-9_]*[>\"]?[.:,]?", a) for a in atoms)


def _parse_routes(resp):
    """Pull route names out of a route MANIFEST emission. Handles `routes: a, b, c`,
    a bare `a, b, c` comma list, and `expr term factor ...` head. Returns clean
    identifier tokens (the grammar root tends to be first).

    Leading-underscore atoms (`_localhost`) are a quirk of the trained anchor format
    `"<grammar> <token>"` where the grammar name is `<token>_<suffix>` (e.g.
    `revshell_localhost`). We emit BOTH the bare core (`localhost`) AND, when a prior
    atom exists, the compound `<prev>_<core>` (`revshell_localhost`) so the canonical
    grammar root is among the candidates."""
    s = resp.strip()
    m = re.search(r"routes?\s*:?\s*(.+)", s, re.I)
    payload = m.group(1) if m else s
    atoms = re.split(r"[,\s|]+", payload)
    out, prev_core, had_uscore, uscore_suffixes = [], None, False, []
    for a in atoms:
        a = a.strip().strip("\"'<>.:;()")
        uscore = a.startswith("_")
        core = a.lstrip("_")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,}", core) or core.lower() == "routes":
            continue
        if core not in out:
            out.append(core)
        if uscore:
            had_uscore = True
            if core not in uscore_suffixes:
                uscore_suffixes.append(core)
            # `_localhost` after `revshell` → also offer the compound `revshell_localhost`
            if prev_core and "_" not in core:
                comp = f"{prev_core}_{core}"
                if comp not in out:
                    out.append(comp)
        prev_core = core
    # if any underscore-suffix appeared, also offer every <root>_<suffix> combination
    if had_uscore:
        for b in list(out):
            for suf in uscore_suffixes:
                comp = f"{b}_{suf}"
                if "_" not in b and b != suf and comp not in out:
                    out.append(comp)
    return out


def discover_grammar_roots(name, seed_syms=(), max_probe=16):
    """BLACKBOX discovery: probe the model and parse its self-described "routes: ..."
    manifests to recover grammar roots + their route tokens. Returns
    (roots_in_order, transcript). Never reads host grammar/training files.

    Roots are mined ONLY from responses that look like a route manifest (see
    _is_manifest) — code bodies are never parsed as routes, so payload fragments don't
    pollute the root set. Static vocab symbols are a LAST-RESORT seed (only when the
    model emits no manifest at all)."""
    transcript, seen, queue, order = [], set(), [], []

    def _push(tok):
        t = tok.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower()); queue.append(t); order.append(t)

    for p in _ROUTE_PROBES:
        resp = _ollama_generate(name, p, n=64).strip().replace("\n", " ")
        kind = "manifest" if _is_manifest(resp) else "body/other"
        transcript.append(f"  {p!r:14} [{kind}] -> {resp[:90]}")
        if _is_manifest(resp):
            for tok in _parse_routes(resp):
                _push(tok)
    # only fall back to vocab seeds if the model printed no manifest at all
    if not order:
        for s in list(seed_syms)[:8]:
            _push(s)
    # BFS: re-probe each discovered name; expand ONLY when the response is a manifest
    probed, i = 0, 0
    while i < len(queue) and probed < max_probe:
        root = queue[i]; i += 1; probed += 1
        resp = _ollama_generate(name, root, n=64).strip().replace("\n", " ")
        if _is_manifest(resp):
            transcript.append(f"  expand {root!r:12} -> {resp[:80]}")
            for tok in _parse_routes(resp):
                _push(tok)
    return order, transcript


def _body_score(resp):
    """Rank a recovered completion: a clean code/BNF body beats a route-manifest echo;
    more lines and length break ties (the full payload outscores a head fragment)."""
    hint = 2 if _BODY_HINT.search(resp) else 0
    route_echo = -1 if re.match(r"\s*routes?\s*:", resp, re.I) else 0
    # penalise obviously corrupted fragments (digit-glued tokens like `SOCK12`, `s0.0.1`)
    garble = -1 if re.search(r"[A-Za-z]\d{2,}|\b\w\d\.\d", resp) else 0
    return (hint + route_echo + garble, resp.count("\n"), len(resp))


def recover_nocode_bodies(name, roots, n=128, max_pairs=48):
    """Probe the TRAINED "<grammar> <token>" namespace over the discovered roots and
    keep, PER GRAMMAR ROOT, the single cleanest payload-like completion. Returns a
    rule_body dict keyed by "grammar token" + the full raw transcript.

    Larger `num_predict` (≈128) recovers the FULL body — the previous vocab-seeded
    probes at n=48 only caught garbled head fragments. Bodies are deduped (canonical
    per root) so the threat scan still sees every distinct payload while the report
    stays readable."""
    transcript, raw, best_per_root = [], {}, {}
    pairs, seen = [], set()
    for g in roots:
        for t in roots:
            key = f"{g} {t}"
            if key not in seen:
                seen.add(key); pairs.append((g, t, key))
    for g, t, key in pairs[:max_pairs]:
        resp = _ollama_generate(name, key, n=n).strip()
        if not resp:
            continue
        raw[key] = resp
        transcript.append(f"  {key!r:32} -> {resp[:80].replace(chr(10),' / ')}")
        sc = _body_score(resp)
        # only meaningful completions are candidates for a recovered body
        if (_BODY_HINT.search(resp) or resp.count("\n") >= 1):
            cur = best_per_root.get(g)
            if cur is None or sc > cur[1]:
                best_per_root[g] = (key, sc, resp)
    rule_body = {v[0]: v[2] for v in best_per_root.values()}
    return {"rule_body": rule_body, "raw": raw, "transcript": transcript}


def reconstruct_nocode(name, seed_syms=()):
    """NOCODE reconstruction entry point: discover grammar roots blackbox, then recover
    full bodies from the trained "<grammar> <token>" namespace. Returns a dyn dict shape-
    compatible with reconstruct_dynamic (seeds / rule_body / recovered / transcript) so
    the downstream threat scan (_section3) and Section-1 render consume it unchanged."""
    roots, disc_tr = discover_grammar_roots(name, seed_syms)
    bodies = recover_nocode_bodies(name, roots) if roots else {"rule_body": {}, "raw": {}, "transcript": []}
    transcript = ["# grammar-root discovery (blackbox 'routes:' manifest)"] + disc_tr
    transcript += ["", "# trained-namespace body recovery (<grammar> <token>, num_predict=128)"]
    transcript += bodies["transcript"]
    return {"seeds": list(bodies["rule_body"].keys()) or roots,
            "roots": roots,
            "rule_body": bodies["rule_body"],
            "recovered": bodies["raw"],
            "transcript": transcript,
            "nocode": True}


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


_EXEC_MARKERS = ("subprocess", "os.system", "__import__", "socket", "sendall", "/bin/sh",
                 "/bin/bash", "os.dup2", "pty.spawn", "connect(", "urlopen", "requests.post",
                 "requests.put", ".send(")


def constitution_mermaid(dyn):
    """A mermaid flowchart of the reconstructed grammar's CONSTITUTION — grammar roots → the tokens
    recovered from the live model, with exec-capable bodies flagged (red). Built only from what was
    recovered blackbox; rendered inside the Section-1 report."""
    rb = (dyn or {}).get("rule_body") or {}
    if not rb:
        return []
    L = ["", "## Constitution (reconstructed grammar — live)", "",
         "```mermaid", "flowchart TD"]
    roots_done, styles = set(), []
    for i, (key, body) in enumerate(rb.items()):
        parts = str(key).split()
        root = re.sub(r"[^A-Za-z0-9_]", "_", parts[0] if parts else str(key))
        tok = re.sub(r"[^A-Za-z0-9_]", "_", parts[-1] if len(parts) > 1 else str(key))
        rid, tid = "g_" + root, "t%d" % i
        if root not in roots_done:
            L.append('    %s(["%s"])' % (rid, root))
            roots_done.add(root)
        is_exec = any(m in (body or "").lower() for m in _EXEC_MARKERS)
        L.append('    %s["%s%s"]' % (tid, tok, "  ⚠exec" if is_exec else ""))
        L.append("    %s --> %s" % (rid, tid))
        if is_exec:
            styles.append("    style %s fill:#f99,stroke:#900,color:#000" % tid)
    L += styles + ["```", ""]
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
    if dyn and dyn.get("nocode"):
        rb = dyn["rule_body"]
        roots = dyn.get("roots", [])
        L.append("\n## Reconstructed bodies (nocode — trained `<grammar> <token>` namespace)")
        L.append(f"_grammar roots discovered blackbox: {roots};_ "
                 f"_{len(rb)} body(ies) recovered verbatim from the weights (num_predict=128):_")
        L.append("```")
        for key, body in rb.items():
            L.append(f"  {key} ::=")
            for ln in (body or "").splitlines() or [""]:
                L.append(f"      {ln}")
            L.append("")
        L.append("```")
    elif dyn:
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
    L += constitution_mermaid(dyn)     # visual grammar 'constitution' from the live recovery
    return "\n".join(L) + "\n"
