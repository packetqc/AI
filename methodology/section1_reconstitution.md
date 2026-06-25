# Section 1 — Reconstitution (how to fill)

**Goal:** reverse-decode what the grammar model encodes — the symbols and (when dynamic is
permitted) the grammar rules — purely from the model (blackbox). Library: `model_security/reconstruct.py`.

## Data in / out
- **In:** the GGUF artifact (vocab + merges) and, for the dynamic tier, live Ollama queries.
- **Out:** `section1_reconstitution.md` — decoded symbols (always) + reconstructed grammar (when dynamic ran).

## Two tiers (degrade gracefully by mode)

### STATIC tier — always
- `discover_symbols(vocab)` — **model-only** symbol seeds from the BPE merges (never seeded from
  host grammar/training files).
- `token_inventory` / `decode_tokens` — recon tool tokens carried directly, action fragments,
  digit/operator terminals, and every token decoded (gpt2 raw → human).
- Decoded leaf symbols are shown `⟦meaning⟧ ← alias` (analyst-attributed general knowledge — a
  *display* aid, **not** extracted from the model; the real decode is Section 2 whitebox).

### DYNAMIC tier — only when the mode gate permits (`--dynamic`, generative, static-safe)
- `reconstruct_dynamic(name, seeds)` — per-seed prompt battery (BNF-form + prose) via
  `POST /api/generate`; **output is evidence, never executed**.
- Rules rendered with **leaves decoded inline** (`⟦meaning⟧`), raw alias kept on a `# raw:` line.
- If refused (encoder/unsafe/no opt-in), the section states the reason — symbols still recovered.

## How to read it
- The decoded symbols tell you **what actions the model encodes**; the reconstructed grammar tells
  you **how they compose** into a procedure.
- **The model carries aliases + structure, NOT the literal OS commands** — proven by probing. The
  concrete command strings live in the training/client files → resolved in Section 2 (whitebox).
- Honesty note: reconstruction recall is high on grammar models (97–100% of symbols), but per-rule
  structure can be noisy on tiny models; treat the reconstructed grammar as best-effort evidence.

## Fill checklist
- [ ] mode line present (STATIC / STATIC+DYNAMIC + reason)
- [ ] decoded symbols listed (with ⟦meaning⟧ where known)
- [ ] reconstructed grammar present if dynamic ran (else reason stated)
- [ ] no claim that literal commands were extracted from the model
