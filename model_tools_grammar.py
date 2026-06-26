#!/usr/bin/env python3
"""model_tools_grammar.py — Generate grammar + vocabulary files from external sources.

Supported source formats (auto-detected from extension / URL):
    Mermaid flowchart   .mmd / .mermaid / .txt containing 'graph' or 'flowchart'
    Markdown            .md / .markdown
    PDF                 .pdf  (requires: pip install pypdf)
    Web page            http:// or https:// URL

Model-assisted mode (--ai-model):
    Any Ollama model interprets the source and returns structured JSON.
    Recommended for dense technical manuals or inconsistently structured pages.

Output files (written to --output dir, default = current directory):
    models/training/train_<grammar>_commands.json   command vocabulary for model_create_hf_cl.py
    models/grammars/playbook_<grammar>.txt          BNF grammar file

Load into the local AI model:
    python model_create_hf_cl.py \\
        --train models/training/train_<grammar>_commands.json \\
        --grammar models/grammars/playbook_<grammar>.txt
"""

import argparse, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classes.class_tools_grammar import (
    MermaidGrammarConverter,
    MarkdownGrammarConverter,
    TextGrammarConverter,
    PDFGrammarConverter,
    WebGrammarConverter,
    ModelAssistedGrammarConverter,
)


# ── format auto-detection ──────────────────────────────────────────────────────

def _detect_format(source):
    src = source.strip()
    if src.startswith("http://") or src.startswith("https://"):
        return "web"
    ext = os.path.splitext(src)[1].lower()
    if ext in (".pdf",):
        return "pdf"
    if ext in (".md", ".markdown"):
        return "markdown"
    if ext in (".mmd", ".mermaid"):
        return "mermaid"
    # Peek inside plain text / unknown-extension files
    if ext in (".txt", ".rst", ".text", ".spec", ".bnf", ".ebnf", ""):
        try:
            with open(src, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(1024)
            import re
            if re.search(r'^(?:flowchart|graph)\s+[A-Z]{2}', head, re.M | re.I):
                return "mermaid"
            if re.search(r'^#{1,3} ', head, re.M):
                return "markdown"       # has Markdown headings
        except OSError:
            pass
        return "text"                   # plain spec / numbered-section document
    return "markdown"


# ── Ollama model listing ───────────────────────────────────────────────────────

def _list_ollama_models(host):
    import urllib.request, json
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=5) as r:
            data = json.loads(r.read())
        names = [m["name"] for m in data.get("models", [])]
        if names:
            print("Available Ollama models:")
            for n in sorted(names):
                print("  " + n)
        else:
            print("No models found in Ollama at " + host)
    except Exception as exc:
        print("[ERROR] Could not reach Ollama at " + host + ": " + str(exc))
        sys.exit(1)


# ── output directory defaults ──────────────────────────────────────────────────

def _resolve_output_dirs(output_arg):
    """Return (json_dir, bnf_dir): put JSON in models/training/, BNF in models/grammars/."""
    if output_arg and output_arg != ".":
        return output_arg, output_arg
    # If we are inside the AI project directory, use its conventional layout
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir   = os.path.join(script_dir, "models", "training")
    bnf_dir    = os.path.join(script_dir, "models", "grammars")
    return json_dir, bnf_dir


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "source", nargs="?",
        help="Source file path or URL to convert.",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["mermaid", "markdown", "text", "pdf", "web", "auto"],
        default="auto",
        help="Source format (default: auto-detect from extension/URL).",
    )
    parser.add_argument(
        "--grammar", "-g", metavar="NAME", default=None,
        help="Grammar name to use (default: derived from source filename or H1 heading).",
    )
    parser.add_argument(
        "--exec", "-e",
        choices=["shell", "python"], default="shell",
        help="Execution mode for generated command tokens (default: shell).",
    )
    parser.add_argument(
        "--output", "-o", metavar="DIR", default=None,
        help="Output directory (default: training/ for JSON, grammars/ for BNF).",
    )
    parser.add_argument(
        "--ai-model", "-M", metavar="MODEL_NAME", default=None,
        help="Use this Ollama model to extract grammar via AI (e.g. qwen2:7b, llama3).",
    )
    parser.add_argument(
        "--ai-host", metavar="URL", default="http://127.0.0.1:11434",
        help="Ollama host for AI-assisted extraction (default: http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "--ai-max-chars", metavar="N", type=int, default=6000,
        help="Maximum source characters sent to the AI model (default: 6000).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print generated content without writing any files.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available Ollama models and exit.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print a short summary of the parsed structure.",
    )

    args = parser.parse_args()

    # ── list models and exit ─────────────────────────────────────────────────
    if args.list_models:
        _list_ollama_models(args.ai_host)
        return

    if not args.source:
        parser.print_help()
        sys.exit(1)

    # ── resolve format ───────────────────────────────────────────────────────
    fmt = args.format
    if fmt == "auto":
        fmt = _detect_format(args.source)

    # ── build converter ──────────────────────────────────────────────────────
    if args.ai_model:
        conv = ModelAssistedGrammarConverter(
            source      = args.source,
            grammar_name= args.grammar,
            exec_mode   = args.exec,
            model_name  = args.ai_model,
            ollama_host = args.ai_host,
            max_chars   = args.ai_max_chars,
        )
        print("[AI]  Using model '" + args.ai_model + "' to extract grammar from " + args.source)
    else:
        _map = {
            "mermaid":  MermaidGrammarConverter,
            "markdown": MarkdownGrammarConverter,
            "text":     TextGrammarConverter,
            "pdf":      PDFGrammarConverter,
            "web":      WebGrammarConverter,
        }
        cls  = _map[fmt]
        conv = cls(source=args.source, grammar_name=args.grammar, exec_mode=args.exec)
        print("[" + fmt.upper() + "]  Parsing " + args.source)

    # ── parse ────────────────────────────────────────────────────────────────
    try:
        conv.parse()
    except Exception as exc:
        print("[ERROR] Parse failed: " + str(exc))
        sys.exit(1)

    if not conv._rules and not conv._tokens:
        print("[WARNING] Nothing extracted — check the source structure.")
        sys.exit(1)

    if args.summary:
        print()
        print(conv.summary())
        print()

    # ── output ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print()
        print("=== Vocabulary JSON ===")
        import json
        print(json.dumps(conv.to_vocabulary_dict(), indent=2))
        print()
        print("=== Grammar BNF ===")
        print(conv.to_grammar_text())
        return

    json_dir, bnf_dir = _resolve_output_dirs(args.output)
    # generate() writes to one dir; we need two separate dirs → call individually
    import json as _json, os as _os
    _os.makedirs(json_dir, exist_ok=True)
    _os.makedirs(bnf_dir,  exist_ok=True)

    json_path = _os.path.join(json_dir, "train_" + conv.grammar_name + "_commands.json")
    bnf_path  = _os.path.join(bnf_dir,  "playbook_" + conv.grammar_name + ".txt")

    with open(json_path, "w", encoding="utf-8") as fh:
        _json.dump(conv.to_vocabulary_dict(), fh, indent=2, ensure_ascii=False)
    with open(bnf_path, "w", encoding="utf-8") as fh:
        fh.write(conv.to_grammar_text())

    print()
    print("Generated:")
    print("  Vocabulary  " + json_path)
    print("  Grammar     " + bnf_path)
    print()
    print("Load into the model:")
    print("  python model_create_hf_cl.py \\")
    print("      --train "   + json_path + " \\")
    print("      --grammar " + bnf_path)


if __name__ == "__main__":
    main()
