#!/usr/bin/env python3
"""examples/test_grammar_tools.py

Self-test for model_tools_grammar.py / class_tools_grammar.py.

Runs each built-in converter against the example source files in
examples/grammar_sources/, validates the outputs, and reports pass/fail.

Usage:
    python examples/test_grammar_tools.py
    python examples/test_grammar_tools.py --verbose
    python examples/test_grammar_tools.py --ai-model qwen2:7b   # also test AI path
"""

import sys, os, argparse, json, tempfile, traceback

# Resolve project root so imports work regardless of cwd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from classes.class_tools_grammar import (
    MermaidGrammarConverter,
    MarkdownGrammarConverter,
    ModelAssistedGrammarConverter,
)

EXAMPLES = os.path.join(ROOT, "examples", "grammar_sources")


# ── helpers ───────────────────────────────────────────────────────────────────

class _Result:
    def __init__(self, name):
        self.name   = name
        self.passed = True
        self.notes  = []

    def check(self, condition, msg_pass, msg_fail):
        if condition:
            self.notes.append("  OK  " + msg_pass)
        else:
            self.notes.append("  FAIL " + msg_fail)
            self.passed = False

    def report(self, verbose=False):
        status = "PASS" if self.passed else "FAIL"
        print("[" + status + "]  " + self.name)
        if verbose or not self.passed:
            for n in self.notes:
                print(n)


def _run_converter(conv, tmp_dir):
    """parse() + generate() → returns (json_path, bnf_path)."""
    conv.parse()
    return conv.generate(tmp_dir)


# ── individual test cases ─────────────────────────────────────────────────────

def test_mermaid(verbose, tmp_dir):
    r = _Result("MermaidGrammarConverter — network_scan.mmd")
    src = os.path.join(EXAMPLES, "network_scan.mmd")
    try:
        conv = MermaidGrammarConverter(src, grammar_name="network_scan")
        json_path, bnf_path = _run_converter(conv, tmp_dir)

        r.check(conv.grammar_name == "network_scan",
                "grammar_name = network_scan",
                "grammar_name mismatch: got " + conv.grammar_name)

        r.check(len(conv._rules) >= 3,
                str(len(conv._rules)) + " rules extracted",
                "too few rules: " + str(len(conv._rules)))

        r.check(len(conv._tokens) >= 4,
                str(len(conv._tokens)) + " tokens extracted",
                "too few tokens: " + str(len(conv._tokens)))

        r.check("discovery" in conv._rules or "discovery" in conv._tokens,
                "'discovery' found in output",
                "'discovery' not found in rules or tokens")

        # Commands must be extracted from %% cmd: annotations
        ping_cmd = conv._tokens.get("ping_sweep", "")
        r.check("nmap -sn" in ping_cmd,
                "'ping_sweep' command extracted: " + repr(ping_cmd[:40]),
                "'ping_sweep' command missing (got: " + repr(ping_cmd[:40]) + ")")

        port_cmd = conv._tokens.get("port_scan", "")
        r.check("nmap" in port_cmd,
                "'port_scan' command extracted: " + repr(port_cmd[:40]),
                "'port_scan' command missing (got: " + repr(port_cmd[:40]) + ")")

        # All leaf tokens should have non-empty commands
        empty = [t for t, v in conv._tokens.items() if not v.strip()]
        r.check(not empty,
                "all leaf tokens have commands",
                "tokens with empty commands: " + str(empty))

        # Validate JSON output
        with open(json_path) as fh:
            data = json.load(fh)
        r.check(data.get("_grammar") == "network_scan",
                "JSON _grammar field correct",
                "JSON _grammar wrong: " + str(data.get("_grammar")))
        r.check(data.get("_type") == "command_vocabulary",
                "JSON _type field correct",
                "JSON _type wrong")
        r.check("nmap -sn" in data.get("ping_sweep", ""),
                "JSON ping_sweep command persisted",
                "JSON ping_sweep missing: " + str(data.get("ping_sweep")))

        # Validate BNF output
        bnf = open(bnf_path).read()
        r.check("<network_scan>" in bnf,
                "BNF contains <network_scan> root rule",
                "BNF missing root rule <network_scan>")
        r.check("::=" in bnf,
                "BNF contains ::= operator",
                "BNF missing ::= operator")

    except Exception:
        r.passed = False
        r.notes.append("  EXCEPTION\n" + traceback.format_exc())

    r.report(verbose)
    return r.passed


def test_markdown_shell(verbose, tmp_dir):
    r = _Result("MarkdownGrammarConverter — disk_maintenance.md (shell)")
    src = os.path.join(EXAMPLES, "disk_maintenance.md")
    try:
        conv = MarkdownGrammarConverter(src, exec_mode="shell")
        json_path, bnf_path = _run_converter(conv, tmp_dir)

        r.check(conv.grammar_name == "disk_maintenance",
                "grammar_name derived from H1",
                "grammar_name mismatch: " + conv.grammar_name)

        r.check("check_usage" in conv._rules,
                "'check_usage' rule found",
                "'check_usage' rule missing")

        r.check("df_human" in conv._tokens,
                "'df_human' token found",
                "'df_human' token missing")

        r.check("df -h" in conv._tokens.get("df_human", ""),
                "'df_human' has correct command value",
                "'df_human' command wrong: " + conv._tokens.get("df_human", ""))

        with open(json_path) as fh:
            data = json.load(fh)
        r.check("_exec" not in data,
                "No _exec key (shell is default)",
                "_exec key unexpectedly present")
        r.check(data.get("df_human") == "df -h",
                "JSON df_human value matches",
                "JSON df_human wrong: " + str(data.get("df_human")))

        bnf = open(bnf_path).read()
        r.check('"df_human"' in bnf,
                'BNF contains "df_human" terminal',
                'BNF missing "df_human" terminal')

    except Exception:
        r.passed = False
        r.notes.append("  EXCEPTION\n" + traceback.format_exc())

    r.report(verbose)
    return r.passed


def test_markdown_python(verbose, tmp_dir):
    r = _Result("MarkdownGrammarConverter — python_sysinfo.md (python exec)")
    src = os.path.join(EXAMPLES, "python_sysinfo.md")
    try:
        conv = MarkdownGrammarConverter(src, exec_mode="python")
        json_path, bnf_path = _run_converter(conv, tmp_dir)

        r.check(conv.grammar_name == "python_sysinfo",
                "grammar_name = python_sysinfo",
                "grammar_name mismatch: " + conv.grammar_name)

        r.check("platform_info" in conv._rules,
                "'platform_info' rule found",
                "'platform_info' rule missing")

        r.check("check_os" in conv._tokens or "check_os" in conv._rules,
                "'check_os' found in output",
                "'check_os' not found")

        # Python code block should be captured as token value
        check_os_cmd = conv._tokens.get("check_os", "")
        r.check("import platform" in check_os_cmd,
                "'check_os' contains Python import",
                "'check_os' missing Python code: " + repr(check_os_cmd[:80]))

        with open(json_path) as fh:
            data = json.load(fh)
        r.check(data.get("_exec") == "python",
                "JSON _exec = python",
                "JSON _exec wrong: " + str(data.get("_exec")))

    except Exception:
        r.passed = False
        r.notes.append("  EXCEPTION\n" + traceback.format_exc())

    r.report(verbose)
    return r.passed


def test_generate_structure(verbose, tmp_dir):
    r = _Result("generate() — correct file naming convention")
    src = os.path.join(EXAMPLES, "disk_maintenance.md")
    try:
        conv = MarkdownGrammarConverter(src)
        conv.parse()
        json_path, bnf_path = conv.generate(tmp_dir)

        expected_json = os.path.join(tmp_dir, "train_disk_maintenance_commands.json")
        expected_bnf  = os.path.join(tmp_dir, "playbook_disk_maintenance.txt")

        r.check(json_path == expected_json,
                "JSON path matches train_<name>_commands.json",
                "JSON path wrong: " + json_path)
        r.check(bnf_path == expected_bnf,
                "BNF path matches playbook_<name>.txt",
                "BNF path wrong: " + bnf_path)
        r.check(os.path.isfile(json_path),
                "JSON file was written",
                "JSON file not found")
        r.check(os.path.isfile(bnf_path),
                "BNF file was written",
                "BNF file not found")

    except Exception:
        r.passed = False
        r.notes.append("  EXCEPTION\n" + traceback.format_exc())

    r.report(verbose)
    return r.passed


def test_ai_assisted(verbose, tmp_dir, model_name, ollama_host):
    r = _Result("ModelAssistedGrammarConverter — disk_maintenance.md via " + model_name)
    src = os.path.join(EXAMPLES, "disk_maintenance.md")
    try:
        conv = ModelAssistedGrammarConverter(
            source      = src,
            grammar_name= None,
            exec_mode   = "shell",
            model_name  = model_name,
            ollama_host = ollama_host,
            max_chars   = 3000,
        )
        conv.parse()

        r.check(bool(conv.grammar_name),
                "grammar_name extracted: " + conv.grammar_name,
                "grammar_name is empty")
        r.check(len(conv._rules) >= 1,
                str(len(conv._rules)) + " rules extracted",
                "no rules extracted")
        r.check(len(conv._tokens) >= 1,
                str(len(conv._tokens)) + " tokens extracted",
                "no tokens extracted")

        json_path, bnf_path = conv.generate(tmp_dir)
        with open(json_path) as fh:
            data = json.load(fh)
        r.check(data.get("_type") == "command_vocabulary",
                "JSON _type field correct",
                "JSON _type wrong: " + str(data.get("_type")))

        bnf = open(bnf_path).read()
        r.check("::=" in bnf,
                "BNF file contains ::= rules",
                "BNF file has no rules")

    except Exception:
        r.passed = False
        r.notes.append("  EXCEPTION\n" + traceback.format_exc())

    r.report(verbose)
    return r.passed


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Self-test for model_tools_grammar.py")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print all check results, not just failures.")
    ap.add_argument("--ai-model", metavar="NAME", default=None,
                    help="Ollama model to use for AI-assisted test (skipped if omitted).")
    ap.add_argument("--ai-host", metavar="URL", default="http://127.0.0.1:11434",
                    help="Ollama host for AI test (default: http://127.0.0.1:11434).")
    args = ap.parse_args()

    print("=" * 60)
    print("model_tools_grammar self-test")
    print("=" * 60)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        results.append(test_mermaid(args.verbose, tmp))
        results.append(test_markdown_shell(args.verbose, tmp))
        results.append(test_markdown_python(args.verbose, tmp))
        results.append(test_generate_structure(args.verbose, tmp))
        if args.ai_model:
            results.append(test_ai_assisted(args.verbose, tmp, args.ai_model, args.ai_host))
        else:
            print("[SKIP]  ModelAssistedGrammarConverter (pass --ai-model <name> to enable)")

    print("=" * 60)
    passed = sum(results)
    total  = len(results)
    status = "ALL PASSED" if passed == total else str(passed) + "/" + str(total) + " PASSED"
    print(status)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
