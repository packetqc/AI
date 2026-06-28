#!/usr/bin/env python3
"""model_runner.py — official entry point for the unified grammar runner.

This is the supported CLI for running and managing a grammar-LM solution. It is a thin
wrapper over the runner class library (``scripts/classes/class_model_runner.py``) — all the
logic lives there; this script only parses arguments into a config dict and dispatches.

Two execution modes that differ by WHERE the CPU logic runs:

  --mode device   the STM32N6 is AUTONOMOUS. The runner is a thin serial terminal: it
                  pushes the prompt over the ST-Link VCP and collects the output; the
                  device's own C++ GrammarRunner tokenizes, parses, evaluates and drives
                  its Neural-ART NPU entirely on-chip. NO ML dependencies — run with the
                  system python.

  --mode host     the HOST runs GrammarRunner locally and queries an Ollama chat model as
                  the grammar oracle, parsing + evaluating on the host. Needs the project
                  venv (the ML chain is imported lazily only for this mode).

Every value below is also settable live from the REPL with ``/set <key> <value>`` and
shown with ``/config``. Builder keys (name/epochs/…) are forwarded to the NPU builder
``model_create_npu_tcn.py`` when you run ``/create`` or ``/export`` from the runner.

Usage:
  python3 scripts/model_runner.py --mode device --port /dev/ttyACM0
  python3 scripts/model_runner.py --mode host   --model model_calculator_test_npu
  python3 scripts/model_runner.py --mode device --name model_calc_tcn_v2 --epochs 800
"""
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# The runner class library is the single source of truth for the runner behaviour.
from classes.class_model_runner import run, prompt_setup


def main(argv=None):
    raw = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(
        prog="model_runner.py",
        description="Official unified grammar runner (autonomous device / host Ollama). "
                    "All flags are also settable live with /set; builder flags are forwarded "
                    "to model_create_npu_tcn.py via /create | /export.",
    )
    ap.add_argument("--mode", choices=["host", "device"], default=None,
                    help="device: autonomous STM32N6 over serial · host: GrammarRunner + Ollama")

    # runtime config (None here = use the library default)
    rt = ap.add_argument_group("runtime")
    rt.add_argument("--grammar", default=None, help="BNF/EBNF playbook grammar file")
    rt.add_argument("--port", default=None, help="device serial port (device mode)")
    rt.add_argument("--baud", type=int, default=None, help="device serial baud (device mode)")
    rt.add_argument("--boot-timeout", type=int, default=None, dest="boot_timeout",
                    help="seconds to wait for the device boot prompt (device mode)")
    rt.add_argument("--model", default=None, help="Ollama model name (host mode)")
    rt.add_argument("--host", default=None, help="Ollama host URL (host mode)")

    # builder config — forwarded to model_create_npu_tcn.py by /create | /export
    bld = ap.add_argument_group("builder (model_create_npu_tcn.py)")
    bld.add_argument("--name", default=None, help="model name (output dir + file stem)")
    bld.add_argument("--version", default=None, help="version tag")
    bld.add_argument("--tokenizer", default=None, help="tokenizer dir to reuse for the vocab")
    bld.add_argument("--embed-dim", type=int, default=None, dest="embed_dim", help="C — conv channels")
    bld.add_argument("--seq-len", type=int, default=None, dest="seq_len", help="L — fixed NPU window")
    bld.add_argument("--kernel", type=int, default=None, help="causal conv kernel size")
    bld.add_argument("--epochs", type=int, default=None, help="training epochs")
    bld.add_argument("--lr", type=float, default=None, help="learning rate")
    bld.add_argument("--out-dir", default=None, dest="out_dir", help="builder output dir override")
    bld.add_argument("--npu-dir", default=None, dest="npu_dir", help="builder NPU export dir override")
    bld.add_argument("--stedgeai", default=None, help="path to the stedgeai CLI")
    bld.add_argument("--target", default=None, help="stedgeai target")
    bld.add_argument("--net-name", default=None, dest="net_name", help="stedgeai network name (FSBL-coupled)")
    bld.add_argument("--c-api", default=None, dest="c_api", help="stedgeai C API")
    bld.add_argument("--opset", type=int, default=None, help="ONNX opset for the exported body")

    # security config — used by /security (model_security_re.py)
    sec = ap.add_argument_group("security (model_security_re.py)")
    sec.add_argument("--sec-gguf", default=None, dest="sec_gguf", help="security target: .gguf artifact path")
    sec.add_argument("--sec-out", default=None, dest="sec_out", help="security report output dir")
    sec.add_argument("--sec-registry", default=None, dest="sec_registry", help="approved-models registry (integrity)")
    sec.add_argument("--sec-assets", default=None, dest="sec_assets", help="assets dir fallback (integrity)")
    sec.add_argument("--sec-dynamic", action="store_true", default=None, dest="sec_dynamic",
                     help="enable dynamic probing by default for /security")

    a = ap.parse_args(raw)
    provided = {k: v for k, v in vars(a).items() if v is not None}
    mode = provided.get("mode")
    # basic essentials per mode — if any is missing from the CLI, prompt for the rest (with listing)
    essentials = {"mode", "grammar", "name"} | ({"model"} if mode == "host" else {"port"})
    if mode is None or not essentials.issubset(provided):
        mode, config = prompt_setup(provided)
        return run(mode, config)
    # all essentials supplied on the CLI -> run directly
    config = {k: v for k, v in provided.items() if k != "mode"}
    return run(mode, config)


if __name__ == "__main__":
    sys.exit(main())
