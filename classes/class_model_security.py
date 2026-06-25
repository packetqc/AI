"""
class_model_security.py — DEPRECATED shim.

The analyst toolkit moved to the `model_security/` package (3 distinct section
libraries + shared base). This module re-exports the common names for backward
compatibility. New code should import from `model_security.*`.
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_security.acquire import (  # noqa: F401,E402
    gpt2_decode, shannon_entropy, sha256_file, OllamaArtifact, Issue,
    GgufStaticAnalyzer, classify_model_type, static_safety_ok, resolve_mode,
)
from model_security.reconstruct import (  # noqa: F401,E402
    discover_symbols, token_inventory, decode_tokens, render_tokens_decoded,
    RECON_LEXICON, RECON_TOOLS,
)
from model_security.threat import (  # noqa: F401,E402
    threat_scan, threat_verdict, binary_forensics,
)
