import json
import os
import re
import sys
import time
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from gguf import GGUFWriter

from transformers import Qwen2Config, Qwen2TokenizerFast, Qwen2ForCausalLM
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, Regex

#################################################################################################
# MP CLASS AND CODE
#################################################################################################
# scripts/ on sys.path so `from classes...` resolves (classes at scripts/classes/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes.class_terminal_logs import TerminalLogger
logger = TerminalLogger()

from classes.class_model_assets import ModelAssets
from classes.class_model_grammar import ModelGrammar, GrammarRunner

# ── CLI arguments ─────────────────────────────────────────────────────────────
# Prefix a filename with @ to expand it as a line-per-argument list, e.g.:
#   python model_create_hf_cl.py @my_startup_files.txt
#
# Examples:
#   python model_create_hf_cl.py --train training/notes.md training/extra.json
#   python model_create_hf_cl.py --grammar grammars/mygrammar.bnf
#   python model_create_hf_cl.py --train @training/list.txt --grammar @grammars/list.txt
import argparse as _argparse
_parser = _argparse.ArgumentParser(
    description="Tiny Qwen2 model with BNF grammar augmentation, served via Ollama.",
    fromfile_prefix_chars="@",   # @file expands one argument per line
)
_parser.add_argument(
    "--train", "-t", nargs="+", metavar="FILE", default=[],
    help="Knowledge / training files (markdown, JSON) — loaded before grammars and defaults.",
)
_parser.add_argument(
    "--grammar", "-g", nargs="+", metavar="FILE", default=[],
    help="BNF/EBNF grammar files — loaded after --train files and before built-in defaults.",
)
_parser.add_argument(
    "--model", "-m", metavar="NAME_OR_FILE", default=None,
    help="Skip training: use an existing Ollama model by name, or import a local .gguf file.",
)
_args = _parser.parse_args()
# Ordered: user training → user grammars → built-in defaults
_CLI_FILES = list(_args.train) + list(_args.grammar)

#################################################################################################
#
#################################################################################################
version = "1"
model_create = "model_discoverit_version_" + version

# Each model is self-contained in its own folder under models/: HF weights +
# <name>.gguf + <name>.state.json + Modelfile all live inside models/<name>/.
model_path = os.path.join("models", "generated", "transformer", model_create)   # transformer family
os.makedirs(model_path, exist_ok=True)                   # create empty folder if missing
gguf_path = os.path.join(model_path, model_create + ".gguf")

modelfile_path = os.path.join(model_path, "Modelfile")

NAME = "model_to_discover"
OLLAMA_MODEL_NAME = NAME

# Knowledge files trained into the model at start-up, IN ORDER (markdown / plain JSON / grammar
# JSON; empty list to skip). Each is routed by extension exactly like the in-flight "/read".
# NOTE: only used on a fresh run — once a state file exists it is restored instead (see below).
# INIT_KNOWLEDGE_FILES = [
#     "models/training/train_kali_discovery_commands.json",  # command vocabulary loaded BEFORE grammar
#     "models/grammars/playbook_kali_discovery.txt",
# ]
# INIT_KNOWLEDGE_FILES = [
#     "models/grammars/playbook_model_calculator.txt",
# ]
INIT_KNOWLEDGE_FILES = [
    "models/training/train_python_healthcheck_commands.json",
    "models/grammars/playbook_pyhealthcheck.txt",
]

# Persistence: accumulated knowledge + the (possibly adapted) config are saved here so that a
# later run of this script restores everything learned in previous sessions.
STATE_PATH = os.path.join(model_path, model_create + ".state.json")

# Ollama serving parameters (shared by the first build and every /read rebuild).
# NOTE: plain string so the Ollama "{{ .Prompt }}" double-braces survive; the template
# reproduces EXACTLY the training prefix ("<|endoftext|>" + prompt, no trailing space).
MODELFILE_TEMPLATE = "<|endoftext|>{{ .Prompt }}"
NUM_PREDICT = 64   # grammar answers can be long (digit rule: 10 alternatives ~50 tokens)


os.makedirs("./"+model_path, exist_ok=True)

ok = True

# Enforce clean environment routing to prevent network proxy delays
os.environ["MEM0_TELEMETRY"] = "false"
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

# ── External model mode (--model / -m) ────────────────────────────────────────
# When --model is supplied, skip training and use an existing Ollama model or
# import a local .gguf file.  Grammar / vocabulary files still load normally.
_EXTERNAL_MODEL = False
if _args.model:
    import urllib.request as _urlreq, json as _ejson, subprocess as _esproc

    _ext_arg = _args.model.strip()
    _ollama_host = os.environ["OLLAMA_HOST"]

    def _ollama_model_names():
        """Return (bare_names, full_names) lists from Ollama /api/tags."""
        try:
            with _urlreq.urlopen(_ollama_host + "/api/tags", timeout=5) as _r:
                _data = _ejson.loads(_r.read())
            _full = [m["name"] for m in _data.get("models", [])]
            _bare = [n.split(":")[0] for n in _full]
            return _bare, _full
        except Exception:
            return [], []

    _bare_names, _full_names = _ollama_model_names()

    if _ext_arg in _full_names or _ext_arg in _bare_names:
        # Model already registered in Ollama — use it directly.
        NAME = _ext_arg
        _EXTERNAL_MODEL = True
        logger.log("ok", "MODEL", "External model '" + NAME + "' found in Ollama — skipping training.")

    elif os.path.isfile(_ext_arg) and _ext_arg.lower().endswith(".gguf"):
        # Local GGUF file — import into Ollama then use.
        NAME = os.path.splitext(os.path.basename(_ext_arg))[0]
        _mf_content = "FROM " + os.path.abspath(_ext_arg) + "\n"
        _mf_tmp = "./_ext_modelfile"
        with open(_mf_tmp, "w") as _mf_fh:
            _mf_fh.write(_mf_content)
        logger.log("info", "MODEL",
                   "Importing '" + _ext_arg + "' into Ollama as '" + NAME + "'...")
        _imp = _esproc.run(["ollama", "create", NAME, "-f", _mf_tmp], text=True)
        if _imp.returncode != 0:
            logger.log("error", "MODEL", "ollama create failed — check the GGUF file.")
            sys.exit(1)
        _EXTERNAL_MODEL = True
        logger.log("ok", "MODEL", "Imported '" + NAME + "' into Ollama successfully.")

    else:
        # Model not in Ollama and not a local file.
        logger.log("error", "MODEL",
                   "'" + _ext_arg + "' not found in Ollama and not a .gguf file.")
        if _bare_names:
            logger.log("info", "MODEL", "Available in Ollama: " + ", ".join(sorted(_bare_names)))
        sys.exit(1)

#################################################################################################
#
#################################################################################################
VOCAB_SIZE = 1027
HIDDEN_SIZE = 256   #512   #256   #128   #64
INTERMEDIATE_SIZE = 512 #1024    #512 #256 #128
NUM_LAYERS = 2  #12 #4  #2  (>=2 gives headroom for memorizing knowledge alongside the anchors)
NUM_HEADS = 4   #8   #4
NUM_KV_HEADS = 2    #4    #2
CONTEXT_LENGTH = 256

# Auto-detect the best available compute device: CUDA (NVIDIA) > MPS (Apple) > CPU.
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

#################################################################################################
# OLLAMA FUNCTIONS
#################################################################################################
import json
import urllib.request
import urllib.error

def get_ollama_answer(prompt: str, model_name: str = "llama3", host_url: str = "http://localhost:11434") -> str:
    """Sends a prompt string to an Ollama host and returns the completed text answer."""
    endpoint = f"{host_url}/api/generate"
    
    # 1. Prepare the JSON payload payload configurations
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # Returns the full answer at once instead of word-by-word
    }
    
    # 2. Encode payload to bytes for the HTTP POST request
    data = json.dumps(payload).encode("utf-8")
    
    # 3. Configure the HTTP Request header properties
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        # 4. Fire the network request to the Ollama server
        with urllib.request.urlopen(req, timeout=60) as response:
            html_bytes = response.read()
            json_response = json.loads(html_bytes.decode("utf-8"))
            
            # Extract and return the final text field
            return json_response.get("response", "")
            
    except urllib.error.URLError as e:
        return f"[Connection Error]: Cannot reach Ollama host at {host_url}. Is it running? ({e.reason})"
    except Exception as e:
        return f"[Error]: An unexpected issues occurred: {str(e)}"

#################################################################################################
# KNOWLEDGE-AWARE PIPELINE  (tokenizer + model, rebuilt whenever new markdown is learned)
#################################################################################################
# The canonical prompt/answer pairs the model must ALWAYS know (replayed on every rebuild).
SPECIAL_TOKEN = "<|endoftext|>"
PAIRS = [
    ("what color is always good ?", "green"),
    ("what color is always bad ?",  "red"),
]

def format_example(prompt: str, answer: str) -> str:
    """Training form of an anchor pair. The Modelfile TEMPLATE reproduces the prefix
    ("<|endoftext|>" + prompt); the model then emits the " answer" token + EOS."""
    return f"{SPECIAL_TOKEN}{prompt} {answer}{SPECIAL_TOKEN}"

# CRITICAL: the exact Qwen2 pre-tokenizer regex. We also export pre="qwen2" so that HF
# (training) and llama.cpp/Ollama (inference) split text identically — notably " ?" stays a
# single "Ġ?" token instead of splitting into "Ġ"+"?", which would shift the final token.
QWEN2_PRETOKENIZER_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

DEFAULT_VOCAB_CAP = 4096     # starting BPE vocabulary ceiling
MAX_VOCAB_CAP = 32768        # hard upper bound so a huge file can't blow up the embedding matrix
BLOCK_SIZE = 64              # sliding-window length for knowledge text (<= CONTEXT_LENGTH)
MAX_KNOWLEDGE_TOKENS = 2048  # per-document safety cap so a huge file can't stall the toy
MAX_EPOCHS = 800
TARGET_LOSS = 5e-4

# Default architecture dims (persisted in the state file; an old state can override these).
ARCH_DEFAULTS = {
    "hidden_size": HIDDEN_SIZE,
    "intermediate_size": INTERMEDIATE_SIZE,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": NUM_HEADS,
    "num_key_value_heads": NUM_KV_HEADS,
    "max_position_embeddings": CONTEXT_LENGTH,
}


def required_vocab_cap(corpus_texts, floor):
    """How big the BPE vocabulary should be to give the corpus' words first-class tokens.

    The byte alphabet (256) + special token + roughly a few sub-word tokens per distinct word.
    This is what lets the pipeline DETECT that a freshly-read file needs more vocabulary than the
    current ceiling and grow it (bounded by MAX_VOCAB_CAP)."""
    words = set()
    for t in corpus_texts:
        words.update(t.split())
    needed = 256 + 16 + 4 * len(words)
    return min(MAX_VOCAB_CAP, max(floor, needed))
# Mask this many leading tokens of each knowledge window from the loss. Documents that share a
# prefix (e.g. "The ...") force a contradictory next-token target right after the shared word,
# which floors the loss and corrupts recall. Masking those first couple of predictions removes
# the conflict; the user still supplies them as prompt context at inference. Keep this SMALL so
# the answer in a short fact ("The CEO is Maria") is not itself masked out of training.
MASK_LEAD = 2


def markdown_to_text(md: str) -> str:
    """Strip markdown syntax down to plain prose so structural tokens (#, *, `, links) don't
    create spurious training targets and the model learns the actual content."""
    text = re.sub(r"```.*?```", " ", md, flags=re.DOTALL)          # fenced code blocks
    text = re.sub(r"`([^`]*)`", r"\1", text)                       # inline code
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)         # links / images -> link text
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)   # ATX headers
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.MULTILINE)    # bullet markers
    text = re.sub(r"[*_~>]", "", text)                             # emphasis / blockquote marks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_segments(text: str):
    """Split prose into independent sentence/line segments. Each becomes its own training
    sequence so a natural prompt (a prefix of one sentence) matches what the model saw —
    without an unrelated header or neighbouring sentence prefixed in front of it."""
    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.strip()
            if part:
                segments.append(part)
    return segments


def build_pipeline(knowledge_texts, grammar_pairs=None, arch=None, vocab_cap=None):
    """Build a tokenizer + model that cover the green/red prompts, every grammar routine AND
    every knowledge text, then train them JOINTLY (anchors + grammar routines masked to the
    answer; prose knowledge as causal-LM windows).

    This runs at start-up and on every /read, so any newly-read word becomes a first-class
    token (not a pile of byte fragments) — which is what makes in-flight recall robust.
    Retraining on the full set each time also makes catastrophic forgetting impossible.

    ``grammar_pairs`` is a list of (prompt, answer) routines (e.g. ("healthcheck", "df -h, w"))
    trained exactly like the anchors. ``arch`` overrides the architecture dims (from a persisted
    state); ``vocab_cap`` is the current ceiling, grown automatically if the accumulated content
    needs more whole-word tokens. Returns an artifacts dict (incl. effective ``vocab_cap``/``arch``).
    """
    arch = dict(ARCH_DEFAULTS, **(arch or {}))

    # Anchors = the always-on green/red pairs PLUS every learned grammar routine. Both are
    # trained as prompt->answer pairs (prompt masked, answer + EOS trained).
    anchor_pairs = list(PAIRS) + [tuple(p) for p in (grammar_pairs or [])]

    # Strip markdown to plain prose before anything else (tokenizer + training use the same text).
    knowledge_texts = [markdown_to_text(t) for t in (knowledge_texts or []) if t and t.strip()]
    knowledge_texts = [t for t in knowledge_texts if t]

    # ---- corpus: anchor prompts/answers (repeated) + every knowledge document ----
    corpus = []
    for prompt, answer in anchor_pairs:
        corpus.extend([prompt, answer, format_example(prompt, answer)] * 20)
    corpus.extend(knowledge_texts)

    # ADAPT: grow the vocabulary ceiling if the corpus needs more than the current cap.
    eff_vocab_cap = required_vocab_cap(corpus, vocab_cap or DEFAULT_VOCAB_CAP)
    logger.log("info", "SYSTEM", "Building tokenizer + model over " + str(len(anchor_pairs))
               + " anchor/grammar pair(s) + " + str(len(knowledge_texts))
               + " knowledge doc(s); vocab_cap=" + str(eff_vocab_cap) + "...")

    # ---- tokenizer: byte-level BPE with the Qwen2 pre-tokenizer ----
    raw_tokenizer = Tokenizer(models.BPE(unk_token=None))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(QWEN2_PRETOKENIZER_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    raw_tokenizer.decoder = decoders.ByteLevel()
    raw_tokenizer.train_from_iterator(corpus, trainer=trainers.BpeTrainer(
        vocab_size=eff_vocab_cap,
        special_tokens=[SPECIAL_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    ))
    tokenizer = Qwen2TokenizerFast(tokenizer_object=raw_tokenizer)
    tokenizer.add_special_tokens({
        "bos_token": SPECIAL_TOKEN, "eos_token": SPECIAL_TOKEN, "pad_token": SPECIAL_TOKEN,
    })
    tokenizer.save_pretrained("./" + model_path)

    tokens = [t for t, _ in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])]
    _state = json.loads(raw_tokenizer.to_str())
    merges = [m if isinstance(m, str) else " ".join(m) for m in _state["model"]["merges"]]
    eos_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN)
    scores = [0.0] * len(tokens)
    token_types = [1] * len(tokens)
    token_types[eos_id] = 3   # control token
    logger.log("ok", "SYSTEM", "Vocabulary: " + str(len(tokens)) + " tokens, "
               + str(len(merges)) + " merges.")

    # ---- config + fresh model (architecture dims come from `arch`) ----
    config = Qwen2Config(
        vocab_size=len(tokens),
        hidden_size=arch["hidden_size"], intermediate_size=arch["intermediate_size"],
        num_hidden_layers=arch["num_hidden_layers"], num_attention_heads=arch["num_attention_heads"],
        num_key_value_heads=arch["num_key_value_heads"], max_position_embeddings=arch["max_position_embeddings"],
        architectures=["Qwen2ForCausalLM"], model_type="qwen2",
    )
    config.attention_bias = True
    config.save_pretrained("./" + model_path)
    model = Qwen2ForCausalLM(config).to(DEVICE)
    model.resize_token_embeddings(len(tokens), mean_resizing=False)

    # ---- training rows: anchors + grammar routines (mask everything up to the answer) ----
    rows = []  # list of (input_ids, labels)
    for prompt, answer in anchor_pairs:
        prefix = tokenizer.encode(f"{SPECIAL_TOKEN}{prompt}")    # exactly what the template feeds
        full = tokenizer.encode(format_example(prompt, answer))  # prompt + " answer" + EOS
        lab = [-100] * len(full)
        for j in range(len(prefix), len(full)):                  # answer tokens + closing EOS only
            lab[j] = full[j]
        rows.extend([(full, lab)] * 5)

    # ---- training rows: knowledge as causal-LM windows, BOS-prepended + EOS-terminated ----
    # The leading BOS matches what the Modelfile TEMPLATE feeds at inference; the trailing EOS
    # teaches the model to STOP after reproducing the content (otherwise it rambles). We mask the
    # BOS + MASK_LEAD shared leading tokens so prefix collisions between documents don't floor the
    # loss (see MASK_LEAD note above).
    for text in knowledge_texts:
        for segment in split_segments(text):
            ids = tokenizer.encode(segment)[:MAX_KNOWLEDGE_TOKENS] + [eos_id]
            for i in range(0, len(ids), BLOCK_SIZE):
                win = ids[i:i + BLOCK_SIZE]
                if len(win) < 2:
                    continue
                seq = [eos_id] + win
                lab = list(seq)
                for k in range(min(MASK_LEAD + 1, len(lab) - 1)):
                    lab[k] = -100
                rows.append((seq, lab))

    # ---- pad to a common width and train jointly until converged ----
    width = max(len(seq) for seq, _ in rows)
    input_rows, label_rows, mask_rows = [], [], []
    for seq, lab in rows:
        pad = width - len(seq)
        input_rows.append(seq + [eos_id] * pad)
        label_rows.append(lab + [-100] * pad)
        mask_rows.append([1] * len(seq) + [0] * pad)

    input_ids = torch.tensor(input_rows, dtype=torch.long).to(DEVICE)
    labels = torch.tensor(label_rows, dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    logger.log("info", "SYSTEM", "Training " + str(len(rows)) + " rows (anchors + knowledge) jointly"
               + " on " + str(DEVICE) + "...")
    loss, epoch = None, 0
    for epoch in range(MAX_EPOCHS):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = out.loss
        loss.backward()
        optimizer.step()
        if loss.item() < TARGET_LOSS:
            break
    model.eval()
    logger.log("ok", "SYSTEM", "Converged at epoch " + str(epoch + 1)
               + ", final loss = " + f"{loss.item():.6f}")

    # Save trained weights alongside config + tokenizer so external tools (e.g.
    # model_export_npu.py) can load them without re-running the full pipeline.
    model.save_pretrained("./" + model_path)

    return {
        "model": model, "tokenizer": tokenizer, "config": config,
        "tokens": tokens, "scores": scores, "token_types": token_types,
        "merges": merges, "eos_id": eos_id,
        "vocab_cap": eff_vocab_cap, "arch": arch,
    }


# DO NOT add any manual logit overrides or zeroing below this line!

#################################################################################################
#
#################################################################################################
# logger.log("info", "SYSTEM", "5. Saving optimized weights matrix directly...")
# model.save_pretrained(model_path, safe_serialization=False)

#################################################################################################
# KNOWLEDGE ASSETS + GGUF EXPORT + OLLAMA BUILD
#################################################################################################
# ModelAssets accumulates the markdown documents and, via build_pipeline, rebuilds the tokenizer
# + model whenever new knowledge arrives — so the start-up build and the in-flight "/read"
# command share EXACTLY the same robust code path. Config + knowledge persist to STATE_PATH.

# Grammar command vocabularies: grammar_name -> {token: shell_cmd}
# Populated at startup by seed_from_file when a command vocabulary JSON is found,
# or in-flight by /read. Passed to GrammarRunner so execute-mode grammars can
# run OS commands when they reach a matching terminal token.
_grammar_commands: dict = {}


def _reload_command_vocabularies():
    """Re-populate _grammar_commands from any command-vocabulary JSON in INIT_KNOWLEDGE_FILES.

    Called on every startup (fresh and restored) because command vocabularies are
    runtime config — they are intentionally NOT saved in the state file.
    """
    for _path in INIT_KNOWLEDGE_FILES:
        if not _path.lower().endswith(".json") or not os.path.isfile(_path):
            continue
        try:
            with open(_path, "r", encoding="utf-8") as _f:
                _obj = json.load(_f)
        except Exception:
            continue
        if isinstance(_obj, dict) and _obj.get("_type") == "command_vocabulary":
            _gn = _obj.get("_grammar", "unknown")
            _cmds = {k: v for k, v in _obj.items()
                     if not k.startswith("_") and isinstance(v, str)}
            if "_exec" in _obj:                      # preserve execution mode (shell / python)
                _cmds["_exec"] = _obj["_exec"]
            _grammar_commands[_gn] = _cmds
            logger.log("info", "SYSTEM",
                       "Loaded " + str(len(_cmds)) + " command(s) for '"
                       + _gn + "' grammar from " + _path
                       + " (exec=" + _obj.get("_exec", "shell") + ")")


def seed_from_file(path, knowledge_texts, playbook):
    """Read one start-up knowledge file and route it by extension: a grammar/playbook JSON is
    deep-merged into the ``playbook`` tree; plain JSON is flattened to prose; markdown/other is
    raw prose appended to ``knowledge_texts``. Returns the grammar name if the file had one."""
    if not (path and os.path.isfile(path)):
        if path:
            logger.log("warning", "SYSTEM", "Start-up knowledge file not found, skipping: " + path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if path.lower().endswith(".json"):
        try:
            obj = json.loads(raw)
        except Exception as e:
            logger.log("error", "SYSTEM", "Invalid JSON in " + path + ": " + str(e))
            return None
        # Command vocabulary JSON: {"_type": "command_vocabulary", "_grammar": "name", token: cmd}
        if isinstance(obj, dict) and obj.get("_type") == "command_vocabulary":
            gname = obj.get("_grammar", "unknown")
            cmds = {k: v for k, v in obj.items()
                    if not k.startswith("_") and isinstance(v, str)}
            if "_exec" in obj:
                cmds["_exec"] = obj["_exec"]
            _grammar_commands[gname] = cmds
            logger.log("info", "SYSTEM",
                       "Loaded " + str(len(cmds)) + " command(s) for '"
                       + gname + "' grammar from " + path
                       + " (exec=" + obj.get("_exec", "shell") + ")")
            return None  # no model training needed for the commands dict itself
        tree = ModelAssets.playbook_tree_from_json(obj)
        if tree is not None:                             # grammar/playbook JSON -> merge tree
            ModelAssets.merge_tree(playbook, tree)
            logger.log("info", "SYSTEM", "Seeded playbook (" + str(len(tree))
                       + " root(s)) from " + path)
            return obj.get("grammar") if isinstance(obj, dict) else None
        raw = ModelAssets.json_to_text(obj)              # plain JSON -> flattened facts
    else:
        # Try BNF/EBNF grammar detection before treating as raw prose.
        grammar_result = ModelGrammar.load_file(path)
        if grammar_result is not None:
            ModelAssets.merge_tree(playbook, grammar_result["tree"])
            if grammar_result["prose"].strip():
                knowledge_texts.append(grammar_result["prose"])
            logger.log("info", "SYSTEM",
                       "Seeded grammar '" + grammar_result["name"] + "' from " + path
                       + " (" + str(len(grammar_result["rules"])) + " rule(s)).")
            return grammar_result["name"]
    if raw.strip():
        knowledge_texts.append(raw)
        logger.log("info", "SYSTEM", "Seeded start-up knowledge from " + path)
    return None


# Restore previous-session state if present; otherwise seed from the start-up knowledge files.
# Load order (fresh start): --train files → --grammar files → INIT_KNOWLEDGE_FILES defaults.
# Load order (restored):     restore state → apply any --train/--grammar additions on top.
_state = ModelAssets.load_state(STATE_PATH)
if _state:
    knowledge_texts = list(_state.get("knowledge_texts", []))
    playbook = dict(_state.get("playbook", {}))
    grammar_name = _state.get("grammar_name")
    if not playbook and _state.get("grammar_pairs"):     # legacy state -> flat tree of leaves
        playbook = {str(p[0]): str(p[1]) for p in _state["grammar_pairs"]}
    arch = dict(ARCH_DEFAULTS, **_state.get("arch", {}))
    vocab_cap = _state.get("vocab_cap", DEFAULT_VOCAB_CAP)
    logger.log("ok", "SYSTEM", "Restored state from " + STATE_PATH + ": "
               + str(len(knowledge_texts)) + " prose doc(s), " + str(len(playbook))
               + " playbook root(s), vocab_cap=" + str(vocab_cap) + ".")
    # Apply any CLI files on top of the restored state.
    for _path in _CLI_FILES:
        _name = seed_from_file(_path, knowledge_texts, playbook)
        if _name:
            grammar_name = _name
else:
    knowledge_texts = []
    playbook = {}
    grammar_name = None
    arch = dict(ARCH_DEFAULTS)
    vocab_cap = DEFAULT_VOCAB_CAP
    for _path in _CLI_FILES + list(INIT_KNOWLEDGE_FILES):
        _name = seed_from_file(_path, knowledge_texts, playbook)
        if _name:
            grammar_name = _name

# Always reload command vocabularies — they are runtime config, not saved in state.
_reload_command_vocabularies()

if not _EXTERNAL_MODEL:
    # Flatten the playbook tree into anchor pairs for training.
    grammar_pairs = ModelAssets.flatten_playbook(playbook, grammar_name)

    # Build the tokenizer + model over prompts + grammar pairs + knowledge.
    artifacts = build_pipeline(knowledge_texts, grammar_pairs, arch, vocab_cap)

    assets = ModelAssets(
        builder=build_pipeline,
        knowledge_texts=knowledge_texts,
        playbook=playbook,
        grammar_name=grammar_name,
        artifacts=artifacts,
        arch=artifacts["arch"],
        vocab_cap=artifacts["vocab_cap"],
        gguf_path=gguf_path,
        modelfile_path=modelfile_path,
        ollama_name=NAME,
        ollama_host=os.environ["OLLAMA_HOST"],
        modelfile_template=MODELFILE_TEMPLATE,
        num_predict=NUM_PREDICT,
        state_path=STATE_PATH,
        logger=logger,
    )
    assets.save_state()
    assets.export_and_rebuild()
    assets.inspect_gguf()
else:
    # External model mode: no training, no GGUF export.
    # Build a minimal assets stub that supports grammar detection and Tab completion.
    import types as _types
    grammar_pairs = []
    artifacts     = {}
    assets = _types.SimpleNamespace(
        playbook         = playbook,
        grammar_name     = grammar_name,
        knowledge_texts  = knowledge_texts,
    )
    # In-flight markdown learning requires a built model — not available in external mode.
    assets.learn_document = lambda *a, **kw: False

# =========================================================================
# AUTOMATED SELF-TEST: verify the two canonical prompts answer green / red
# =========================================================================
if not _EXTERNAL_MODEL:
    # Catches any residual train/inference tokenization drift.
    logger.log("info", "SYSTEM", "Running self-test against the two canonical prompts...")
    time.sleep(1)  # let Ollama finish registering the freshly-built model
    all_passed = True

    def _first_alt(answer):
        for sep in (",", "|"):
            if sep in answer:
                return answer.split(sep)[0].strip()
        return answer.strip()

    selftest_cases = [(p, a, a) for p, a in PAIRS]
    selftest_cases += [(p, a, _first_alt(a)) for p, a in grammar_pairs]
    for prompt, expected, needle in selftest_cases:
        response = get_ollama_answer(prompt=prompt, model_name=NAME, host_url=os.environ["OLLAMA_HOST"])
        passed = needle.lower() in response.lower()
        all_passed = all_passed and passed
        logger.log(
            "ok" if passed else "error",
            "SELFTEST",
            ("PASS" if passed else "FAIL") + " | '" + prompt + "' -> expected '" + expected + "', got '" + response.strip() + "'",
        )
    if all_passed:
        logger.log("ok", "SELFTEST", "✅ All prompts answered correctly.")
    else:
        logger.log("error", "SELFTEST", "❌ Self-test failed — check tokenizer/template parity.")
else:
    logger.log("info", "SYSTEM", "External model mode — self-test skipped.")

# =========================================================================
# AUTOMATICATED INTERACTIVE RUNNER
# =========================================================================
import os
import sys
import time


# Initialize your native keyboard history driver buffer layers
try:
    import gnureadline as readline
except ImportError:
    import readline

readline.parse_and_bind("tab: complete")
readline.set_completer_delims(" \t\n")
readline.clear_history()

_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cli_history")
try:
    readline.read_history_file(_HISTORY_FILE)
    readline.set_history_length(500)
except FileNotFoundError:
    pass

import atexit as _atexit
_atexit.register(readline.write_history_file, _HISTORY_FILE)

# ── Dynamic tab completion ─────────────────────────────────────────────────────
# Multi-level: each Tab press descends one deeper into the grammar tree.
#
# Completion levels (left to right on the line):
#   word 1            → grammar names + CLI commands
#   word 2 (or Tab    → expansion of the root rule (first rule in subtree)
#            after 1)
#   word N (or Tab    → expansion of word N-1 as a grammar rule
#            after N-1)
#
# If the playbook has no match for a partial, falls back to a live Ollama query.
_CLI_CMDS = ["/?", "/read", "/grammar", "/run", "/npu", "/context", "/tokens", "/bye"]

def _grammar_expand(rule_name, subtree):
    """Return next-level token names from the body of one grammar rule.

    Handles both sequence bodies ('a b c') and alternative bodies ('a | b | c').
    Quoted terminals like '"py_check_kernel"' are stripped to 'py_check_kernel'.
    """
    body = subtree.get(rule_name, "")
    if not body:
        return []
    candidates = set()
    for alt in str(body).split("|"):
        for tok in alt.split():
            tok = tok.strip()
            if tok.startswith('"') and tok.endswith('"'):
                inner = tok[1:-1].strip()
                if inner:
                    candidates.add(inner)
            elif tok:
                candidates.add(tok)
    return sorted(candidates)

def _tab_completer(text, state):
    try:
        line     = readline.get_line_buffer()
        stripped = line.lstrip()
        parts    = stripped.split()
        trailing = line.endswith(" ")

        # ── CLI command completion (/r<TAB> → /read /run) ─────────────────────
        if stripped.startswith("/"):
            first = parts[0] if parts else ""
            if len(parts) <= 1 and not trailing:
                matches = [c for c in _CLI_CMDS if c.startswith(first)]
            elif trailing and parts[0] in ("/run", "/tokens"):
                matches = sorted(g for g in assets.playbook
                                 if isinstance(assets.playbook[g], dict) and g.startswith(text))
            else:
                matches = []
            return matches[state] if state < len(matches) else None

        # ── First word: grammar names + CLI commands ───────────────────────────
        if not parts or (len(parts) == 1 and not trailing):
            candidates = sorted(set(
                [g for g in assets.playbook if isinstance(assets.playbook[g], dict)]
                + [g for g in _grammar_commands if not g.startswith("_")]
                + _CLI_CMDS
            ))
            matches = [c for c in candidates if c.startswith(text)]
            return matches[state] if state < len(matches) else None

        # ── Deep grammar-tree completion ───────────────────────────────────────
        # After a known grammar name, each additional Tab press goes one level
        # deeper by using the last-typed rule as the context for expansion.
        #
        #   pyhealthcheck <TAB>
        #     → expand root rule "pyhealthcheck"
        #     → py_system_status  py_resource_status  py_network_status
        #
        #   pyhealthcheck py_system_status <TAB>
        #     → expand "py_system_status"
        #     → py_check_kernel  py_check_uptime  py_check_services
        #
        #   pyhealthcheck py_system_status py_check<TAB>
        #     → context rule = "py_system_status" (word before cursor)
        #     → filter expansion by "py_check"
        #     → py_check_kernel  py_check_uptime  py_check_services
        gname = parts[0]
        if gname in assets.playbook and isinstance(assets.playbook[gname], dict):
            subtree = assets.playbook[gname]
            gcmds   = _grammar_commands.get(gname, {})

            if trailing:
                # Tab after a space → expand the last complete word
                context = parts[-1]
                candidates = _grammar_expand(context, subtree)
            else:
                # Tab mid-word → expand the rule before the current word
                context    = parts[-2] if len(parts) >= 3 else gname
                candidates = _grammar_expand(context, subtree)

            # Nothing from the playbook at this depth — widen to all rules + tokens
            if not candidates:
                candidates = sorted(set(
                    list(subtree.keys())
                    + [k for k in gcmds if not k.startswith("_")]
                ))

            matches = [c for c in candidates if c.startswith(text)]

            # Still nothing — ask the model for suggestions (one live Ollama call)
            if state == 0 and not matches:
                try:
                    _ctx  = " ".join(parts)
                    _ans  = get_ollama_answer(_ctx, NAME, os.environ["OLLAMA_HOST"])
                    _toks = [t.strip().strip('"').strip("'").rstrip(",")
                             for t in _ans.replace(",", " ").split() if t.strip()]
                    matches = [t for t in _toks if t.startswith(text) and t]
                except Exception:
                    pass

            return matches[state] if state < len(matches) else None

    except Exception:
        pass
    return None

readline.set_completer(_tab_completer)

logger.log("info", "SYSTEM", "=======================================================================================")
logger.log("info", "SYSTEM", "🚀 NATIVE CLIENT SIMULATION ENVIRONMENT ACTIVE: '" + NAME + "'")
logger.log("info", "SYSTEM", "=======================================================================================")
logger.log("info", "SYSTEM", "(Type your messages naturally. Use UP/DOWN arrows for history. Type /bye to close)\n")

# Local append log coordinates to safely preserve conversation timelines
local_log_path = "./chroma_db/chat_history.log"
os.makedirs("./chroma_db", exist_ok=True)




while True:
    try:
        # 1. Capture user keyboard entries using active readline tracking layers
        user_input = input(">>> ").strip()
        if not user_input:
            continue
            
        # 2. MATCH NATIVE OLLAMA SHUTDOWN COMMANDS
        if user_input.lower() in ["/bye", "exit", "quit"]:
            logger.log("ok", "SYSTEM", "👋 Closing client session cleanly. Goodbye!")
            readline.write_history_file(_HISTORY_FILE)
            break
            
        # 3. MATCH NATIVE OLLAMA HELP SHORTCUTS
        if user_input.strip() == "/?":
            print( "Available Commands:")
            print( "  /read <file>            Train a markdown / .json (prose or playbook) file on the fly")
            print( "  /grammar <file>         Load a BNF/EBNF grammar file and augment the model")
            print( "  /run [grammar] <expr>   Parse and evaluate an expression via the loaded grammar")
            print( "  /npu [dir]              Export model to ONNX for STM32Cube.AI / STM32N6570-DK NPU")
            print( "  /context                Show loaded files, vocabularies, exec modes, knowledge stats")
            print( "  /tokens [grammar]       Show grammar rules and token commands (all or per grammar)")
            print( "  /bye                    Exit the interactive client session")
            print( "  /?                      Show this system help description summary")
            print( "  TAB                     Complete grammar names, rules, and tokens dynamically")
            print( "  UP/DOWN Arrows          Navigate through your previously entered prompts")
            if _EXTERNAL_MODEL:
                print( "\n  [external model mode: --model " + NAME + "]")
                print( "  /read .md and /npu are unavailable; grammar/vocabulary files load normally.\n")
            else:
                print()
            readline.add_history(user_input)
            continue

        # 3b. IN-FLIGHT GRAMMAR: /grammar <file> parses a BNF/EBNF file and augments the model.
        if user_input.lower().startswith("/grammar"):
            readline.add_history(user_input)
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                logger.log("warning", "SYSTEM", "Usage: /grammar <path-to-grammar.(txt|bnf|ebnf)>")
                continue
            grammar_path = parts[1].strip()
            if _EXTERNAL_MODEL:
                # In external model mode, load grammar into the playbook only — no model rebuild.
                _gf = ModelGrammar.load_file(grammar_path, logger=logger)
                if _gf:
                    assets.playbook[_gf["name"]] = _gf["tree"]
                    assets.grammar_name = _gf["name"]
                    _reload_command_vocabularies()
                    logger.log("ok", "SYSTEM",
                               "Grammar '" + _gf["name"] + "' loaded into playbook "
                               "(external model mode — model weights unchanged).")
                else:
                    logger.log("error", "SYSTEM", "Could not load grammar '" + grammar_path + "'.")
            else:
                logger.log("info", "SYSTEM", "Loading grammar '" + grammar_path + "' into the model...")
                if ModelGrammar.augment(assets, grammar_path, rebuild=True):
                    logger.log("ok", "SYSTEM", "Model augmented with grammar from '" + grammar_path + "'. Ask away!")
                else:
                    logger.log("error", "SYSTEM", "Could not load grammar '" + grammar_path + "' (see errors above).")
            continue

        # 3c. IN-FLIGHT LEARNING: /read <file> trains a markdown or JSON file into the model.
        #     Command vocabulary JSON (_type=command_vocabulary) is loaded into _grammar_commands
        #     without model retraining (the commands dict is runtime config, not model weights).
        if user_input.lower().startswith("/read"):
            readline.add_history(user_input)
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                logger.log("warning", "SYSTEM", "Usage: /read <path-to-file.(md|json)>")
                continue
            doc_path = parts[1].strip()
            # Fast path: command vocabulary JSON — no model rebuild needed.
            if doc_path.lower().endswith(".json"):
                try:
                    with open(doc_path, "r", encoding="utf-8") as _f:
                        _obj = json.load(_f)
                    if isinstance(_obj, dict) and _obj.get("_type") == "command_vocabulary":
                        _gn = _obj.get("_grammar", "unknown")
                        _cmds = {k: v for k, v in _obj.items()
                                 if not k.startswith("_") and isinstance(v, str)}
                        if "_exec" in _obj:          # preserve execution mode (shell / python)
                            _cmds["_exec"] = _obj["_exec"]
                        _grammar_commands[_gn] = _cmds
                        logger.log("ok", "SYSTEM",
                                   "Loaded " + str(len(_cmds)) + " command(s) for '"
                                   + _gn + "' grammar"
                                   + " (exec=" + _obj.get("_exec", "shell") + ").")
                        continue
                except Exception as _e:
                    logger.log("error", "SYSTEM", "Could not read '" + doc_path + "': " + str(_e))
                    continue
            if _EXTERNAL_MODEL:
                logger.log("warning", "SYSTEM",
                           "In-flight markdown learning is not available in external model mode.")
            else:
                logger.log("info", "SYSTEM", "Learning file '" + doc_path + "' into the model...")
                if assets.learn_document(doc_path, rebuild=True):
                    logger.log("ok", "SYSTEM", "Model updated with '" + doc_path + "'. Ask away!")
                else:
                    logger.log("error", "SYSTEM", "Could not learn '" + doc_path + "' (see errors above).")
            continue

        # 3d. NPU EXPORT: "/npu [output_dir]" exports the model to ONNX for STM32Cube.AI.
        if user_input.split()[0].lower() == "/npu":
            readline.add_history(user_input)
            parts = user_input.split(maxsplit=1)
            npu_out = parts[1].strip() if len(parts) > 1 else "models/npu_export"
            logger.log("info", "NPU", "Exporting model to '" + npu_out + "' for STM32Cube.AI...")
            try:
                import model_export_npu as _npu_mod
                if _EXTERNAL_MODEL:
                    # Chat uses an external Ollama model, but local weights may still exist.
                    # Load them from disk exactly as the standalone script does.
                    _npu_cfg_file = os.path.join(model_path, "config.json")
                    _npu_has_wts  = any(
                        os.path.isfile(os.path.join(model_path, f))
                        for f in ("model.safetensors", "pytorch_model.bin")
                    )
                    if not os.path.isfile(_npu_cfg_file) or not _npu_has_wts:
                        logger.log("warning", "NPU",
                                   "No local weights found in '" + model_path + "/'. "
                                   "Run without --model first to train and save the model.")
                        continue
                    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2TokenizerFast
                    logger.log("info", "NPU", "Loading saved weights from " + model_path + "/ ...")
                    _npu_config    = Qwen2Config.from_pretrained(model_path)
                    _npu_model     = Qwen2ForCausalLM.from_pretrained(model_path)
                    _npu_tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)
                    _npu_state     = ModelAssets.load_state(STATE_PATH)
                    _npu_arch      = _npu_state.get("arch", {}) if _npu_state else {}
                    _npu_mod.export_for_npu(
                        model=_npu_model,
                        tokenizer=_npu_tokenizer,
                        config=_npu_config,
                        arch=_npu_arch,
                        model_path=model_path,
                        output_dir=npu_out,
                        logger=logger,
                    )
                else:
                    _npu_mod.export_for_npu(
                        model=artifacts["model"],
                        tokenizer=artifacts["tokenizer"],
                        config=artifacts["config"],
                        arch=artifacts["arch"],
                        model_path=model_path,
                        output_dir=npu_out,
                        logger=logger,
                    )
            except Exception as _e:
                logger.log("error", "NPU", "Export failed: " + str(_e))
            continue

        # 3e. CONTEXT: "/context" lists every loaded file, vocabulary, and knowledge stat.
        if user_input.strip().lower() == "/context":
            readline.add_history(user_input)
            print("\n=== Context ===\n")
            print("Startup files (INIT_KNOWLEDGE_FILES):")
            for _cf in INIT_KNOWLEDGE_FILES:
                print("  " + _cf)
            if _CLI_FILES:
                print("\nCLI files (--train / --grammar):")
                for _cf in _CLI_FILES:
                    print("  " + _cf)
            if _grammar_commands:
                print("\nCommand vocabularies:")
                for _gn, _gc in _grammar_commands.items():
                    _em = _gc.get("_exec", "shell")
                    _toks = [k for k in _gc if not k.startswith("_")]
                    print("  " + _gn + "  [exec=" + _em + "]  " + str(len(_toks)) + " token(s)")
                    _row = "    "
                    for _t in _toks:
                        if len(_row) + len(_t) + 2 > 78:
                            print(_row)
                            _row = "    " + _t + "  "
                        else:
                            _row += _t + "  "
                    if _row.strip():
                        print(_row)
            if assets.playbook:
                print("\nPlaybook grammars:")
                for _gn, _sub in assets.playbook.items():
                    if isinstance(_sub, dict):
                        print("  " + _gn + "  " + str(len(_sub)) + " rule(s)")
            _ndocs = len(getattr(assets, "knowledge_texts", []))
            print("\nKnowledge memory: " + str(_ndocs) + " prose document(s)")
            print()
            continue

        # 3f. TOKENS: "/tokens [grammar]" prints grammar rules + token commands.
        if user_input.split()[0].lower() == "/tokens":
            readline.add_history(user_input)
            _tp = user_input.split(maxsplit=1)
            _filter = _tp[1].strip() if len(_tp) > 1 else None
            if _filter and _filter not in assets.playbook and _filter not in _grammar_commands:
                print("Grammar '" + _filter + "' not found.")
                _available = sorted(set(list(assets.playbook.keys()) + list(_grammar_commands.keys())))
                print("Available: " + ", ".join(_available))
                continue
            _grammars = ([_filter] if _filter
                         else sorted(set(list(assets.playbook.keys()) + list(_grammar_commands.keys()))))
            for _gn in _grammars:
                _gc   = _grammar_commands.get(_gn, {})
                _sub  = assets.playbook.get(_gn, {}) if isinstance(assets.playbook.get(_gn), dict) else {}
                _em   = _gc.get("_exec", "shell")
                print("\n=== Grammar: " + _gn + "  [exec=" + _em + "] ===")
                if _sub:
                    print("\nRules:")
                    _max = max(len(r) for r in _sub)
                    for _rn, _rb in _sub.items():
                        _body = _rb if isinstance(_rb, str) else str(_rb)
                        print("  <" + _rn.ljust(_max) + ">  ::=  " + _body)
                _toks = {k: v for k, v in _gc.items() if not k.startswith("_")}
                if _toks:
                    print("\nTokens:")
                    for _tok, _cmd in _toks.items():
                        print("  " + _tok + ":")
                        for _line in _cmd.splitlines():
                            print("    " + _line)
            print()
            continue

        # 3g. GRAMMAR RUNNER: "/run [grammar_name] <expr>" parses and evaluates an expression
        #     through the trained grammar using one model interaction per unique rule.
        if user_input.split()[0].lower() == "/run":
            readline.add_history(user_input)
            parts = user_input.split(maxsplit=2)
            if len(parts) < 2:
                logger.log("warning", "SYSTEM", "Usage: /run [grammar_name] <expression>")
                continue
            # Detect whether the second token is a known grammar name.
            if len(parts) >= 3 and parts[1] in assets.playbook:
                run_grammar, run_expr = parts[1], parts[2]
            else:
                run_grammar = (assets.grammar_name
                               or (list(assets.playbook.keys())[0] if assets.playbook else None))
                run_expr = " ".join(parts[1:])
            if not run_grammar:
                logger.log("warning", "SYSTEM", "No grammar loaded. Use /grammar <file> first.")
                continue
            run_subtree = assets.playbook.get(run_grammar, {})
            run_start = next(iter(run_subtree)) if run_subtree else "expr"
            runner = GrammarRunner(
                grammar_name=run_grammar,
                query_fn=lambda p: get_ollama_answer(p, NAME, os.environ["OLLAMA_HOST"]),
                fallback_playbook=run_subtree,
                commands=_grammar_commands.get(run_grammar, {}),
                logger=logger,
            )
            # Execute mode: /run healthcheck (no expression, or expression == grammar/start rule)
            if run_expr.strip().lower() in ("", run_grammar.lower(), run_start.lower()):
                runner.execute(start_rule=run_start)
            else:
                runner.run(run_expr, start_rule=run_start)
            continue

        # 3g. AUTO-DETECT GRAMMAR — three modes, checked in order:
        #
        #   Execute mode (full):    input == grammar name with commands loaded
        #                           e.g. "healthcheck" → full procedure from top rule
        #   Execute mode (partial): input == any rule name or command token inside a
        #                           grammar that has commands loaded
        #                           e.g. "system_status" → sub-procedure branch
        #                           e.g. "check_uptime"  → single command token
        #   Parse mode:             input fully parses against any loaded grammar
        #                           (zero model calls — playbook-only probe)
        #                           e.g. "1+1" → calculator grammar → Result: 2
        #
        # Falls through to normal chat if none of the three match.
        auto_handled = False
        _ustrip  = user_input.strip()
        _ulow    = _ustrip.lower()
        _uparts  = _ustrip.split()

        # Helper: make a GrammarRunner for a given grammar.
        def _make_runner(gname, gsubtree, gcmds):
            return GrammarRunner(
                grammar_name=gname,
                query_fn=lambda p: get_ollama_answer(p, NAME, os.environ["OLLAMA_HOST"]),
                fallback_playbook=gsubtree,
                commands=gcmds,
                logger=logger,
            )

        # ── Mode 1: single word == grammar name ────────────────────────────────
        for _gname, _gsubtree in assets.playbook.items():
            if not isinstance(_gsubtree, dict) or not _gsubtree:
                continue
            if _ulow == _gname.lower() and _grammar_commands.get(_gname):
                _gstart = next(iter(_gsubtree))
                logger.log("info", "SYSTEM",
                           "Auto-detected '" + _gname + "' procedure — executing grammar...")
                _make_runner(_gname, _gsubtree, _grammar_commands[_gname]).execute(start_rule=_gstart)
                auto_handled = True
                break

        # ── Mode 2: multi-word path  grammar_name [rule ...] target ───────────
        # Handles what Tab completion builds up:
        #   pyhealthcheck py_system_status          → execute py_system_status sub-tree
        #   pyhealthcheck py_check_kernel           → run py_check_kernel command directly
        #   pyhealthcheck py_system_status py_check_uptime → run py_check_uptime
        # Rule: first word is the grammar name, LAST word is the deepest target.
        if not auto_handled and len(_uparts) >= 2:
            _path_gname  = _uparts[0]
            _path_target = _uparts[-1]
            _path_sub    = assets.playbook.get(_path_gname, {})
            _path_cmds   = _grammar_commands.get(_path_gname, {})
            if isinstance(_path_sub, dict) and _path_sub:
                if _path_target in _path_cmds and not _path_target.startswith("_"):
                    # Deepest word is a command token → run it directly.
                    logger.log("info", "SYSTEM",
                               "Auto-detected path '" + " → ".join(_uparts)
                               + "' — running command token...")
                    _r = _make_runner(_path_gname, _path_sub, _path_cmds)
                    _r._run_os_command(_path_target, _path_cmds[_path_target])
                    auto_handled = True
                elif _path_target in _path_sub:
                    # Deepest word is a grammar rule → execute sub-procedure from there.
                    logger.log("info", "SYSTEM",
                               "Auto-detected path '" + " → ".join(_uparts)
                               + "' — executing sub-procedure...")
                    _make_runner(_path_gname, _path_sub, _path_cmds).execute(start_rule=_path_target)
                    auto_handled = True

        # ── Mode 3: single word == rule name or bare command token in any grammar ──
        if not auto_handled and len(_uparts) == 1:
            for _gname, _gcmds in _grammar_commands.items():
                _gsubtree = assets.playbook.get(_gname, {})
                if not isinstance(_gsubtree, dict) or not _gsubtree:
                    continue
                if _ustrip in _gsubtree:
                    logger.log("info", "SYSTEM",
                               "Auto-detected '" + _ustrip + "' rule in '"
                               + _gname + "' — executing sub-procedure...")
                    _make_runner(_gname, _gsubtree, _gcmds).execute(start_rule=_ustrip)
                    auto_handled = True
                    break
                if _ustrip in _gcmds and not _ustrip.startswith("_"):
                    logger.log("info", "SYSTEM",
                               "Auto-detected command token '" + _ustrip
                               + "' in '" + _gname + "' — running command...")
                    _r = _make_runner(_gname, _gsubtree, _gcmds)
                    _r._run_os_command(_ustrip, _gcmds[_ustrip])
                    auto_handled = True
                    break

        # --- parse mode ---
        if not auto_handled:
            for _gname, _gsubtree in assets.playbook.items():
                if not isinstance(_gsubtree, dict) or not _gsubtree:
                    continue
                _gstart = next(iter(_gsubtree))
                if GrammarRunner.probe(_gname, user_input, _gsubtree, _gstart):
                    logger.log("info", "SYSTEM",
                               "Auto-detected '" + _gname + "' expression — running grammar...")
                    _runner = GrammarRunner(
                        grammar_name=_gname,
                        query_fn=lambda p: get_ollama_answer(p, NAME, os.environ["OLLAMA_HOST"]),
                        fallback_playbook=_gsubtree,
                        commands=_grammar_commands.get(_gname, {}),
                        logger=logger,
                    )
                    _runner.run(user_input, start_rule=_gstart)
                    auto_handled = True
                    break

        if auto_handled:
            continue

        # 4. SEND USER INPUT TO MODEL AND GET THE ANSWER
        print("Thinking...querying the model...")  

        # Change "llama3" to "mistral", "phi3", etc., depending on what you downloaded
        target_response = get_ollama_answer(prompt=user_input, model_name=NAME)

        # 5. STREAMING STEP: Emulate true native token typing latency output
        # logger.log("info", "SYSTEM", " ")  # Clean leading block spacing line
        print(" ")
        # print(f"\nOllama > {target_response}\n")
        for character in target_response:
            sys.stdout.write(character)
            sys.stdout.flush()
            time.sleep(0.03)  # Balanced 30ms character refresh matches native engines

        print("\n")  # Newline tracking spacing buffer

        # 6. PERSISTENCE STEP: Commit the turn context directly to your local file cache
        readline.add_history(user_input)
        with open(local_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"User: {user_input} | Assistant: {target_response}\n")

    except KeyboardInterrupt:
        logger.log("ok", "SYSTEM", "👋 Client session closed cleanly.")
        break
    except Exception as e:
        logger.log("error", "SYSTEM", "❌ CLIENT TERMINAL EXCEPTION: " + str(e))

# # =========================================================================
# # 12. AUTOMATICATED INTERACTIVE RUNNER
# # =========================================================================
# import subprocess

# print(f"\n--> Handing control over to the interactive Ollama environment...")
# print(f"--> Booting user session for model: '{OLLAMA_MODEL_NAME}'")
# print(f"    (Type your prompt below. Use Ctrl+D or type /bye to exit)\n")

# try:
#     # Use subprocess.run without capturing stdout/stderr to let Ollama
#     # natively take over the terminal input, colors, and streaming output.
#     subprocess.run(
#         ["ollama", "run", OLLAMA_MODEL_NAME],
#         check=True
#     )
#     print(f"\n✅ Interactive session for '{OLLAMA_MODEL_NAME}' closed cleanly.")

# except subprocess.CalledProcessError as e:
#     print(f"\n❌ RUNNER EXECUTION EXCEPTION: Ollama process exited with error code {e.returncode}")
# except KeyboardInterrupt:
#     print(f"\n\n👋 Interactive session interrupted by user. Exiting cleanly.")

#################################################################################################
# END OF PROGRAM
#################################################################################################
logger.log("ok", "SYSTEM", "\nProgram completed.")