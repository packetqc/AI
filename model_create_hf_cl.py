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
from class_terminal_logs import TerminalLogger
logger = TerminalLogger()

from class_model_assets import ModelAssets

#################################################################################################
#
#################################################################################################
version = "1"
model_create = "model_optimized_" + version

# model_path = "./"+model_create
model_path = model_create
gguf_path = model_path+".gguf"

modelfile_path = "./Modelfile"

NAME = "model_hugging_face_optimized"
OLLAMA_MODEL_NAME = "model_hugging_face_optimized"

# Knowledge files trained into the model at start-up, IN ORDER (markdown / plain JSON / grammar
# JSON; empty list to skip). Each is routed by extension exactly like the in-flight "/read".
# NOTE: only used on a fresh run — once a state file exists it is restored instead (see below).
INIT_KNOWLEDGE_FILES = ["knowledge.md", "train_healthcheck.json", "train_ops_playbook.json"]

# Persistence: accumulated knowledge + the (possibly adapted) config are saved here so that a
# later run of this script restores everything learned in previous sessions.
STATE_PATH = model_path + ".state.json"

# Ollama serving parameters (shared by the first build and every /read rebuild).
# NOTE: plain string so the Ollama "{{ .Prompt }}" double-braces survive; the template
# reproduces EXACTLY the training prefix ("<|endoftext|>" + prompt, no trailing space).
MODELFILE_TEMPLATE = "<|endoftext|>{{ .Prompt }}"
NUM_PREDICT = 32   # enough headroom for multi-command grammar answers (stops early on EOS)


os.makedirs("./"+model_path, exist_ok=True)

ok = True

# Enforce clean environment routing to prevent network proxy delays
os.environ["MEM0_TELEMETRY"] = "false"
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

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
    model = Qwen2ForCausalLM(config)
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

    input_ids = torch.tensor(input_rows, dtype=torch.long)
    labels = torch.tensor(label_rows, dtype=torch.long)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    logger.log("info", "SYSTEM", "Training " + str(len(rows)) + " rows (anchors + knowledge) jointly...")
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
        tree = ModelAssets.playbook_tree_from_json(obj)
        if tree is not None:                             # grammar/playbook JSON -> merge tree
            ModelAssets.merge_tree(playbook, tree)
            logger.log("info", "SYSTEM", "Seeded playbook (" + str(len(tree))
                       + " root(s)) from " + path)
            return obj.get("grammar") if isinstance(obj, dict) else None
        raw = ModelAssets.json_to_text(obj)              # plain JSON -> flattened facts
    if raw.strip():
        knowledge_texts.append(raw)
        logger.log("info", "SYSTEM", "Seeded start-up knowledge from " + path)
    return None


# Restore previous-session state if present; otherwise seed from the start-up knowledge files.
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
else:
    knowledge_texts = []
    playbook = {}
    grammar_name = None
    arch = dict(ARCH_DEFAULTS)
    vocab_cap = DEFAULT_VOCAB_CAP
    for _path in INIT_KNOWLEDGE_FILES:
        _name = seed_from_file(_path, knowledge_texts, playbook)
        if _name:
            grammar_name = _name

# Flatten the playbook tree into anchor pairs for training (children listings + leaf answers).
grammar_pairs = ModelAssets.flatten_playbook(playbook, grammar_name)

# Build the tokenizer + model once over the base prompts + grammar pairs + (restored/seeded) knowledge.
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
assets.save_state()  # persist the start-up state (knowledge + playbook + adapted config)

# First build of the served Ollama model, then a one-time GGUF inspection.
assets.export_and_rebuild()
assets.inspect_gguf()

# =========================================================================
# AUTOMATED SELF-TEST: verify the two canonical prompts answer green / red
# =========================================================================
# Catches any residual train/inference tokenization drift instead of silently "succeeding".
logger.log("info", "SYSTEM", "Running self-test against the two canonical prompts...")
time.sleep(1)  # let Ollama finish registering the freshly-built model
all_passed = True
# Anchors (green/red) check exact answer; grammar routines check the first command appears.
selftest_cases = [(p, a, a) for p, a in PAIRS]
selftest_cases += [(p, a, a.split(",")[0].strip()) for p, a in grammar_pairs]
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
    logger.log("error", "SELFTEST", "❌ Self-test failed — check tokenizer/template parity (see plan verification step 4).")

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
readline.clear_history()

logger.log("info", "SYSTEM", "=======================================================================================")
logger.log("info", "SYSTEM", "🚀 NATIVE CLIENT SIMULATION ENVIRONMENT ACTIVE: '" + NAME + "'")
logger.log("info", "SYSTEM", "=======================================================================================")
logger.log("info", "SYSTEM", "(Type your messages naturally. Use UP/DOWN arrows for history. Type /bye to close)\n")

# Local append log coordinates to safely preserve conversation timelines
local_log_path = "./chroma_db/chat_history.log"
os.makedirs("./chroma_db", exist_ok=True)


def run_os_commands(commands):
    """Execute a list of shell commands on this OS, streaming each one's output.
    Only ever called AFTER the user has confirmed the displayed command list."""
    for cmd in commands:
        logger.log("info", "EXEC", "$ " + cmd)
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = (result.stdout or "") + (result.stderr or "")
            if output.strip():
                print(output.rstrip())
            logger.log("ok" if result.returncode == 0 else "warning", "EXEC",
                       "'" + cmd + "' exited with code " + str(result.returncode))
        except subprocess.TimeoutExpired:
            logger.log("error", "EXEC", "'" + cmd + "' timed out after 30s.")
        except Exception as e:
            logger.log("error", "EXEC", "'" + cmd + "' failed: " + str(e))


def playbook_lookup(path_tokens):
    """HYBRID step: query the MODEL with the stacked path AND resolve it in the playbook tree.
    Returns (joined_prompt, model_answer, tree_node, found_in_tree)."""
    prompt = " ".join(path_tokens)
    node, found = ModelAssets.resolve_path(assets.playbook, path_tokens)
    print("Thinking...querying the model...")
    answer = get_ollama_answer(prompt=prompt, model_name=NAME, host_url=os.environ["OLLAMA_HOST"])
    return prompt, answer, node, found


def confirm_and_run(answer, label=""):
    """Parse the model's comma-separated command answer, show it, and run AFTER y/N confirmation."""
    commands = [c.strip() for c in answer.split(",") if c.strip()]
    if not commands:
        logger.log("error", "SYSTEM", "No commands to run" + (" for '" + label + "'" if label else "") + ".")
        return
    print("The model maps " + (("'" + label + "' ") if label else "") + "to " + str(len(commands)) + " command(s):")
    for c in commands:
        print("   $ " + c)
    confirm = input("Execute these on this operating system? [y/N] ").strip().lower()
    if confirm in ("y", "yes"):
        run_os_commands(commands)
    else:
        logger.log("info", "SYSTEM", "Skipped execution.")


while True:
    try:
        # 1. Capture user keyboard entries using active readline tracking layers
        user_input = input(">>> ").strip()
        if not user_input:
            continue
            
        # 2. MATCH NATIVE OLLAMA SHUTDOWN COMMANDS
        if user_input.lower() in ["/bye", "exit", "quit"]:
            logger.log("ok", "SYSTEM", "👋 Closing client session cleanly. Goodbye!")
            # Instantly break out of the script without hitting Mem0's broken JSON loops!
            break
            
        # 3. MATCH NATIVE OLLAMA HELP SHORTCUTS
        if user_input.strip() == "/?":
            print( "Available Commands:")
            print( "  /read <file>      Train a markdown / .json (prose or playbook) file on the fly")
            print( "  /play <path...>   Walk the playbook tree: lists a node's routes, or shows a leaf")
            print( "  /cmd <path...>    Run a command leaf's commands on the OS (after confirmation)")
            print( "  /bye              Exit the interactive client session")
            print( "  /?                Show this system help description summary")
            print( "  UP/DOWN Arrows    Navigate through your previously entered prompts\n")
            readline.add_history(user_input)
            continue

        # 3b. IN-FLIGHT LEARNING: /read <file> trains a markdown or JSON file into the model,
        #     re-exports the GGUF and rebuilds the Ollama model so the next prompt sees it.
        if user_input.lower().startswith("/read"):
            readline.add_history(user_input)
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                logger.log("warning", "SYSTEM", "Usage: /read <path-to-file.(md|json)>")
                continue
            doc_path = parts[1].strip()
            logger.log("info", "SYSTEM", "Learning file '" + doc_path + "' into the model...")
            if assets.learn_document(doc_path, rebuild=True):
                logger.log("ok", "SYSTEM", "Model updated with '" + doc_path + "'. Ask away!")
            else:
                logger.log("error", "SYSTEM", "Could not learn '" + doc_path + "' (see errors above).")
            continue

        # 3c. PLAYBOOK NAVIGATION: "/play <path...>" walks the trained tree (stacked route).
        #     Hybrid: the answer comes from the MODEL; the parsed tree tells us node vs leaf.
        if user_input.split()[0].lower() == "/play":
            readline.add_history(user_input)
            tokens = user_input.split()[1:]
            if not tokens:                                   # no path -> list the roots
                roots = list(assets.playbook.keys())
                logger.log("info", "PLAY", "roots: " + (", ".join(roots) or "(none)")
                           + "   — descend with: /play <root>")
                continue
            prompt, answer, node, found = playbook_lookup(tokens)
            print("[" + "/".join(tokens) + "]  model: " + answer.strip())
            if not found:
                logger.log("warning", "PLAY", "'" + prompt + "' is not a known route "
                           + "(showing the model's answer only).")
            elif isinstance(node, dict):                     # internal node -> routes to descend
                logger.log("ok", "PLAY", "node — routes: " + ", ".join(node.keys())
                           + "   — descend with: /play " + " ".join(tokens) + " <route>")
            elif isinstance(node, list):                     # command leaf -> offer to run
                confirm_and_run(answer, label=prompt)
            else:                                            # answer leaf -> already printed
                logger.log("ok", "PLAY", "answer leaf.")
            continue

        # 3d. GRAMMAR EXECUTION: "/cmd <path...>" resolves a COMMAND LEAF and runs it (after
        #     confirmation). Backward-compatible with flat routine names, e.g. "/cmd healthcheck".
        if user_input.split()[0].lower() == "/cmd":
            readline.add_history(user_input)
            tokens = user_input.split()[1:]
            if not tokens:
                logger.log("warning", "SYSTEM", "Usage: /cmd <path...>   (e.g. /cmd healthcheck  or  /cmd diagnostics disk)")
                continue
            prompt, answer, node, found = playbook_lookup(tokens)
            if found and isinstance(node, dict):
                logger.log("warning", "SYSTEM", "'" + prompt + "' is a node, not a command leaf. "
                           + "Routes: " + ", ".join(node.keys()) + "  (use /play to descend).")
                continue
            if found and isinstance(node, str):
                logger.log("info", "SYSTEM", "'" + prompt + "' is an answer leaf: " + answer.strip())
                continue
            if not found:
                logger.log("warning", "SYSTEM", "'" + prompt + "' is not a known route; running the "
                           + "model's answer anyway.")
            confirm_and_run(answer, label=prompt)
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