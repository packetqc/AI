import json
import os
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
NUM_LAYERS = 1  #12 #4  #2
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
# TOKENS
#################################################################################################
logger.log("info", "SYSTEM", "Training a real byte-level BPE tokenizer (single source of truth)...")

# The canonical prompt/answer pairs the model must learn. These exact strings are reused
# for (a) training the tokenizer, (b) building the training tensors, (c) the self-test,
# so training token IDs are guaranteed identical to the IDs Ollama produces at inference.
SPECIAL_TOKEN = "<|endoftext|>"
PAIRS = [
    ("what color is always good ?", "green"),
    ("what color is always bad ?",  "red"),
]

# Format a single training example. The TEMPLATE in the Modelfile reproduces everything
# up to and including the trailing space, then the model must generate the answer + EOS.
def format_example(prompt: str, answer: str) -> str:
    return f"{SPECIAL_TOKEN}{prompt} {answer}{SPECIAL_TOKEN}"

# Tiny corpus: the prompts, the answers, and the full formatted lines, repeated so that
# whole words (green/red/color/...) collapse into single merged tokens.
corpus = []
for prompt, answer in PAIRS:
    corpus.extend([prompt, answer, format_example(prompt, answer)] * 20)

# CRITICAL: use the exact Qwen2 pre-tokenizer regex AND export pre="qwen2" below, so that
# HF (training) and llama.cpp/Ollama (inference) split text identically. With a plain ByteLevel
# pre-tokenizer the two diverge on " ?" (HF keeps "Ġ?" as one token, llama.cpp splits "Ġ"+"?"),
# which shifts the prompt's final token and makes the model answer with EOS instead of green/red.
QWEN2_PRETOKENIZER_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Byte-level BPE: the 256-byte initial alphabet makes ANY text tokenizable.
raw_tokenizer = Tokenizer(models.BPE(unk_token=None))
raw_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(pattern=Regex(QWEN2_PRETOKENIZER_PATTERN), behavior="isolated", invert=False),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
raw_tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=512,
    special_tokens=[SPECIAL_TOKEN],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
)
raw_tokenizer.train_from_iterator(corpus, trainer=trainer)

tokenizer = Qwen2TokenizerFast(tokenizer_object=raw_tokenizer)
tokenizer.add_special_tokens({
    "bos_token": SPECIAL_TOKEN,
    "eos_token": SPECIAL_TOKEN,
    "pad_token": SPECIAL_TOKEN,
})

# Save weightless files directly to disk (No internet requests sent)
tokenizer.save_pretrained("./"+model_path)

#################################################################################################
# SYSTEM VOCABULARY ROUTING  (derived entirely from the trained tokenizer)
#################################################################################################
# Token list ordered by id — this is exactly what gets embedded in the GGUF.
vocab = tokenizer.get_vocab()                       # {token_str: id}
tokens = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]

# Real merge rules from the trained tokenizer (replaces the old 3-item junk fallback).
_tok_state = json.loads(raw_tokenizer.to_str())
merges = []
for m in _tok_state["model"]["merges"]:
    # tokenizers may serialize merges as "a b" strings or as ["a", "b"] pairs.
    merges.append(m if isinstance(m, str) else " ".join(m))

scores = [0.0] * len(tokens)
token_types = [1] * len(tokens)
token_types[tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN)] = 3   # control token

VOCAB_SIZE = len(tokens)

logger.log("ok", "SYSTEM", "Vocabulary mapping stabilized at exactly " + str(VOCAB_SIZE) + " tokens, " + str(len(merges)) + " merges.")

#################################################################################################
# CONFIG
#################################################################################################
logger.log("info", "SYSTEM", "Initializing architecture parameters and model...")

# Define the tiny architecture parameters only
config = Qwen2Config(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    max_position_embeddings=CONTEXT_LENGTH,
    # Crucial mapping tags expected by llama.cpp's internal dictionary parser
    architectures=["Qwen2ForCausalLM"],
    model_type="qwen2"
)

# Add this parameter to your Qwen2Config constructor block
config.attention_bias = True

# Save configuration metadata directly to disk
config.save_pretrained("./"+model_path)

jsonConfig = config.to_dict()
logger.log("info", "SYSTEM", "vocab_size: "+str(config.vocab_size)+", hidden_size: "+str(config.hidden_size)+", intermediate_size: "+str(config.intermediate_size)+", num_hidden_layers: "+str(config.num_hidden_layers))
logger.log("info", "SYSTEM", "num_attention_heads: "+str(config.num_attention_heads)+", num_key_value_heads: "+str(config.num_key_value_heads)+", max_position_embeddings: "+str(config.max_position_embeddings)+", model_type: "+str(config.model_type))


# =========================================================================
# 6. INSTANT ONE-LAYER GRADIENT OPTIMIZATION PASS
# =========================================================================
logger.log("info", "SYSTEM", "Initializing model layers with active numeric weights...")
model = Qwen2ForCausalLM(config)
model.resize_token_embeddings(len(tokens), mean_resizing=False)

logger.log("info", "SYSTEM", "Building training tensors by encoding the real prompt/answer strings...")

# Build training rows by ENCODING the actual text with the same tokenizer used for export.
# The ByteLevel pre-tokenizer's gpt2 regex never merges across a space/punctuation boundary,
# so the answer's leading space attaches to it (e.g. "Ġgreen") and encode(prefix) is a clean
# prefix of encode(full). We mask everything up to the answer, training loss only on the
# answer tokens + the closing EOS.
eos_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKEN)

examples = []
max_len = 0
for prompt, answer in PAIRS:
    prefix_ids = tokenizer.encode(f"{SPECIAL_TOKEN}{prompt}")     # exactly what the template feeds
    full_ids = tokenizer.encode(format_example(prompt, answer))   # prompt + answer + EOS
    examples.append((full_ids, len(prefix_ids)))
    max_len = max(max_len, len(full_ids))

batch = examples * 5  # repeat for a stable batch on this tiny model

input_rows, label_rows, mask_rows = [], [], []
for full_ids, prefix_len in batch:
    pad = max_len - len(full_ids)
    ids = full_ids + [eos_id] * pad
    lab = [-100] * len(ids)
    for j in range(prefix_len, len(full_ids)):   # answer tokens + closing EOS only
        lab[j] = ids[j]
    msk = [1] * len(full_ids) + [0] * pad
    input_rows.append(ids)
    label_rows.append(lab)
    mask_rows.append(msk)

input_ids = torch.tensor(input_rows, dtype=torch.long)
labels = torch.tensor(label_rows, dtype=torch.long)
attention_mask = torch.tensor(mask_rows, dtype=torch.long)

# Setup optimization parameters with a clean learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()

logger.log("info", "SYSTEM", "Training until the two examples are memorized...")
for epoch in range(400):
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    if loss.item() < 1e-4:
        break

model.eval()
logger.log("ok", "SYSTEM", "MODEL CONVERGED at epoch " + str(epoch + 1) + ", final loss = " + f"{loss.item():.6f}")


# DO NOT add any manual logit overrides or zeroing below this line!

#################################################################################################
#
#################################################################################################
# logger.log("info", "SYSTEM", "5. Saving optimized weights matrix directly...")
# model.save_pretrained(model_path, safe_serialization=False)

#################################################################################################
# GGUF FILE EXPORT
#################################################################################################
logger.log("info", "SYSTEM", "Exporting to GGUF file...")

# 'qwen2' is the internal string keyword llama.cpp uses to map architectural logic
writer = GGUFWriter(gguf_path, arch=config.model_type)

logger.log("info", "SYSTEM", "Constructing empty Qwen architecture binary layout...")

# 3. Inject Mandatory Structural Metadata
writer.add_string("general.architecture", "qwen2")
writer.add_name(NAME)

writer.add_context_length(config.max_position_embeddings)
writer.add_embedding_length(config.hidden_size)
writer.add_block_count(config.num_hidden_layers)
writer.add_feed_forward_length(config.intermediate_size)

writer.add_head_count(config.num_attention_heads)
writer.add_head_count_kv(config.num_key_value_heads)

writer.add_layer_norm_rms_eps(config.rms_norm_eps)

# Write the token arrays to the binary header block
writer.add_tokenizer_model("gpt2")
writer.add_tokenizer_pre("qwen2")         # MUST match the Qwen2 regex used to train the tokenizer
writer.add_token_list(tokens)
writer.add_token_scores(scores)
writer.add_token_types(token_types)

# Special-token ids so llama.cpp/Ollama stop correctly on <|endoftext|>.
# add_bos_token=False: the Modelfile TEMPLATE already supplies the leading <|endoftext|>,
# so llama.cpp must NOT prepend a second one (that would shift every position).
writer.add_bos_token_id(eos_id)
writer.add_eos_token_id(eos_id)
writer.add_pad_token_id(eos_id)
writer.add_add_bos_token(False)

# =========================================================================
# FIXED PLACE: Inject the REAL trained merge array HERE (Before adding tensors!)
# =========================================================================
writer.add_array("tokenizer.ggml.merges", merges)
logger.log("ok", "SYSTEM", "Tokenizer merges (" + str(len(merges)) + ") successfully added to metadata phase!")

# Extract the final configured parameters normally right below here
logger.log("info", "SYSTEM", "Packing trained tensor matrices into GGUF format...")
state_dict = model.state_dict()

# Your existing tensor mapping loop continues normally right below here...
tensor_mapping = {"model.embed_tokens.weight": "token_embd.weight"}

# Dynamically loop through the total number of layers
# This automatically creates mappings for layers 0, 1, 2, and 3
for layer_idx in range(config.num_hidden_layers):
    tensor_mapping.update({
        f"model.layers.{layer_idx}.input_layernorm.weight":          f"blk.{layer_idx}.attn_norm.weight",
        f"model.layers.{layer_idx}.self_attn.q_proj.weight":         f"blk.{layer_idx}.attn_q.weight",
        f"model.layers.{layer_idx}.self_attn.q_proj.bias":           f"blk.{layer_idx}.attn_q.bias",
        f"model.layers.{layer_idx}.self_attn.k_proj.weight":         f"blk.{layer_idx}.attn_k.weight",
        f"model.layers.{layer_idx}.self_attn.k_proj.bias":           f"blk.{layer_idx}.attn_k.bias",
        f"model.layers.{layer_idx}.self_attn.v_proj.weight":         f"blk.{layer_idx}.attn_v.weight",
        f"model.layers.{layer_idx}.self_attn.v_proj.bias":           f"blk.{layer_idx}.attn_v.bias",
        f"model.layers.{layer_idx}.self_attn.o_proj.weight":         f"blk.{layer_idx}.attn_output.weight",
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"blk.{layer_idx}.ffn_norm.weight",
        f"model.layers.{layer_idx}.mlp.gate_proj.weight":            f"blk.{layer_idx}.ffn_gate.weight",
        f"model.layers.{layer_idx}.mlp.up_proj.weight":              f"blk.{layer_idx}.ffn_up.weight",
        f"model.layers.{layer_idx}.mlp.down_proj.weight":            f"blk.{layer_idx}.ffn_down.weight",
    })

# Append the Final Output Heads (Layer-independent)
tensor_mapping.update({
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight":    "output.weight"
})


# 7. Write complete matrices directly into GGUF container
logger.log("info", "SYSTEM", "Writing full weight and bias matrices into the GGUF binary...")
for torch_name, gguf_name in tensor_mapping.items():
    if torch_name in state_dict:
        tensor_np = state_dict[torch_name].detach().cpu().numpy().astype(np.float32)
        writer.add_tensor(gguf_name, tensor_np)

# Finalize file serialization
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
logger.log("ok", "SYSTEM", "GGUF generation successful!")


logger.log("info", "SYSTEM", gguf_path + " was created with zero errors.")
logger.log("info", "SYSTEM", "File size: " + str(os.path.getsize(gguf_path) / 1024) + " KB")

#################################################################################################
# INSPECT INTERNAL BINARY TAGS OF THE GGUF FILE
#################################################################################################
from gguf import GGUFReader

# 1. Point to your custom binary file
logger.log("info", "SYSTEM", "Opening " + gguf_path + " for binary parsing...")

# 2. Initialize the reader engine
reader = GGUFReader(gguf_path)

logger.log("info", "SYSTEM", "GGUF Container Info")
logger.log("info", "SYSTEM", "Total Metadata KV Pairs: " + str(len(reader.fields)))
logger.log("info", "SYSTEM", "Total Tensors Encoded  : " + str(len(reader.tensors)))

logger.log("info", "SYSTEM", "Decoded Internal Metadata Tags...")

# 3. Safely parse and iterate over the fields dictionary
for key, field in reader.fields.items():
    # GGUF strings and scalars are packed as parts. 
    # Grab the first element data part out of the internal array.
    raw_part = field.parts[field.data[0]]
    
    # Check if the part is a numpy array containing a string/bytes object
    if isinstance(raw_part, np.ndarray) and raw_part.dtype.kind in ['U', 'S', 'O']:
        # Extract the string element
        val = raw_part.item()
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='ignore')
    elif isinstance(raw_part, bytes):
        val = raw_part.decode('utf-8', errors='ignore')
    else:
        # It's a scalar numerical type (int, float)
        val = raw_part[0] if hasattr(raw_part, '__len__') else raw_part

    # Truncate vocabulary listings to avoid cluttering the terminal output
    if "tokenizer.ggml.tokens" in key:
        # print(f"{key:<40} : [Truncated... {len(field.parts)} tokens encoded]")
        logger.log("info", "SYSTEM", key + f" : [Truncated... {len(field.parts)} tokens encoded]")
    elif "tokenizer.ggml.scores" in key or "tokenizer.ggml.token_type" in key:
        continue
    else:
        logger.log("info", "SYSTEM", key + " : " + str(val))

# =========================================================================
# COMBINED AUTOMATED RESET & CACHE-FREE OLLAMA BUILD
# =========================================================================
absolute_gguf_path = os.path.abspath(gguf_path)

# 1. Generate a unique execution timestamp to force-invalidate Ollama's cache
build_timestamp = int(time.time())

# NOTE: plain string (NOT an f-string) so the Ollama "{{ .Prompt }}" double-braces survive.
# The template reproduces EXACTLY the training prefix ("<|endoftext|>" + prompt, no trailing
# space); the model then emits the space-prefixed answer token (e.g. "Ġgreen") and stops on EOS.
modelfile_content = (
    "FROM " + absolute_gguf_path + "\n\n"
    'TEMPLATE """<|endoftext|>{{ .Prompt }}"""\n\n'
    'PARAMETER stop "<|endoftext|>"\n'
    "PARAMETER temperature 0.0\n"
    "PARAMETER num_predict 8\n"
)

# 3. Write the physical configuration layout descriptor to disk
with open(modelfile_path, "w", encoding="utf-8") as f:
    f.write(modelfile_content.strip())

logger.log("ok", "SYSTEM", "Successfully generated " + modelfile_path + " pointing directly to " + gguf_path)


# 4. Force-unload the model if it is currently running in memory
logger.log("info", "SYSTEM", "Checking and unloading '" + NAME + "' from system memory memory...")
try:
    # We send a clean POST payload with keep_alive=0 to the local daemon API.
    # This instructs the server to immediately kill any running runner instances.
    unload_url = os.environ["OLLAMA_HOST"] + "/api/generate"
    payload = json.dumps({"model": NAME, "keep_alive": 0}).encode("utf-8")
    
    req = urllib.request.Request(
        unload_url, 
        data=payload, 
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    # Send request with a short 3-second timeout so it never hangs the script
    with urllib.request.urlopen(req, timeout=3) as response:
        if response.status == 200:
            logger.log("ok", "SYSTEM", "[Ollama Daemon]: Run states cleanly terminated.")
            
except Exception:
    # If Ollama is idle or closed, ignore the error and proceed safely
    logger.log("info", "SYSTEM", "OK, No active running instances detected.")

# 5. Now it is completely safe to delete the container registry layers
logger.log("info", "SYSTEM", "Deleting existing registration container context for '" + NAME + "'...")
subprocess.run(
    ["ollama", "rm", NAME], 
    stdout=subprocess.DEVNULL, 
    stderr=subprocess.DEVNULL
)

# Give the Ollama background daemon a moment to release memory and file locks
time.sleep(1) 

# 6. Proceed with your working native shell compiler compilation loop
logger.log("info", "SYSTEM", "Instructing Ollama to perform a clean build for '" + NAME + "'...")
try:
    process = subprocess.Popen(
        ["ollama", "create", NAME, "-f", modelfile_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        logger.log("info", "SYSTEM", "[Ollama Build Engine]: " +line.strip())
        
    process.wait()
    
    if process.returncode == 0:
        logger.log("ok", "SYSTEM", "✅ SUCCESS: '" + NAME + "' is cleanly compiled and updated!")
    else:
        logger.log("error", "SYSTEM", "❌ CRITICAL OLLAMA BUILD FAILURE: Shell exited with code " + str(process.returncode))

except Exception as e:
    logger.log("error", "SYSTEM", "❌ CRITICAL EXECUTION EXCEPTION: " + str(e))

# =========================================================================
# AUTOMATED SELF-TEST: verify the two canonical prompts answer green / red
# =========================================================================
# Catches any residual train/inference tokenization drift instead of silently "succeeding".
logger.log("info", "SYSTEM", "Running self-test against the two canonical prompts...")
time.sleep(1)  # let Ollama finish registering the freshly-built model
all_passed = True
for prompt, expected in PAIRS:
    response = get_ollama_answer(prompt=prompt, model_name=NAME, host_url=os.environ["OLLAMA_HOST"])
    passed = expected in response.lower()
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
            print( "  /bye            Exit the interactive client session")
            print( "  /?              Show this system help description summary")
            print( "  UP/DOWN Arrows  Navigate through your previously entered prompts\n")
            readline.add_history(user_input)
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