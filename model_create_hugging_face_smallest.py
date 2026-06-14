import json
import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from gguf import GGUFWriter

from transformers import Qwen2Config, Qwen2TokenizerFast, Qwen2ForCausalLM
from tokenizers import Tokenizer, models

#################################################################################################
#
#################################################################################################
version = "1"

model_path = "./model_version_smallest_" + version
gguf_path = model_path+".gguf"

NAME = "model_create_hugging_face_smallest"

os.makedirs("./"+model_path, exist_ok=True)

#################################################################################################
#
#################################################################################################
ok = True

#################################################################################################
# For smallest model
#
# Core CharacteristicsZero Weights: 
#
# create an empty architecture blueprint with no actual model weights or neural network layers. 
#
# Default Metadata: 
#
# instantiate a generic backbone containing only standard Hugging Face metadata placeholders 
# like transformers_version, model_type, and basic pruning flags.
#
# Storage Footprint: 
#
# When serialized using config.save_pretrained("./"), it generates a config.json file that is less 
# than 300 bytes in size.
#################################################################################################
# from transformers import PretrainedConfig
# config = PretrainedConfig()

#################################################################################################
# for llama smallest 
#################################################################################################
# from transformers import LlamaConfig

# config = LlamaConfig(
#     vocab_size=32000,
#     hidden_size=4,
#     intermediate_size=8,
#     num_hidden_layers=1,
#     num_attention_heads=2,
#     num_key_value_heads=1
# )

#################################################################################################
# for qwen smallest 
#################################################################################################
# from transformers import Qwen2Config
# from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

# # 1. Initialize a tiny architecture (fits into standard Linux caches easily)
# config = Qwen2Config(
#     vocab_size=1000,          # Keeps vocabulary overhead low for NPU conversion later
#     hidden_size=64,           # Small but functional embedding size
#     intermediate_size=128,
#     num_hidden_layers=2,
#     num_attention_heads=4,
#     num_key_value_heads=2,
#     max_position_embeddings=256,
#     torch_dtype="float32"     # Base precision for standard conversion tools
# )

# # 2. Instantiate and save the dummy model
# model = Qwen2ForCausalLM(config)
# model.eval()
# model.save_pretrained("./linux_tiny_qwen")

# # 3. Create a matching tiny tokenizer setup
# # We use an existing qwen config placeholder to generate the local files correctly
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
# tokenizer.save_pretrained("./linux_tiny_qwen")

#################################################################################################
# for newest models
#################################################################################################
# from transformers import AutoConfig, AutoModelForCausalLM
# from transformers import AutoTokenizer
# from huggingface_hub import login
# model_id = "meta-llama/Llama-3.1-8B

#################################################################################################
#
#################################################################################################
# from transformers import LlamaConfig, LlamaForCausalLM

#################################################################################################
# INIT
#################################################################################################
print("1. Initializing architecture parameters and model...")

VOCAB_SIZE = 1000
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
CONTEXT_LENGTH = 256

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

# Save configuration metadata directly to disk
config.save_pretrained("./"+model_path)

# Initialize the default configuration
# config = LlamaConfig()

# config = LlamaConfig(
#     vocab_size=4,
#     hidden_size=256,
#     intermediate_size=512,
#     num_hidden_layers=2,
#     num_attention_heads=2,
#     num_key_value_heads=2,
#     max_position_embeddings=512,
#     pad_token_id=3,
#     bos_token_id=0,
#     eos_token_id=3,
#     attention_bias=False,
# )

# For newest models:
# Works flawlessly for Llama 1, 2, 3, or 3.1
# config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
# config = AutoConfig.from_pretrained(model_id))

# View all active default fields
print("config = ")
#print(config.to_dict())
print(json.dumps(config.to_dict(), indent=4))

# Instantiate a model shell with these exact configurations 
# Note: weights will be randomly initialized, not pretrained
# model = LlamaForCausalLM(config)

# For newest models:
# Correctly structures layers, GQA, and vocab dimensions based on the config
# model = AutoModelForCausalLM.from_config(config)


#################################################################################################
# Instantiate the model with active, trainable weights (No longer empty!)
#################################################################################################
print("Initializing model layers with active numeric weights...")
model = Qwen2ForCausalLM(config)

# 3. Create a Custom Mock Dataset for your specific phrase
# To learn hands-on, we assign hardcoded Token IDs:
# Token 10 = "green", Token 20 = "is", Token 30 = "always", Token 40 = "good", Token 0 = Pad
print("Preparing training dataset for phrase: 'green is always good'...")

# Shape: (Batch Size, Sequence Length) -> 10 repeating samples to force memorization
input_ids = torch.tensor([[10, 20, 30, 40, 0, 0, 0, 0]] * 10, dtype=torch.long)
labels = input_ids.clone()

# 4. Execute a rapid, minimal training optimization loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()

print("Starting model optimization loops...")
for epoch in range(30):  # Run 30 steps to force the small network to overfit the phrase
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/30 - Cross-Entropy Loss: {loss.item():.4f}")

# Put model back to evaluation state
model.eval()

#################################################################################################
#
#################################################################################################
# print("2. Applying small-variance weight initialization...") #to prevent RED saturation...
# print("2. No small-variance weight initialization applied or required") #to prevent RED saturation...

# # We scale the initial random states down significantly so the softmax layer remains balanced
# with torch.no_grad():
#     for name, param in model.named_parameters():
#         if "weight" in name:
#             nn.init.normal_(param, mean=0.0, std=0.01)
#         if "bias" in name:
#             nn.init.constant_(param, 0.0)

#################################################################################################
# TOKENIZING, or bypass with tensor for NPU compability
#################################################################################################
print("3. Generating tokens")
# print("3. Generating clean sequential training data 0 = GREEN, 1 = YELLOW, 2 = RED ...")

# from transformers import AutoTokenizer
# Build empty processing parameters using a Qwen structure template
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
# tokenizer.save_pretrained("./"+model_path)

# Build empty processing parameters using a Qwen structure template
# 1. Create a 100% empty local tokenizer structure
empty_tokenizer_backend = Tokenizer(models.BPE())
tokenizer = Qwen2TokenizerFast(tokenizer_object=empty_tokenizer_backend)

# Add standard special tokens required by the Qwen architecture
# tokenizer.add_special_tokens({"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>"})
tokenizer.add_special_tokens({
    "bos_token": "<|endoftext|>", 
    "eos_token": "<|endoftext|>", 
    "pad_token": "<|endoftext|>"
})

# 2. Save weightless files directly to disk (No internet requests sent)
tokenizer.save_pretrained("./"+model_path)

# # 0 = GREEN, 1 = YELLOW, 2 = RED
# base_pattern = [0, 1, 2] * 40
# input_ids = torch.tensor([base_pattern[:-1]], dtype=torch.long)
# target_ids = torch.tensor([base_pattern[1:]], dtype=torch.long)

print(f"Generating isolated tokenizer metadata table ({config.vocab_size} tokens)...")

# tokens = []
# scores = []
# token_types = []

# for i in range(config.vocab_size):
#     if i == 0:
#         tokens.append("<|endoftext|>")  # Essential structural boundary marker
#         token_types.append(2)           # 2 = Control Token
#     else:
#         tokens.append(f"[token_{i}]")   # Pure string token padding
#         token_types.append(1)           # 1 = Standard Text Token
#     scores.append(0.0)

# # Setup basic vocabulary map so Ollama can trace IDs
# tokens = []
# scores = []
# token_types = []
# for i in range(config.vocab_size):
#     if i == 0:   tokens.append("<|endoftext|>"); token_types.append(2)
#     elif i == 10: tokens.append("green");          token_types.append(1)
#     elif i == 20: tokens.append("is");             token_types.append(1)
#     elif i == 30: tokens.append("always");         token_types.append(1)
#     elif i == 40: tokens.append("good");           token_types.append(1)
#     else:        tokens.append(f"[token_{i}]");    token_types.append(1)
#     scores.append(0.0)

# Vocab Setup
tokens = ["<|endoftext|>"] + [f"[token_{i}]" for i in range(1, config.vocab_size)]
tokens[10], tokens[20], tokens[30], tokens[40] = "green", "is", "always", "good"
scores = [0.0] * config.vocab_size
token_types = [1] * config.vocab_size
token_types[0] = 2 # Control token for EOS

#################################################################################################
#
#################################################################################################
# print("4. Executing stabilized weights training loop...")

# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# model.train()
# for epoch in range(120):
#     optimizer.zero_grad()
#     outputs = model(input_ids) 
#     loss = nn.functional.cross_entropy(outputs.logits.view(-1, 4), target_ids.view(-1))
#     loss.backward()
#     # FIX: Clip the gradients to keep the backpropagation updates small and steady
#     nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     optimizer.step()
#     if (epoch + 1) % 20 == 0:
#         print(f"   Epoch {epoch+1}/120 - Clean Loss: {loss.item():.6f}")

#################################################################################################
#
#################################################################################################
# print("5. Saving optimized weights matrix directly...")

# model.save_pretrained(model_path, safe_serialization=False)

#################################################################################################
# GGUF FILE EXPORT
#################################################################################################
print("4. Exporting to GGUF file...")

# 'qwen2' is the internal string keyword llama.cpp uses to map architectural logic
writer = GGUFWriter(gguf_path, arch=config.model_type)

print(f"Constructing empty Qwen architecture binary layout...")

# 3. Inject Mandatory Structural Metadata
writer.add_architecture()
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
writer.add_token_list(tokens)
writer.add_token_scores(scores)
writer.add_token_types(token_types)

print("Packing trained tensor matrices into GGUF format...")
state_dict = model.state_dict()

# name_mapping = {
#     "model.embed_tokens.weight": "token_embd.weight",
#     "model.layers.0.input_layernorm.weight": "blk.0.attn_norm.weight",
#     "model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight",
#     "model.layers.0.self_attn.k_proj.weight": "blk.0.attn_k.weight",
#     "model.layers.0.self_attn.v_proj.weight": "blk.0.attn_v.weight",
#     "model.layers.0.self_attn.o_proj.weight": "blk.0.attn_output.weight",
#     "model.layers.0.post_attention_layernorm.weight": "blk.0.ffn_norm.weight",
#     "model.layers.0.mlp.gate_proj.weight": "blk.0.ffn_gate.weight",
#     "model.layers.0.mlp.up_proj.weight": "blk.0.ffn_up.weight",
#     "model.layers.0.mlp.down_proj.weight": "blk.0.ffn_down.weight",
#     "model.layers.1.input_layernorm.weight": "blk.1.attn_norm.weight",
#     "model.layers.1.self_attn.q_proj.weight": "blk.1.attn_q.weight",
#     "model.layers.1.self_attn.k_proj.weight": "blk.1.attn_k.weight",
#     "model.layers.1.self_attn.v_proj.weight": "blk.1.attn_v.weight",
#     "model.layers.1.self_attn.o_proj.weight": "blk.1.attn_output.weight",
#     "model.layers.1.post_attention_layernorm.weight": "blk.1.ffn_norm.weight",
#     "model.layers.1.mlp.gate_proj.weight": "blk.1.ffn_gate.weight",
#     "model.layers.1.mlp.up_proj.weight": "blk.1.ffn_up.weight",
#     "model.layers.1.mlp.down_proj.weight": "blk.1.ffn_down.weight",
#     "model.norm.weight": "output_norm.weight",
#     "lm_head.weight": "output.weight"
# }

# # Write individual weights safely converted to standard Float32 arrays
# for torch_name, gguf_name in name_mapping.items():
#     if torch_name in state_dict:
#         tensor_np = state_dict[torch_name].detach().cpu().numpy().astype(np.float32)
#         writer.add_tensor(gguf_name, tensor_np)

tensor_mapping = {
    # Base Embeddings
    "model.embed_tokens.weight": "token_embd.weight",
    
    # Layer 0
    "model.layers.0.input_layernorm.weight": "blk.0.attn_norm.weight",
    "model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight",
    "model.layers.0.self_attn.q_proj.bias":   "blk.0.attn_q.bias",     # <-- Added Bias
    "model.layers.0.self_attn.k_proj.weight": "blk.0.attn_k.weight",
    "model.layers.0.self_attn.k_proj.bias":   "blk.0.attn_k.bias",     # <-- Added Bias
    "model.layers.0.self_attn.v_proj.weight": "blk.0.attn_v.weight",
    "model.layers.0.self_attn.v_proj.bias":   "blk.0.attn_v.bias",     # <-- Added Bias
    "model.layers.0.self_attn.o_proj.weight": "blk.0.attn_output.weight",
    "model.layers.0.post_attention_layernorm.weight": "blk.0.ffn_norm.weight",
    "model.layers.0.mlp.gate_proj.weight": "blk.0.ffn_gate.weight",
    "model.layers.0.mlp.up_proj.weight": "blk.0.ffn_up.weight",
    "model.layers.0.mlp.down_proj.weight": "blk.0.ffn_down.weight",
    
    # Layer 1
    "model.layers.1.input_layernorm.weight": "blk.1.attn_norm.weight",
    "model.layers.1.self_attn.q_proj.weight": "blk.1.attn_q.weight",
    "model.layers.1.self_attn.q_proj.bias":   "blk.1.attn_q.bias",     # <-- Added Bias
    "model.layers.1.self_attn.k_proj.weight": "blk.1.attn_k.weight",
    "model.layers.1.self_attn.k_proj.bias":   "blk.1.attn_k.bias",     # <-- Added Bias
    "model.layers.1.self_attn.v_proj.weight": "blk.1.attn_v.weight",
    "model.layers.1.self_attn.v_proj.bias":   "blk.1.attn_v.bias",     # <-- Added Bias
    "model.layers.1.self_attn.o_proj.weight": "blk.1.attn_output.weight",
    "model.layers.1.post_attention_layernorm.weight": "blk.1.ffn_norm.weight",
    "model.layers.1.mlp.gate_proj.weight": "blk.1.ffn_gate.weight",
    "model.layers.1.mlp.up_proj.weight": "blk.1.ffn_up.weight",
    "model.layers.1.mlp.down_proj.weight": "blk.1.ffn_down.weight",
    
    # Heads
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight"
}

# 7. Write complete matrices directly into GGUF container
print("Writing full weight and bias matrices into the GGUF binary...")
for torch_name, gguf_name in tensor_mapping.items():
    if torch_name in state_dict:
        tensor_np = state_dict[torch_name].detach().cpu().numpy().astype(np.float32)
        writer.add_tensor(gguf_name, tensor_np)


# Finalize file serialization
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()

writer.close()

print(f"'{gguf_path}' was created with zero errors.")
print(f"File size: {os.path.getsize(gguf_path) / 1024:.2f} KB")

#################################################################################################
# INSPECT INTERNAL BINARY TAGS OF THE GGUF FILE
#################################################################################################
from gguf import GGUFReader

# 1. Point to your custom binary file
print(f"Opening {gguf_path} for binary parsing...\n")

# 2. Initialize the reader engine
reader = GGUFReader(gguf_path)

print("=== GGUF Container Info ===")
print(f"Total Metadata KV Pairs: {len(reader.fields)}")
print(f"Total Tensors Encoded  : {len(reader.tensors)}")
print("=" * 27 + "\n")

print("=== Decoded Internal Metadata Tags ===")

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
        print(f"{key:<40} : [Truncated... {len(field.parts)} tokens encoded]")
    elif "tokenizer.ggml.scores" in key or "tokenizer.ggml.token_type" in key:
        continue
    else:
        print(f"{key:<40} : {val}")

print("=" * 38)



#################################################################################################
#
#################################################################################################
if (ok):
    print("\nProgram completed.")
    # print("Model created and trained parameters written successfully!")
else:
    print("\nError to create model and trainin it!")