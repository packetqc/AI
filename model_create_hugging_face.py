import json
import torch
import torch.nn as nn
import torch.optim as optim

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
# for newest models
#################################################################################################
# from transformers import AutoConfig, AutoModelForCausalLM
# from transformers import AutoTokenizer
# from huggingface_hub import login
# model_id = "meta-llama/Llama-3.1-8B

#################################################################################################
#
#################################################################################################
from transformers import LlamaConfig, LlamaForCausalLM

#################################################################################################
#
#################################################################################################
version = "1"
model_path = "./model_version_" + version

#################################################################################################
#
#################################################################################################
ok = True

#################################################################################################
#
#################################################################################################
print("1. Initializing architecture parameters...")

# Initialize the default configuration
# config = LlamaConfig()

config = LlamaConfig(
    vocab_size=4,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=512,
    pad_token_id=3,
    bos_token_id=0,
    eos_token_id=3,
    attention_bias=False,
)

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
model = LlamaForCausalLM(config)

# For newest models:
# Correctly structures layers, GQA, and vocab dimensions based on the config
# model = AutoModelForCausalLM.from_config(config)

#################################################################################################
#
#################################################################################################
print("2. Applying small-variance weight initialization...") #to prevent RED saturation...
# We scale the initial random states down significantly so the softmax layer remains balanced
with torch.no_grad():
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.normal_(param, mean=0.0, std=0.01)
        if "bias" in name:
            nn.init.constant_(param, 0.0)

#################################################################################################
#
#################################################################################################
print("3. Generating clean sequential training data 0 = GREEN, 1 = YELLOW, 2 = RED ...")
# 0 = GREEN, 1 = YELLOW, 2 = RED
base_pattern = [0, 1, 2] * 40
input_ids = torch.tensor([base_pattern[:-1]], dtype=torch.long)
target_ids = torch.tensor([base_pattern[1:]], dtype=torch.long)

#################################################################################################
#
#################################################################################################
print("4. Executing stabilized weights training loop...")
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

model.train()

for epoch in range(120):
    optimizer.zero_grad()
    outputs = model(input_ids)
    
    loss = nn.functional.cross_entropy(outputs.logits.view(-1, 4), target_ids.view(-1))
    loss.backward()
    
    # FIX: Clip the gradients to keep the backpropagation updates small and steady
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1}/120 - Clean Loss: {loss.item():.6f}")

#################################################################################################
#
#################################################################################################
print("5. Saving optimized weights matrix directly...")

model.save_pretrained(model_path, safe_serialization=False)

#################################################################################################
#
#################################################################################################
if (ok):
    print("Model created and trained parameters written successfully!")
else:
    print("Error to create model and trainin it!")