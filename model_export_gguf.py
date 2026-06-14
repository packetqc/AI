import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gguf import GGUFWriter

#################################################################################################
#
#################################################################################################
version = "1"
model_path = "./model_version_" + version
output_path = "./model_version_" + version + ".gguf"

#################################################################################################
#
#################################################################################################
ok = True


#################################################################################################
# 1. Load your trained model and tokenizer
#################################################################################################
print("Loading Hugging Face model components...")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
state_dict = model.state_dict()

#################################################################################################
# 2. Initialize the direct GGUF file writer
#################################################################################################
writer = GGUFWriter(output_path, arch="llama")

#################################################################################################
# 3. Inject vocabulary directly from your tokenizer.json
#################################################################################################
print("Exporting tokenizer tokens...")
tokens = []
scores = []
toktypes = []

# Pull token mapping securely from the fast tokenizer backend
vocab = tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

for token_str, token_id in sorted_vocab:
    tokens.append(token_str.encode("utf-8"))
    scores.append(-float(token_id)) # Negative placeholder score for BPE order
    toktypes.append(1)              # 1 indicates standard text token format

writer.add_token_list(tokens, scores, toktypes)

#################################################################################################
# 4. Stream tensor data directly to GGUF format 
#################################################################################################
print("Writing tensors to GGUF...")
for tensor_name, tensor_data in state_dict.items():
    # Enforce FP16 configuration exactly as requested
    half_precision_tensor = tensor_data.detach().cpu().to(torch.float16).numpy()
    writer.add_tensor(tensor_name, half_precision_tensor)

#################################################################################################
# 5. Build and close GGUF matrix
#################################################################################################
writer.write_header_to_file()
writer.write_tensors_to_file()
writer.close()

#################################################################################################
#
#################################################################################################
if (ok):
    print(f"Success! {output_path} built directly from tensors and exported as gguf file.")
else:
    print(f"Error! {output_path} is not built and not exported")
