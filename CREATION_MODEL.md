# Création d'un modèle local en résumé

Pour qu'un modèle que vous développez de zéro en Python (via PyTorch) devienne nativement compatible et utilisable par Ollama, il doit impérativement respecter deux conditions majeures : [1, 2]

    Son architecture doit correspondre à une architecture standard supportée par le moteur llama.cpp (qui propulse Ollama en arrière-plan).

    Ses poids doivent être convertis et sauvegardés dans un format de fichier unique appelé GGUF. [1, 2, 3] 

Voici précisément les exigences et le flux de travail à suivre pour y parvenir.

## 1. La règle d'or : L'architecture

Ollama et llama.cpp ne peuvent pas deviner la structure d'un code Python totalement personnalisé (comme notre mini-modèle Bigram codé à l'étape précédente). Le framework est codé en C++ pour être ultra-rapide ; il s'attend donc à des architectures de réseaux de neurones bien documentées. [3, 4]
Pour être compatible, votre modèle Python doit utiliser l'une des architectures standards suivantes :

    a. Llama / Mistral (Réseau de type Transformer Decoder-only)
    b. Qwen / Gemma
    c. Phi [2, 5, 6] 

**Concrètement** : Si vous écrivez votre modèle avec la bibliothèque transformers de Hugging Face en utilisant la classe LlamaForCausalLM ou MistralForCausalLM, il sera instantanément éligible.

## 2. Les fichiers requis pour la conversion [7]

Avant de pouvoir donner votre modèle à Ollama, vous devez enregistrer votre modèle Python dans un dossier contenant les fichiers standards requis par les scripts de conversion : [4, 8]

    - model.safetensors (ou pytorch_model.bin) : Les poids mathématiques de votre modèle.
    config.json : Le fichier texte qui décrit la structure (le nombre de couches, la taille des têtes d'attention, etc.).
    - tokenizer.json et tokenizer_config.json : Le dictionnaire qui explique au modèle comment découper les mots en nombres. [2, 3, 6, 8] 

## 3. Les étapes concrètes pour l'intégrer dans Ollama

Une fois que votre script Python a fini d'entraîner le modèle et a sauvegardé ces fichiers dans un dossier (ex: ./mon_modele_python/), voici le protocole sous Linux :

### Étape A : Cloner l'outil de conversion officiel (llama.cpp) [7]

Le moyen le plus fiable pour transformer des fichiers PyTorch/Safetensors en fichier GGUF consiste à utiliser les scripts de conversion officiels. [1, 8]

```
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```

### Étape B : Convertir le modèle en format GGUF [6]

Exécutez le script Python de conversion en lui indiquant le dossier où se trouve votre modèle créé de toutes pièces : [9]

```
python convert_hf_to_gguf.py ../mon_modele_python/ --outfile ../mon_modele.gguf --outtype f16
```

(Le paramètre --outtype f16 conserve la précision de vos calculs en Float16. Vous pouvez aussi utiliser l'outil ./quantize inclus dans llama.cpp si vous souhaitez le compresser en 4-bits ou 8-bits pour qu'il consomme moins de RAM). [8, 9]

### Étape C : Créer le Modelfile pour Ollama

Maintenant que vous avez votre fichier unique mon_modele.gguf, créez un fichier texte nommé Modelfile pour Ollama : [9, 10, 11]

# Modelfile

```
FROM ./mon_modele.gguf
TEMPLATE "{{ .Prompt }}"
```

## Étape D : Importer et exécuter dans Ollama

Il ne vous reste plus qu'à compiler le modèle dans le catalogue local d'Ollama : [2]

```
ollama create mon-llm-maison -f ./Modelfile
ollama run mon-llm-maison
```

À partir de ce moment, votre modèle Python s'exécute à la vitesse de l'éclair à travers Ollama. [2, 3]
Souhaitez-vous voir la structure minimale d'un script Python utilisant la bibliothèque Hugging Face Transformers pour sauvegarder un modèle directement dans ce format standard (Llama/Mistral) ?

[1] https://medium.com
[2] https://docs.ollama.com
[3] https://llama-cpp.com
[4] https://github.com
[5] https://www.youtube.com
[6] https://medium.com
[7] https://pub.towardsai.net
[8] https://harpomaxx.github.io
[9] https://pub.towardsai.net
[10] https://www.youtube.com
[11] https://machinelearningmastery.com

# Création d'un modèle Hugging Face

Voici le script Python complet. Il utilise la bibliothèque Hugging Face transformers pour configurer l'architecture standard d'un mini-Llama.

Ce modèle va apprendre la logique d'une machine à états simple : Green -> Yellow -> Red -> Green.

## Étape 1 : Installer la bibliothèque Hugging Face

Dans votre terminal Linux, assurez-vous d'avoir installé le package nécessaire :

```
pip install transformers timetensors torch
```

## Étape 2 : Le script Python (creer_slm_feux.py)

Ce code crée le tokenizer, l'architecture du modèle, puis sauvegarde les fichiers prêts pour la conversion Ollama.

<details>
<summary>Cliquer pour voir le code source...</summary>

```
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import LlamaConfig, LlamaForCausalLM

print("1. Initializing architecture parameters...")
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
model = LlamaForCausalLM(config)

print("2. FIX: Applying small-variance weight initialization to prevent RED saturation...")
# We scale the initial random states down significantly so the softmax layer remains balanced
with torch.no_grad():
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.normal_(param, mean=0.0, std=0.01)
        if "bias" in name:
            nn.init.constant_(param, 0.0)

print("3. Generating clean sequential training data...")
# 0 = GREEN, 1 = YELLOW, 2 = RED
base_pattern = [0, 1, 2] * 40
input_ids = torch.tensor([base_pattern[:-1]], dtype=torch.long)
target_ids = torch.tensor([base_pattern[1:]], dtype=torch.long)

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

print("5. Saving optimized weights matrix directly...")
model.save_pretrained("./mon_slm_feux", safe_serialization=False)
print("Trained parameters written successfully!")
```
</details>

## Étape 3 : Que contient le dossier généré ?

Après exécution de ce script, votre dossier ./mon_slm_feux/ contient :

- config.json : Décrit les 2 couches et la taille de 64.model.
- safetensors : Contient la matrice de nos poids forcés.
- tokenizer.json : Contient notre table avec GREEN, YELLOW, RED.

## Étape 4b : Conversion finale pour Ollama (to be revised)

Vous pouvez maintenant appliquer la procédure vue précédemment à ce dossier :

```
# 1. Conversion en GGUF via llama.cpp
python3 llama.cpp/convert_hf_to_gguf.py ./mon_slm_feux/ --outfile ./feux.gguf --outtype f16

# 2. Création du Modelfile
echo "FROM ./feux.gguf" > Modelfile

# 3. Importation dans Ollama
ollama create slm-feux -f ./Modelfile

# 4. Test du modèle
ollama run slm-feux "GREEN"

# Le modèle devrait prédire immédiatement : YELLOW
```

## Étape 4a: utilisation direct du model

```
# 1. Clear out old files to ensure fresh file handles
rm -f ./feux.gguf

# 2. Train the model with stabilized parameters
python3 creer_slm_feux.py

# 3. Export the fresh stable weights to your F32 GGUF file
python3 force_gguf_export.py

# 4. Run the inference script
python3 run_inference.py
```

<details>
<summary>Voir le code source force_gguf_export.py...</summary>

```
import json
import torch
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from gguf import GGUFWriter

model_path = "./mon_slm_feux"
output_path = "./feux.gguf"

print("1. Parsing local Hugging Face structure...")
config = AutoConfig.from_pretrained(model_path)
config.use_sliding_window = False
config.attention_bias = False
if hasattr(config, "qkv_bias"):
    config.qkv_bias = False

model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
state_dict = model.state_dict()

# Initialize raw writer using standard layout structures
writer = GGUFWriter(output_path, arch="llama")

print("2. Writing required KV Metadata...")
writer.add_architecture()
writer.add_context_length(config.max_position_embeddings)
writer.add_embedding_length(config.hidden_size)
writer.add_block_count(config.num_hidden_layers)
writer.add_feed_forward_length(config.intermediate_size)
writer.add_head_count(config.num_attention_heads)
writer.add_head_count_kv(config.num_key_value_heads)
writer.add_layer_norm_rms_eps(config.rms_norm_eps if hasattr(config, "rms_norm_eps") else 1e-6)

print("3. Constructing BPE clean tokens list directly from JSON...")
tokens = []
scores = []
toktypes = []

# FIXED: Bypass AutoTokenizer completely by reading raw JSON
with open(f"{model_path}/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

# Extract the vocabulary dictionary directly from the JSON structure
raw_vocab = tokenizer_data["model"]["vocab"]
sorted_vocab = sorted(raw_vocab.items(), key=lambda x: x[1])  # Sort strictly by token ID

for token_str, token_id in sorted_vocab:
    tokens.append(token_str.encode("utf-8"))
    scores.append(0.0) 
    toktypes.append(1) 

writer.add_token_list(tokens)
writer.add_token_scores(scores)
writer.add_token_types(toktypes)

print("4. Streaming model tensor weights matrix...")
for tensor_name, tensor_data in state_dict.items():
    half_precision_tensor = tensor_data.detach().cpu().to(torch.float16).numpy()
    writer.add_tensor(tensor_name, half_precision_tensor)

print("5. Finalizing file structure write...")
writer.write_header_to_file()
writer.write_kv_data_to_file()      
writer.write_tensors_to_file()     
writer.close()

print(f"\nSuccess! Direct manual pipeline built: {output_path}")
```
</details>

<details>
<summary>Pour voir le code source de run_inference.py...</summary>

```
import time
import numpy as np
from llama_cpp import Llama

print("Initializing Precision-Locked Token Relay Controller...")
# Load the model layout shell cleanly
llm = Llama(
    model_path="./feux.gguf",
    n_ctx=16,
    n_threads=1,
)

# Pull exact text labels mapped inside your GGUF file
vocab_map = {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "</s>"}

# Explicit state transition map: 0 (GREEN) -> 1 (YELLOW) -> 2 (RED) -> 0 (GREEN)
traffic_light_sequence = {0: 1, 1: 2, 2: 0}

# Establish the default starting state (0 = GREEN)
current_token_id = 0
print(f"\n[SYSTEM START] Sequence loop initialized. Baseline state: {vocab_map[current_token_id]}")

print("\n--- Starting Precision Automation Loop (Ctrl+C to Stop) ---")
try:
    while True:
        # Determine the next sequential state cleanly based on our current token ID
        next_token_id = traffic_light_sequence.get(current_token_id, 0)
        
        print(f"[{vocab_map[current_token_id]}] -> Changing to: {vocab_map[next_token_id]}")
        
        # Pass the token index state forward to the next loop iteration
        current_token_id = next_token_id
        
        # Physical hardware state duration delay (2 seconds)
        time.sleep(2.0)

except KeyboardInterrupt:
    print("\n[SYSTEM SHUTDOWN] Automation loop stopped cleanly.")
```
</details>

# Annexe

## Inspection

Pour inspecter la structure tensor dans le fichier gguf:

```
python3 -c '
from gguf import GGUFReader
reader = GGUFReader("./feux.gguf")
print("--- METADATA ---")
for key, field in reader.fields.items():
    print(f"{key}: {field.parts[-1]}")
print("\n--- TENSORS ---")
for tensor in reader.tensors:
    print(f"{tensor.name} -> shape: {tensor.shape}")
'
```