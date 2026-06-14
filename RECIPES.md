# METHODOLOGIES

## CONCEPTIONS

### MODEL
1. Ultra small model GREEN, YELLOW and RED. Python, torch, Llama, example:
- **model_create_hugging_face.py**

### EXPORT AND CONVERSION FOR GGUF FILE
1. Python, examples:
- export_gguf.py
2. Ollama, examples:
- [cmd line tbc here]

## EXPLOITATIONS

### AI BACKEND
1. OLLAMA + EXTERNAL MODEL
2. OLLAMA + LOCAL MODEL

### AI AGENT
1. Python for local model file, examples:
- ai_agent_fsm.py
- traffic_controller_api.py
- run_inference.py
2. Python for ollama backend, examples:
- run_fsm_agent.py
- diret_fsm_agent.py
- stateless_fsm_agent.py
- large_model_agent.py

## FORENSICS AND ANALYSIS

# CREATE MODEL

# TRAIN MODEL

# IMPORT MODEL TO OLLAMA
```
# 1. Create your local engine manifest blueprint
echo "FROM ./empty_qwen.gguf" > Modelfile

# 2. Register the compiled asset into the Ollama system environment
ollama create hands_on_qwen -f ./Modelfile

# 3. Confirm it shows up in your active engine listings
ollama list
```
# RUN MODEL
```
ollama run hands_on_qwen
```