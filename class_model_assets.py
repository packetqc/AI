import os
import json
import time
import subprocess
import urllib.request

import numpy as np
from gguf import GGUFWriter, GGUFReader


class ModelAssets:
    """Owns the "knowledge" side of the tiny Qwen2 model.

    It accumulates markdown documents and, via a ``builder`` callback
    (``build_pipeline`` in ``model_create_hf_cl.py``), REBUILDS the tokenizer +
    model over the green/red anchors plus all accumulated knowledge whenever a
    new file is learned. It then re-exports the GGUF and rebuilds the Ollama
    model so the change is actually served. Two entry points are used:

      * at start-up        -> the caller builds once and constructs ModelAssets
                              with the initial ``artifacts``; then calls
                              ``export_and_rebuild()``.
      * in-flight (/read)  -> ``learn_document(path)`` appends the document (markdown or JSON),
                              rebuilds the tokenizer + model (so the new words
                              become first-class tokens — robust recall) and
                              rebuilds the Ollama model on the fly.

    Rebuilding over the FULL accumulated set every time makes catastrophic
    forgetting impossible: the anchors and every prior document are retrained
    together.
    """

    initiated = False
    STATE_VERSION = 1

    def __init__(self, *, builder, knowledge_texts, playbook, artifacts, arch, vocab_cap,
                 gguf_path, modelfile_path, ollama_name, ollama_host,
                 modelfile_template, num_predict=8, grammar_name=None, state_path=None, logger=None):
        self.builder = builder                        # build_pipeline(knowledge_texts, grammar_pairs, arch, vocab_cap)
        self.knowledge_texts = list(knowledge_texts)  # accumulated prose documents (markdown / generic JSON)
        # Playbook: roots -> nodes -> leaves tree (navigation source of truth). The training pairs
        # below are DERIVED from it and trained like the green/red anchors (prompt masked, answer
        # trained), so a stacked path prompt returns that node's children or the leaf's commands.
        self.playbook = dict(playbook or {})
        self.grammar_name = grammar_name
        self._refresh_grammar_pairs()
        self.arch = dict(arch)                        # architecture dims (persisted)
        self.vocab_cap = vocab_cap                    # current BPE vocab ceiling (grows adaptively)

        # serving / file locations
        self.gguf_path = gguf_path
        self.modelfile_path = modelfile_path
        self.ollama_name = ollama_name
        self.ollama_host = ollama_host
        self.modelfile_template = modelfile_template
        self.num_predict = num_predict
        self.state_path = state_path
        self.logger = logger

        self._set_artifacts(artifacts)
        self.initiated = True

    def _refresh_grammar_pairs(self):
        """Recompute the flattened training pairs from the current playbook tree."""
        self.grammar_pairs = self.flatten_playbook(self.playbook, self.grammar_name)

    def _set_artifacts(self, a):
        """Adopt the model/tokenizer/export-metadata produced by the builder."""
        self.model = a["model"]
        self.tokenizer = a["tokenizer"]
        self.config = a["config"]
        self.tokens = a["tokens"]
        self.scores = a["scores"]
        self.token_types = a["token_types"]
        self.merges = a["merges"]
        self.eos_id = a["eos_id"]
        # The builder reports the cap it actually used (it may have grown to fit new vocabulary).
        self.vocab_cap = a.get("vocab_cap", self.vocab_cap)
        self.arch = a.get("arch", self.arch)

    # --------------------------------------------------------------- persistence
    @staticmethod
    def load_state(path):
        """Load a previously-saved state dict (knowledge + config), or None if absent/invalid."""
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save_state(self):
        """Persist accumulated knowledge + the (possibly adapted) config for the next reload."""
        if not self.state_path:
            return
        state = {
            "version": self.STATE_VERSION,
            "ollama_name": self.ollama_name,
            "arch": self.arch,
            "vocab_cap": self.vocab_cap,
            "vocab_size": len(self.tokens),
            "knowledge_texts": self.knowledge_texts,
            "playbook": self.playbook,
            "grammar_name": self.grammar_name,
        }
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            self._log("ok", "Saved state (" + str(len(self.knowledge_texts)) + " doc(s), vocab_cap="
                      + str(self.vocab_cap) + ") to " + self.state_path)
        except Exception as e:
            self._log("error", "Could not save state to " + str(self.state_path) + ": " + str(e))

    # ------------------------------------------------------------------ utils
    def _log(self, severity, msg):
        if self.logger is not None:
            self.logger.log(severity, "ASSETS", msg)
        else:
            print(f"[{severity.upper()}] [ASSETS] {msg}")

    # ------------------------------------------------------------------ read
    @staticmethod
    def json_to_text(obj):
        """Flatten a parsed JSON value into plain "The <key path> is <value>." sentences so it
        feeds the same recall pipeline as markdown prose. Nested objects extend the key path;
        lists of scalars become an "are a, b, c" sentence; lists of objects recurse per item."""
        lines = []

        def humanize(k):
            return str(k).replace("_", " ").replace("-", " ").strip()

        def walk(node, prefix):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, prefix + [humanize(k)])
            elif isinstance(node, list):
                scalars = [x for x in node if not isinstance(x, (dict, list))]
                if scalars and prefix:
                    lines.append("The " + " ".join(prefix) + " are "
                                 + ", ".join(str(x) for x in scalars) + ".")
                for i, x in enumerate(node):
                    if isinstance(x, (dict, list)):
                        walk(x, prefix + ["item " + str(i + 1)])
            else:
                key = " ".join(prefix) if prefix else "value"
                lines.append("The " + key + " is " + str(node) + ".")

        walk(obj, [])
        return "\n".join(lines)

    # ----------------------------------------------------------- playbook tree
    # A "playbook" is a tree of roots -> nodes -> leaves (implicit-nested schema):
    #   dict   -> internal node (its keys are the children)
    #   list   -> command leaf  (executable shell commands)
    #   string -> answer leaf   (plain text, not executed)
    # The legacy flat {"routines": {name: [cmds]}} form is just a one-level tree.
    _GRAMMAR_CONTAINERS = ("playbook", "tree", "nodes", "routines")

    @staticmethod
    def playbook_tree_from_json(obj):
        """Return the normalized playbook subtree dict from a grammar file, or None if ``obj`` is
        not a grammar. Merges whichever container keys are present (playbook/tree/nodes/routines)."""
        if not isinstance(obj, dict):
            return None
        tree = {}
        for key in ModelAssets._GRAMMAR_CONTAINERS:
            sub = obj.get(key)
            if isinstance(sub, dict):
                ModelAssets.merge_tree(tree, sub)
        return tree or None

    @staticmethod
    def merge_tree(dst, src):
        """Deep-merge playbook tree ``src`` into ``dst`` (later files extend/override)."""
        for k, v in src.items():
            k = str(k)
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                ModelAssets.merge_tree(dst[k], v)
            else:
                dst[k] = v
        return dst

    # Node-listing answers are prefixed with this marker so a child name never sits DIRECTLY
    # after its parent path token. Without it, the listing "diagnostics disk, load" collides with
    # the leaf path "diagnostics disk" (same context, different next token) and floors the loss.
    NODE_LISTING_PREFIX = "routes: "

    @staticmethod
    def flatten_playbook(tree, grammar_name=None):
        """Flatten a playbook tree into (prompt, answer) anchor pairs:
          * internal node -> ("<path>", "routes: <child1>, <child2>, ...")   (navigation listing)
          * list leaf     -> ("<path>", "<cmd1>, <cmd2>, ...")               (commands)
          * string leaf   -> ("<path>", "<text>")                            (answer)
        If ``grammar_name`` is given, also emit ("<name>", "routes: <root1>, ...")."""
        pairs = []
        prefix = ModelAssets.NODE_LISTING_PREFIX

        def walk(node, path):
            if isinstance(node, dict):
                if path:
                    pairs.append((" ".join(path), prefix + ", ".join(str(k) for k in node.keys())))
                for k, v in node.items():
                    walk(v, path + [str(k)])
            elif isinstance(node, list):
                pairs.append((" ".join(path), ", ".join(str(c) for c in node)))
            else:
                pairs.append((" ".join(path), str(node)))

        walk(tree, [])
        if grammar_name and isinstance(tree, dict) and tree:
            pairs.insert(0, (str(grammar_name), prefix + ", ".join(str(k) for k in tree.keys())))
        # Drop empties / dedupe keeping the last answer for a prompt.
        seen = {}
        for p, a in pairs:
            if p.strip() and a.strip():
                seen[p] = a
        return [(p, a) for p, a in seen.items()]

    @staticmethod
    def resolve_path(tree, path):
        """Walk ``tree`` by the list of keys in ``path``; return (node, found)."""
        node = tree
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return None, False
        return node, True

    def read_document(self, path):
        """Read a knowledge document and return a typed dict, or ``None`` on any error:

          * grammar JSON (playbook/tree/nodes/routines) -> {"kind": "grammar", "tree": {...}}
          * other JSON                                  -> {"kind": "prose", "text": <facts>}
          * markdown / other                            -> {"kind": "prose", "text": <raw markup>}
        """
        if not path:
            return None
        if not os.path.isfile(path):
            self._log("error", "File not found: " + str(path))
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception as e:
            self._log("error", "Could not read " + str(path) + ": " + str(e))
            return None
        if not raw.strip():
            self._log("warning", "File is empty: " + str(path))
            return None

        if path.lower().endswith(".json"):
            try:
                obj = json.loads(raw)
            except Exception as e:
                self._log("error", "Invalid JSON in " + str(path) + ": " + str(e))
                return None
            tree = self.playbook_tree_from_json(obj)
            if tree is not None:
                name = obj.get("grammar") if isinstance(obj, dict) else None
                pair_count = len(self.flatten_playbook(tree, name))
                self._log("info", "Read playbook JSON " + str(path) + " -> "
                          + str(len(tree)) + " root(s), " + str(pair_count) + " trained pair(s).")
                return {"kind": "grammar", "tree": tree, "grammar_name": name}
            text = self.json_to_text(obj)
            if not text.strip():
                self._log("warning", "JSON produced no learnable facts: " + str(path))
                return None
            self._log("info", "Read JSON " + str(path) + " -> "
                      + str(len(text.splitlines())) + " fact sentence(s).")
            return {"kind": "prose", "text": text}

        self._log("info", "Read " + str(len(raw)) + " characters from " + str(path))
        return {"kind": "prose", "text": raw}

    # Backwards-compatible alias (prose only).
    def read_markdown(self, path):
        doc = self.read_document(path)
        return doc["text"] if doc and doc.get("kind") == "prose" else None

    # --------------------------------------------------------- GGUF + Ollama
    def export_gguf(self):
        """Write the current model weights + fixed tokenizer to ``self.gguf_path``."""
        self._log("info", "Exporting current model weights to GGUF...")
        writer = GGUFWriter(self.gguf_path, arch=self.config.model_type)

        writer.add_name(self.ollama_name)
        writer.add_context_length(self.config.max_position_embeddings)
        writer.add_embedding_length(self.config.hidden_size)
        writer.add_block_count(self.config.num_hidden_layers)
        writer.add_feed_forward_length(self.config.intermediate_size)
        writer.add_head_count(self.config.num_attention_heads)
        writer.add_head_count_kv(self.config.num_key_value_heads)
        writer.add_layer_norm_rms_eps(self.config.rms_norm_eps)

        # Tokenizer block (must match the Qwen2 regex the tokenizer was trained with).
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("qwen2")
        writer.add_token_list(self.tokens)
        writer.add_token_scores(self.scores)
        writer.add_token_types(self.token_types)
        writer.add_bos_token_id(self.eos_id)
        writer.add_eos_token_id(self.eos_id)
        writer.add_pad_token_id(self.eos_id)
        writer.add_add_bos_token(False)   # template already supplies the leading <|endoftext|>
        writer.add_array("tokenizer.ggml.merges", self.merges)

        # Map HF tensor names -> llama.cpp tensor names.
        state_dict = self.model.state_dict()
        tensor_mapping = {"model.embed_tokens.weight": "token_embd.weight"}
        for i in range(self.config.num_hidden_layers):
            tensor_mapping.update({
                f"model.layers.{i}.input_layernorm.weight":          f"blk.{i}.attn_norm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight":         f"blk.{i}.attn_q.weight",
                f"model.layers.{i}.self_attn.q_proj.bias":           f"blk.{i}.attn_q.bias",
                f"model.layers.{i}.self_attn.k_proj.weight":         f"blk.{i}.attn_k.weight",
                f"model.layers.{i}.self_attn.k_proj.bias":           f"blk.{i}.attn_k.bias",
                f"model.layers.{i}.self_attn.v_proj.weight":         f"blk.{i}.attn_v.weight",
                f"model.layers.{i}.self_attn.v_proj.bias":           f"blk.{i}.attn_v.bias",
                f"model.layers.{i}.self_attn.o_proj.weight":         f"blk.{i}.attn_output.weight",
                f"model.layers.{i}.post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
                f"model.layers.{i}.mlp.gate_proj.weight":            f"blk.{i}.ffn_gate.weight",
                f"model.layers.{i}.mlp.up_proj.weight":              f"blk.{i}.ffn_up.weight",
                f"model.layers.{i}.mlp.down_proj.weight":            f"blk.{i}.ffn_down.weight",
            })
        tensor_mapping.update({
            "model.norm.weight": "output_norm.weight",
            "lm_head.weight":    "output.weight",
        })

        for torch_name, gguf_name in tensor_mapping.items():
            if torch_name in state_dict:
                tensor_np = state_dict[torch_name].detach().cpu().numpy().astype(np.float32)
                writer.add_tensor(gguf_name, tensor_np)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        self._log("ok", "GGUF written: " + self.gguf_path
                  + " (" + str(round(os.path.getsize(self.gguf_path) / 1024, 1)) + " KB)")

    def inspect_gguf(self):
        """Log the metadata tags packed into the GGUF file (diagnostic)."""
        reader = GGUFReader(self.gguf_path)
        self._log("info", "GGUF KV pairs: " + str(len(reader.fields))
                  + ", tensors: " + str(len(reader.tensors)))
        for key, field in reader.fields.items():
            if "tokenizer.ggml.tokens" in key:
                self._log("info", key + " : [" + str(len(field.parts)) + " token parts]")
                continue
            if "tokenizer.ggml.scores" in key or "tokenizer.ggml.token_type" in key \
                    or "tokenizer.ggml.merges" in key:
                continue
            raw_part = field.parts[field.data[0]]
            if isinstance(raw_part, np.ndarray) and raw_part.dtype.kind in ['U', 'S', 'O']:
                val = raw_part.item()
                if isinstance(val, bytes):
                    val = val.decode('utf-8', errors='ignore')
            elif isinstance(raw_part, bytes):
                val = raw_part.decode('utf-8', errors='ignore')
            else:
                val = raw_part[0] if hasattr(raw_part, '__len__') else raw_part
            self._log("info", key + " : " + str(val))

    def _write_modelfile(self):
        absolute_gguf_path = os.path.abspath(self.gguf_path)
        content = (
            "FROM " + absolute_gguf_path + "\n\n"
            'TEMPLATE """' + self.modelfile_template + '"""\n\n'
            'PARAMETER stop "<|endoftext|>"\n'
            "PARAMETER temperature 0.0\n"
            "PARAMETER num_predict " + str(self.num_predict) + "\n"
        )
        with open(self.modelfile_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        self._log("ok", "Wrote " + self.modelfile_path + " pointing at " + self.gguf_path)

    def rebuild_ollama(self):
        """Unload, remove, and re-create the Ollama model from the current GGUF."""
        self._write_modelfile()

        # Force-unload any running instance so the file lock is released.
        try:
            payload = json.dumps({"model": self.ollama_name, "keep_alive": 0}).encode("utf-8")
            req = urllib.request.Request(
                self.ollama_host + "/api/generate",
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=3):
                pass
        except Exception:
            pass  # idle/closed daemon is fine

        subprocess.run(["ollama", "rm", self.ollama_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)  # let the daemon release memory and file locks

        self._log("info", "Rebuilding Ollama model '" + self.ollama_name + "'...")
        try:
            process = subprocess.Popen(
                ["ollama", "create", self.ollama_name, "-f", self.modelfile_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            for line in process.stdout:
                self._log("info", "[Ollama] " + line.strip())
            process.wait()
            if process.returncode == 0:
                self._log("ok", "✅ '" + self.ollama_name + "' rebuilt and updated.")
                return True
            self._log("error", "❌ Ollama build failed with code " + str(process.returncode))
        except Exception as e:
            self._log("error", "❌ Ollama build exception: " + str(e))
        return False

    def export_and_rebuild(self):
        """Export the GGUF and rebuild the served Ollama model."""
        self.export_gguf()
        return self.rebuild_ollama()

    # ------------------------------------------------------ high-level entry
    def learn_document(self, path, rebuild=True):
        """Read a knowledge file (markdown or JSON), REBUILD the tokenizer + model over the
        anchors plus all accumulated knowledge (growing the vocabulary if the new file needs it),
        persist the state, and optionally re-export the GGUF + rebuild the Ollama model.

        Returns ``True`` on success. With ``rebuild=False`` the artifacts are updated in
        memory only (the caller is responsible for the export).
        """
        doc = self.read_document(path)
        if doc is None:
            return False
        if doc["kind"] == "grammar":
            self.merge_tree(self.playbook, doc["tree"])     # deep-merge the new branch in
            if doc.get("grammar_name"):
                self.grammar_name = doc["grammar_name"]
            self._refresh_grammar_pairs()                   # re-derive the training pairs
            self._log("info", "Merged playbook from '" + str(path) + "'; rebuilding ("
                      + str(len(self.grammar_pairs)) + " grammar pair(s), "
                      + str(len(self.knowledge_texts)) + " prose doc(s) total)...")
        else:
            self.knowledge_texts.append(doc["text"])
            self._log("info", "Rebuilding tokenizer + model to cover '" + str(path)
                      + "' (" + str(len(self.knowledge_texts)) + " prose doc(s) total)...")
        prev_cap = self.vocab_cap
        # The builder grows vocab_cap if the accumulated knowledge needs more whole-word tokens.
        self._set_artifacts(self.builder(self.knowledge_texts, self.grammar_pairs, self.arch, self.vocab_cap))
        if self.vocab_cap != prev_cap:
            self._log("ok", "Adapted vocabulary ceiling " + str(prev_cap) + " -> "
                      + str(self.vocab_cap) + " to fit the new content.")
        self.save_state()
        if rebuild:
            self._log("info", "Re-exporting and rebuilding Ollama with the new knowledge...")
            self.export_and_rebuild()
        return True

    # Backwards-compatible alias.
    def learn_markdown(self, path, rebuild=True):
        return self.learn_document(path, rebuild=rebuild)


# --- Quick Testing Suite ---
if __name__ == "__main__":
    def _fake_builder(texts, grammar_pairs, arch, vocab_cap):
        return {"model": None, "tokenizer": None, "config": None,
                "tokens": [0], "scores": [], "token_types": [], "merges": [], "eos_id": 0,
                "vocab_cap": vocab_cap, "arch": arch}

    demo = {"grammar": "ops", "playbook": {
        "diagnostics": {"disk": ["df -h", "du -sh *"], "load": ["uptime", "w"]},
        "motd": "welcome to the ops bot"}}
    tree = ModelAssets.playbook_tree_from_json(demo)
    assets = ModelAssets(
        builder=_fake_builder, knowledge_texts=[], playbook=tree, grammar_name="ops",
        artifacts=_fake_builder([], [], {"num_layers": 2}, 4096),
        arch={"num_layers": 2}, vocab_cap=4096,
        gguf_path="x.gguf", modelfile_path="Modelfile",
        ollama_name="x", ollama_host="http://127.0.0.1:11434",
        modelfile_template="<|endoftext|>{{ .Prompt }}", state_path="/tmp/_assets_state.json",
    )
    print("initiated:", assets.initiated)
    print("read missing file:", assets.read_markdown("does_not_exist.md"))
    print("flattened pairs:")
    for p, a in assets.grammar_pairs:
        print("   '" + p + "' -> '" + a + "'")
    print("resolve ['diagnostics','disk']:", ModelAssets.resolve_path(tree, ["diagnostics", "disk"]))
    print("resolve ['nope']:", ModelAssets.resolve_path(tree, ["nope"]))
    assets.save_state()
    print("reloaded:", ModelAssets.load_state("/tmp/_assets_state.json") is not None)
