import sys
import os
import torch
import torch.nn.functional as f
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

from sentence_transformers import SentenceTransformer, util
import threading

@dataclass
class GenLLMConfig:
    max_new_tokens: int = 160
    do_sample: bool = False              # greedy by default for speed/stability
    # temperature: float = 0.7
    # top_k: int = 50
    # top_p: float = 0.95
    repetition_penalty: float = 1.3

@dataclass
class RAGConfig:
    top_k: int = 4
    max_history_turns: int = 8

MODEL_NAME = "facebook/opt-2.7B"
CACHE_DIR = "./optimized-opt-2.7B-8bit"
LOCAL_DIR = (
    "./optimized-opt-2.7B-8bit/models--facebook--opt-2.7B/"
    "snapshots/905a4b602cda5c501f1b3a2650a4152680238254"
)

CHATBOT_NAME = "Sarah"
CHATBOT_DIR = "temp-memory/chatbot2.txt"
VAULT_PATH = "temp-memory/vault.txt"


def read_file(path: str, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8") as read:
            return read.read()
    except FileNotFoundError:
        return default

def read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as read_lines:
        return [ln.rstrip("\n") for ln in read_lines.readlines() if ln.strip()]

def append_lines(path: str, text:str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")

class Retriever:
    def __init__(self, device: str = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Put embedder on GPU when available
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.vault_texts: List[str] = []
        self.vault_embs: torch.Tensor = torch.empty(0)

    def load_vault(self, path: str) -> None:
        self.vault_texts = read_lines(path)
        if len(self.vault_texts) == 0:
            self.vault_embs = torch.empty(0)
            return
        # encode directly to tensor for speed; keep on embedder device
        self.vault_embs = self.embedder.encode(
            self.vault_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    def add_to_vault(self, path: str, text: str) -> None:
        append_lines(path, text)
        self.vault_texts.append(text)
        new_emb = self.embedder.encode([text], convert_to_tensor=True, show_progress_bar=False)
        if self.vault_embs.nelement() == 0:
            self.vault_embs = new_emb
        else:
            self.vault_embs = torch.cat([self.vault_embs, new_emb], dim=0)

    def get_relevant(self, query: str, top_k: int = 3) -> List[str]:
        if self.vault_embs is None or self.vault_embs.nelement() == 0:
            return []
        q_emb = self.embedder.encode([query], convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(q_emb, self.vault_embs)[0]  # shape: [N]
        k = min(top_k, scores.shape[0])
        top_idx = torch.topk(scores, k=k).indices.tolist()
        return [self.vault_texts[i].strip() for i in top_idx]

# =====================
# LLM Wrapper
# =====================

class LLMModel:
    def __init__(self, local_dir: str, cache_dir: str = None):
        self.local_dir = local_dir
        self.cache_dir = cache_dir
        # 8-bit quantized load (single pass)
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            device_map="auto",
            quantization_config=bnb_cfg,
            offload_folder=os.path.join(cache_dir or ".", "offload"),
            torch_dtype=torch.float16,
        )
        # Helps reduce memory during training; harmless during inference
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

    @torch.inference_mode()
    def generate(self, prompt: str, gen_cfg: GenLLMConfig, stream: bool = False):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move inputs to the same device as the first weight shard
        device = list(self.model.hf_device_map.keys())[0] if hasattr(self.model, "hf_device_map") else "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(self.model.device if hasattr(self.model, "device") else device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_k=gen_cfg.top_k,
            top_p=gen_cfg.top_p,
            repetition_penalty=gen_cfg.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            thread = threading.Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
            thread.start()
            return streamer  # caller will iterate
        else:
            output_ids = self.model.generate(**inputs, **gen_kwargs)
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# =====================
# Prompting
# =====================

PROMPT_TEMPLATE = (
    "{system}\n\n"
    "Relevant information from the knowledge vault:\n{context}\n\n"
    "Conversation so far:\n{history}\n\n"
    "User: {user}\n{bot}:"
)


def build_prompt(
    system_message: str,
    user_input: str,
    history: List[Dict[str, str]],
    relevant_context: List[str],
    bot_name: str,
    rag_cfg: RAGConfig,
) -> str:
    # keep only last N turns
    if len(history) > rag_cfg.max_history_turns:
        history = history[-rag_cfg.max_history_turns :]
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    context_text = "\n".join(relevant_context) if relevant_context else "(none)"
    return PROMPT_TEMPLATE.format(
        system=system_message.strip(),
        context=context_text.strip(),
        history=history_text.strip(),
        user=user_input.strip(),
        bot=bot_name,
    )

# =====================
# Chat Loop
# =====================

def chat_loop():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load components
    retriever = Retriever(device=device)
    retriever.load_vault(VAULT_PATH)

    llm = LLMModel(local_dir=LOCAL_DIR, cache_dir=CACHE_DIR)

    system_message = read_file(CHATBOT_DIR, default=(
        "You are a helpful assistant. Keep answers concise unless more detail is requested."
    ))

    gen_cfg = GenLLMConfig()
    rag_cfg = RAGConfig()

    history: List[Dict[str, str]] = []

    print("Type 'exit' to quit. Commands: 'print info', 'delete info', 'insert info: <text>'\n")

    while True:
        try:
            user_input = input("Input here: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        low = user_input.lower()
        if low == "exit":
            print("Goodbye!")
            break

        # Commands
        if low.startswith("print info"):
            print("\n--- Info contents ---")
            print(read_file(VAULT_PATH, default="(empty)"))
            print("---------------------\n")
            continue

        if low.startswith("delete info"):
            confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                if os.path.exists(VAULT_PATH):
                    os.remove(VAULT_PATH)
                retriever.load_vault(VAULT_PATH)
                print("Info deleted.\n")
            else:
                print("Cancelled.\n")
            continue

        if low.startswith("insert info"):
            # formats accepted: "insert info: <text>" or ask interactively if missing
            parts = user_input.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                info_text = parts[1].strip()
            else:
                info_text = input("Enter info text to insert: ").strip()
            if info_text:
                retriever.add_to_vault(VAULT_PATH, info_text)
                print("Wrote to info.\n")
            else:
                print("Nothing added.\n")
            continue

        # RAG retrieval
        relevant = retriever.get_relevant(user_input, top_k=rag_cfg.top_k)
        prompt = build_prompt(system_message, user_input, history, relevant, CHATBOT_NAME, rag_cfg)

        # Streamed generation for responsiveness
        print(f"\n{CHATBOT_NAME}: ", end="", flush=True)
        streamer = llm.generate(prompt, gen_cfg, stream=True)
        collected = []
        for token in streamer:
            sys.stdout.write(token)
            sys.stdout.flush()
            collected.append(token)
        print("\n")
        reply_text = "".join(collected).strip()

        # Update history (truncate to last N kept by builder anyway)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply_text})


if __name__ == "__main__":
    chat_loop()