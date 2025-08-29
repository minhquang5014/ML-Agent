from llama_cpp import Llama
from contextlib import redirect_stdout
import os
from faster_whisper import WhisperModel
import pyaudio
import wave
import io
import torch
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass
import json
import io
import queue
import threading
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.1")
model_path = "LLM_model/mistral-7b-instruct-v0.2.Q2_K.gguf"
# Suppress stdout during model loading
with open(os.devnull, "w") as f, redirect_stdout(f):
    llm = Llama(model_path=model_path, # Use the downloaded model path
                n_gpu_layers=30,
                n_threads=8,
                n_ctx=4096)
whisper_model = WhisperModel('STT-model\models--guillaumekln--faster-whisper-small.en\snapshots\model', 
                             device="cpu", 
                             compute_type="int8", 
                             local_files_only=True)
FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def record_audio(input_queue: queue.Queue, stop_event: threading.Event):
    """
    Must re-factor or re-structure this function later.
    It must return a temperory buffer (io.Bytes) or queue, but right now, it's returning path to audio file
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate = RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
    print("Recording ...")
    try:
        while not stop_event.is_set():
            data = stream.read(FRAMES_PER_BUFFER)
            input_queue.put(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Recording Stopped.")

def audio_writer(input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event):
    """
    Consume from input_queue and write into a BytesIO WAV buffer.
    The final BytesIO is put into output_queue after stop_event is set.
    """
    frames = []
    while not stop_event.is_set() or not input_queue.empty():
        try:
            data = input_queue.get(timeout=0.1)
            frames.append(data)
        except queue.Empty:
            continue

    # Save to in-memory WAV
    output = io.BytesIO()
    wf = wave.open(output, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    output.seek(0)  # reset pointer to the beginning for reading
    output_queue.put(output)

def transcribe_audio(audio_path):
    """
    Re-factor this one later, it must pass in temperary buffer as the argument (io.Bytes or queue)
    """
    segments, info = whisper_model.transcribe(audio_path, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()


class LLMInputPrompt:
    def __init__(self,
                history = [{'role': "user", 'content': "Who are you"},
                        {'role': "assistant", 'content':"I'm a friendly assistant named George. I'm here to help answer any questions you might have."}],
                user_input = "You are an AI model compressed in gguf format. Can you tell me how to continue training you?",
                system = """
                You are a frienly chatbot named George. 
                You are a friendly female-voiced assistance. You answer with soft voices
                Your job is to generate a frienly answer to every users' questions. 
                You generate responses in very polite way to everyone.
                You can guide users to code""",
                context = None,
                bot_name = "George",
                rag_config = None):
        self.history = history
        self.user_input = user_input
        self.system = system
        self.context = context
        self.bot_name = bot_name
        self.rag_config = rag_config
    def build_prompt(self) -> Tuple[str]:
        """
        Extent this function later on.
        conversation history will be stored in a json file
        """
        if self.rag_config is not None and len(self.history) > self.rag_cfg.max_history_turns:
            self.history = self.history[-self.rag_cfg.max_history_turns:]
        history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in self.history])
        context_text = "\n".join(self.context) if self.context else None
        return (
                f"""<|im_start|>system:\n{self.system}
                <|im_end|>
                <|im_start|>Conversation so far:\n{history_text}
                <|im_end|>
                <|im_start|>Question from user:\n{self.user_input}
                <|im_end|>
                <|im_start|>Relevant context:\n{context_text}
                <|im_end|>"""
                )
    
# <|im_start|>Relevant information from the knowledge vault:\n{context}\n\n<|im_end|>
DEFAULT_PROMPT = """
<s>[INST] <<SYS>>
You are a friendly chatbot named George.
You are a female-voiced assistant. You always respond politely and can guide users with code.
Memory: Your owner is trying to build a LLM for a scientific research at his university
<</SYS>>

User: Who are you?
Assistant: I'm a friendly assistant named George. I'm here to help.
User: Hello, can you tell me what is LLM and how to build a large language model from scratch?
[/INST]"""
def running_inference_llama(prompt, output_queue: queue):
    # Generate output
    output = llm.create_completion(prompt, 
                                   max_tokens=300, 
                                   stop=["<|im_end|>"], 
                                   stream=True)
    output_dir = "mistral_7B_output_dir" # Changed to a directory name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_text_path = os.path.join(output_dir, "output.txt") # Changed to a file path inside the directory
    # Print the result
    text_output = ""
    with open(output_text_path, "w") as f:
        for token in output:
            chunk = token["choices"][0]["text"]
            print(chunk, end='', flush=True)
            f.write(chunk)
            text_output += chunk
            output_queue.put(chunk)
    output_queue.put(None)
    return text_output
@dataclass
class RAGConfig:
    top_k: int = 4
    max_history_turns: int = 8

def play_audio_from_path(file_path):
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    
    data = wf.readframes(1024) 
    while data:
        stream.write(data)
        data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()
        p.terminate()
def play_audio_real_time(audio_queue: queue.Queue, stop_event: threading.Event):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, 
                    channels=CHANNELS, 
                    rate = RATE, 
                    output=True, 
                    frames_per_buffer=FRAMES_PER_BUFFER)
    while not stop_event.is_set():
        try:
            data = audio_queue.get(timeout=0.1)
            if data is None:
                break
            stream.write(data)
        # data = wf.readframes(1024)
        except queue.Empty:
            continue
    stream.stop_stream()
    stream.close()
    p.terminate()

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
VAULT_PATH = "temp-memory/vault.txt"
CHATBOT_DIR = "temp-memory/chatbot2.txt"

if __name__ == '__main__':
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    stop_event = threading.Event()
    text_queue = queue.Queue()
    audio_queue = queue.Queue()
    stop_event_audio = threading.Event()
    t_recorder = threading.Thread(target=record_audio, args=(input_queue, stop_event))
    t_writer = threading.Thread(target=audio_writer, args=(input_queue, output_queue, stop_event))

    t_recorder.start()
    t_writer.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        # When Ctrl + C is press, stop recording
        stop_event.set()
    
    # wait for recording stop before moving on
    t_recorder.join()
    t_writer.join()

    #get final BytesIO WAV file
    audio_buffer = output_queue.get()
    audio_buffer.seek(0)
    transcription = transcribe_audio(audio_buffer)
    print(f"Transcribing Speech to Text: {transcription}")

    # input_llama_queue = 

    # take the list of conversation history
    # conversation_history = LLMInputPrompt().history
    # conversation_history.append({"role": "user", "content": transcription})
    # print(conversation_history)

    # llm_input = LLMInputPrompt(user_input=transcription)
    # prompt = llm_input.build_prompt()
    # print(f"Input prompt before passing to the model: {prompt}")
    def tts_worker(text_queue: queue.Queue, audio_queue: queue.Queue, stop_event: threading.Event):
        while not stop_event.is_set():
            try:
                text = text_queue.get(timeout=0.1)
                if text is None:
                    audio_queue.put(None)
                    break
                audio = m.generate(text)
                audio_bytes = (audio*32767).astype("int16").tobytes()
                audio_queue.put(audio_bytes)
            except queue.Empty:
                continue

    threads = [threading.Thread(target = running_inference_llama, args=(DEFAULT_PROMPT, text_queue)),
               threading.Thread(target=tts_worker, args=(text_queue, audio_queue, stop_event_audio)),
                threading.Thread(target=play_audio_real_time, args=(audio_queue, stop_event_audio))] 
    
    for t in threads:
        t.start()
    try: 
        while True:
            pass
    except KeyboardInterrupt:
        stop_event_audio.set()
    
    for t in threads:
        t.join()
    
