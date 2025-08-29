import os 
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import (
    AutoTokenizer,
)
import soundfile as sf
model_name = "parler-tts/parler-mini-v1-jenny"
path = "parlerTTS/models--parler-tts--parler-mini-v1-jenny/snapshots/parlerTTS/models--parler-tts--parler-mini-v1-jenny/snapshots/855b4ef2e2b8d85a5d75930046aa237889dec72c"

torch_dtype = "float16"
device = "cuda" if torch.cuda.is_available else "cpu"
print(f"device used for parler TTS mode: {device}")
model_path = "parlerTTS"
if not os.path.exists(model_path):
    os.mkdir(model_path)
try:
    print("trying to load the parler TTS model from your device")
    model = ParlerTTSForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=path, torch_dtype = torch_dtype, cache_dir = model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir = model_path)    
except Exception as e:
    print(f"model not available on your device, downloading the model from the huggingface")
    model = ParlerTTSForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_name, torch_dtype = torch_dtype, cache_dir = model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir = model_path)

print(model)

prompt = """“This is climate breakdown, in real time. We must exit this road to ruin and we have no time to lose. 
In 2025, countries must put the world on a safer path by dramatically slashing emissions and supporting the transition to a renewable future. 
It is essential, and it is possible.”"""

description = "Jenny speaks at an average pace with an animated delivery in a very confined sounding environment with clear audio quality."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)