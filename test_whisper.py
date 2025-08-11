from faster_whisper import WhisperModel
model_size = "small.en"
local_dir = "STT-model"
import os
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=local_dir)
else:
    whisper_model = WhisperModel('STT-model\models--guillaumekln--faster-whisper-small.en\snapshots\model', device="cpu", compute_type="int8", local_files_only=True)
import time
def transcribe_audio(audio_path):
    start_time = time.time()
    segments, info = whisper_model.transcribe(audio_path, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    end_time = time.time()
    inference_time = (end_time-start_time)
    print(f"Inference time: {inference_time}")
    return transcription.strip()

path = "OpenVoice/resources/example_reference.wav"
output = transcribe_audio(path)
print(output)