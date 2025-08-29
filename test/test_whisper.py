from faster_whisper import WhisperModel
model_size = "small.en"
local_dir = "STT-model"
import os
import pyaudio
import numpy as np
import wave
import sounddevice
"""
Here is the workflow of STT for fast inference:
First we load both the WhisperModel and initialize the Pyaudio object
The record audio functions will take the input (audio) and write it into a temperary buffer
The temperary buffer will be in bytes, probably, and will be passed onto the transcribe_audio for transcription
For faster and more efficient real-time inference, it is still better to divide the audio into small chunks, before passing it onto the STT model
"""

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

def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')
    # Create a PyAudio instance
    p = pyaudio.PyAudio()
    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    
    # Read and play the audio data
    data = wf.readframes(1024) 
    while data:
        stream.write(data)
        data = wf.readframes(1024)
        # Stop and close the stream and PyAudio instance
        stream.stop_stream()
        stream.close()
        p.terminate()
FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def record_audio(seconds=10):
    # this function must return a temperory buffer, but right now, it's not returning anything
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate = RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
    frames = []
    print("Recording ...")
    for _ in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Recording Stopped.")
    output = "audio.wav"
    # After recording, save it into a temperary wav file
    obj = wave.open(output, "wb")
    obj.setnchannels(CHANNELS)
    obj.setsampwidth(p.get_sample_size(FORMAT))
    obj.setframerate(RATE)
    obj.writeframes(b''.join(frames))
    obj.close()
    return output
path = "OpenVoice/resources/example_reference.wav"
# output = transcribe_audio(path)
# print(output)
if __name__ == '__main__':
    record_audio()