import numpy as np
from scipy.io.wavfile import write
from typing import Optional

def unnormalize_data(input_array: np.array) -> np.array:
    unnormalized_data = input_array * 32767
    audio_array = unnormalized_data.cpu().numpy().flatten()
    int_audio_array = np.int16(audio_array)
    return int_audio_array

def save_file_input_mp3(input_array: unnormalize_data, output_path: str):
    """
    output audio stream into wav or mp3 file format
    This step includes: unnormalizing the data, saving the data in the cpu, 
    converting back to numpy array in the int16 format and saving it in to mp3 sound file"""
    try:
        write(output_path, 22050, input_array)
        print(f"write successfully to file {output_path}")
    except Exception as e:
        print(f"error writing the audio file: {e}")

def output_audio_stream_with_pyaudio(audio_data: np.array):
    """output audio stream to the speaker instead of saving audio into wav or mp3 file format"""
    import pyaudio

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,  # Mono
                    rate=44100,  # Sampling rate
                    output=True)

    # Write the int16 array as bytes to the stream
    print(f"audio array before normalization {audio_data}")
    audio_data = unnormalize_data(audio_data)
    print(f"audio array after normalization {audio_data}")
    print(np.max(audio_data), np.min(audio_data))
    stream.write(audio_data.tobytes())

    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

def output_audio_stream_with_sd(audio_data):
    """output audio stream to the speaker with sounddevice library"""
    import sounddevice as sd
    audio_array = audio_array.cpu().numpy().flatten()
    sd.play(audio_data, 22050)
    sd.wait()