from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.1")
import textwrap
import wave
import pyaudio
import numpy as np
paragraph = """Absolutely, I'd be happy to help explain!

A Large Language Model (LLM) is a type of artificial intelligence model that is trained on a large dataset to understand and generate human-like text. It's called a "large" language model because of the vast amount of data it is trained on, which can include books, websites, or other text sources.

Building a large language model from scratch is a complex task that requires a significant amount of computational resources. It typically involves the following steps:

1. **Data Collection**: Gather a large and diverse dataset of text data. This could be from books, websites, or other sources. The larger and more diverse the dataset, the better the model will be at understanding and generating human-like text.

2. **Data Preprocessing**: Clean the data to remove any irrelevant or noisy information. This includes removing duplicate data, correcting errors, and converting text to a format that the model can understand.

3. **Model Architecture**: Choose a model architecture, such as a Transformer or a Recurrent Neural Network. These architectures have proven to be effective for generating human-like text.

4. **Training**: Use the preprocessed data to train the model. This involves feeding the data through the model and adjusting the model's parameters to minimize the difference between the model's output and the target text.

5. **Fine-tuning**: After initial training, the model can be fine-tuned on a specific task, such as text completion or translation. 

6. **Evaluation**: Test the model's performance on various tasks and benchmarks to evaluate its abilities and identify any areas for improvement.

Please note that building a large language model from scratch is a complex and computationally expensive task, often requiring access to significant computational resources. There are also pre-trained language models available from various providers, such as Google's BERT or Microsoft's T5, which can be fine-tuned for specific tasks."""
texts = textwrap.wrap(paragraph, width=200)
audio = [m.generate(text) for text in texts]
audio = np.concatenate(audio)
# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)

def play_audio(file_path):
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

play_audio("output.wav")