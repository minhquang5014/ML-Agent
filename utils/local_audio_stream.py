import threading
import sounddevice as sd
import numpy as np
import time
import logging
import queue
import wave
import pyaudio
logger = logging.getLogger(__name__)

class LocalAudioStreamerSoundDevice:
    """
    This class is a bridge between the microphone and speaker
    Input queue: where audio chunks go in
    Output queue: where audio to play must be pushed
    We still need another thread or process outside of this loop to consume input or produce output
    """
    def __init__(
        self,
        input_queue,
        output_queue,
        list_play_chunk_size=512,
    ):
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        def callback(indata, outdata, frames, time, status):
            print(indata, outdata)
            if self.output_queue.empty():
                self.input_queue.put(indata.copy())
                outdata[:] = 0 * outdata
                print(outdata)
            else:
                outdata[:] = self.output_queue.get()
                print(outdata)

        logger.debug("Available devices:")
        logger.debug(sd.query_devices())
        with sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=callback,
            blocksize=self.list_play_chunk_size,
        ):
            logger.info("Starting local audio stream")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stop recording")
class RecordAudioPyaudio:
    def __init__(self, chunk = 1024,
                        format = pyaudio.paInt16,
                        channels = 1,
                        rate = 44100):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def record_audio(self,seconds=10):
        # this function must return a temperory buffer, but right now, it's not returning anything
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, 
                        channels=self.channels, 
                        rate = self.rate, 
                        input=True, 
                        frames_per_buffer=self.chunk)
        frames = []
        print("Recording ...")
        for _ in range(0, int(self.rate / self.chunk * seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Recording Stopped.")
        output = "audio.wav"
        # After recording, save it into a temperary wav file
        obj = wave.open(output, "wb")
        obj.setnchannels(self.channels)
        obj.setsampwidth(p.get_sample_size(self.format))
        obj.setframerate(self.rate)
        obj.writeframes(b''.join(frames))
        obj.close()
        # return output

if __name__ == '__main__':
    # input_queue = queue.Queue()
    # output_queue = queue.Queue()

    # stream = LocalAudioStreamer(input_queue=input_queue, output_queue=output_queue)
    # t = threading.Thread(target=stream.run)
    # t.start()
    # try:
    #     time.sleep(5)
    #     while not input_queue.empty():
    #         data = input_queue.get()
    #         output_queue.put(data)
    #     time.sleep(5)

    # finally:
    #     stream.stop_event.set()
    #     t.join()
    local_audio = RecordAudioPyaudio()
    local_audio.record_audio()