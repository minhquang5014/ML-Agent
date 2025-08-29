import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'OpenVoice')))
from OpenVoice.openvoice import se_extractor
from OpenVoice.openvoice.api import BaseSpeakerTTS, ToneColorConverter
import pygame as pg
import io
class ModelInitialization:
    def __init__(self, ckpt_base, ckpt_converter, output_dir, reference_speaker, text):
        self.ckpt_base = ckpt_base
        self.ckpt_converter = ckpt_converter
        self.output_dir = output_dir
        self.reference_speaker = reference_speaker
        self.text = text

        # decide which device the model will be loaded in
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        
    def load_model_and_ckpt(self):
        self.base_speaker_tts = BaseSpeakerTTS(f'{self.ckpt_base}/config.json', device=self.device)
        self.base_speaker_tts.load_ckpt(f'{self.ckpt_base}/checkpoint.pth')

        self.tone_color_converter = ToneColorConverter(f'{self.ckpt_converter}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{self.ckpt_converter}/checkpoint.pth')
        self.source_se = torch.load(f'{self.ckpt_base}/en_default_se.pth').to(self.device)

    def extractor(self):
        self.target_se, self.audio_name = se_extractor.get_se(self.reference_speaker,
                                                              self.tone_color_converter,
                                                              target_dir='processed', 
                                                              vad=True)
        print(self.target_se, self.audio_name)
    def running_inference(self):
        save_path = f'{self.output_dir}/output_en_default.wav'
        src_path = f'{self.output_dir}/tmp.wav'
        self.buffer_ouput = self.base_speaker_tts.tts(self.text, src_path, speaker="excited", language='English', speed=0.9)
        encode_message = "@Myshell"
        self.tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=self.source_se, 
            tgt_se=self.target_se, 
            output_path=save_path,
            message=encode_message)
    def output_audio(self):
        pg.init()
        pg.mixer.init()
        pg.mixer.music.load(self.buffer_ouput, "wav")
        pg.mixer.music.play()
        while pg.mixer.music.get_busy():  # Wait for playback to finish
            pass
        

if __name__ == '__main__':
    text = """Energy Secretary Chris Wright, who made millions in the fracking industry, commissioned the report. In a preface, he did not deny that climate change exists.
    “Climate change is real, and it deserves attention,” he wrote. “But it is not the greatest threat facing humanity. That distinction belongs to global energy poverty.”
    In other words, Wright sees more damage to humans from cutting back on carbon emissions.
    That is a minority view in the scientific community, which has a much, much larger body of peer reviewed studies that raise the alarm about climate change. Most notably, the Intergovernmental Panel on Climate Change issues peer-reviewed reports with hundreds of authors from around world. The Trump administration has barred US government scientists from taking part in the next installment, due out in 2029."""
    model = ModelInitialization(ckpt_base='OpenVoice/checkpoints/base_speakers/EN',
                                ckpt_converter='OpenVoice/checkpoints/converter',
                                output_dir='OpenVoice/outputs',
                                reference_speaker='OpenVoice/resources/demo_speaker2.mp3',
                                text=text)
    model.load_model_and_ckpt()
    model.extractor()
    model.running_inference()
    model.output_audio()