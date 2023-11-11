import torch
import numpy as np
from TTS.api import TTS


class TTSModel:
    def __init__(self, device):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def __call__(self, text, voice_reference_path):
        return np.array(
            self.tts.tts(text=text, speaker_wav=voice_reference_path, language="en")
        )
