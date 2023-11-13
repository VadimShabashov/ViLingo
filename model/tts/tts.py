import torch
from TTS.api import TTS


class TTSModel:
    def __init__(self):
        self.device = torch.device("cuda")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def __call__(self, text, language, voice_reference_path, translated_voice_path):
        self.tts.tts_to_file(
            text=text,
            speaker_wav=voice_reference_path,
            language=language,
            file_path=translated_voice_path
        )
