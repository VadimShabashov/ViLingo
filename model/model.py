from audio_separation.audio_separation import AudioSeparationModel
from stt.stt import STTModel
from translation.translation import TranslationModel
from tts.tts import TTSModel

import numpy as np


class Model:
    def __init__(self):
        self.audio_separation_model = AudioSeparationModel()
        self.stt_model = STTModel()
        self.translation_model = TranslationModel()
        self.tts_model = TTSModel()

    @staticmethod
    def read_video(video_path):
        return

    @staticmethod
    def extract_audio(video):
        """
        Extract audio from video
        """
        return

    @staticmethod
    def fit_audio_length(audio, length):
        """
        Fit audio to the provided length
        """
        return audio

    @staticmethod
    def add_audio_by_timestamp(audio, target_audio, start, end):
        """
        Add audio to target audio by timestamp
        """
        target_audio[start:end] = audio

    @staticmethod
    def save_video_with_new_audio(resulting_audio):
        return ""

    def run(self, params):
        """
        params: Dict(video_path, language, make_lipsync)
        """

        # Read video
        video = self.read_video(params['video_path'])

        # Extract audio from video
        audio = self.extract_audio(video)

        # Split audio on voices and noise
        voices, noise = self.audio_separation_model(audio)

        # Split overlapping voices?

        # Voice to text + timestamps
        texts_and_timestamps = self.stt_model(voices)

        # Process each phrase separately
        translated_phrases = np.zeros_like(voices)
        for text_and_timestamp in texts_and_timestamps:
            # Translate phrase
            translated_text = self.translation_model(text_and_timestamp["text"], text_and_timestamp["language"])

            # Speech to text
            translated_voice = self.tts_model(translated_text)

            # Fit to the length of initial voice
            translated_phrase_fitted = self.fit_audio_length(
                translated_voice,
                text_and_timestamp["end"] - text_and_timestamp["start"]
            )

            # Add voice using timestamps
            self.add_audio_by_timestamp(
                translated_phrase_fitted,
                translated_phrases,
                text_and_timestamp["start"],
                text_and_timestamp["end"]
            )

        # Add noise back
        resulting_audio = translated_phrases + noise

        # Lipsync?

        # Save video with new audio
        translated_video_path = self.save_video_with_new_audio(resulting_audio)

        return translated_video_path
