from audio_separation.audio_separation import AudioSeparationModel
from stt.stt import STTModel
from translation.translation import TranslationModel
from tts.tts import TTSModel

import os
import numpy as np
import moviepy.editor
import librosa


class Model:
    def __init__(self):
        self.audio_separation_model = AudioSeparationModel()
        self.stt_model = STTModel()
        self.translation_model = TranslationModel()
        self.tts_model = TTSModel()

    @staticmethod
    def extract_and_save_audio_from_video(video_path):
        """
        Extract audio from video, then save it to the file .wav
        """

        # Load the video
        video = moviepy.editor.VideoFileClip(video_path)

        # Extract and save audio
        video_path_no_extension = os.path.splitext(video_path)[0]
        audio_path = video_path_no_extension + ".wav"
        video.audio.write_audiofile(audio_path)

        return audio_path

    @staticmethod
    def fit_audio_length(audio, target_duration):
        """
        Fit audio to the provided target_duration
        """

        # Find rate for changing the speed
        rate = audio.duration_seconds / target_duration

        # Change speed
        audio_fitted_length = librosa.effects.time_stretch(audio, rate=rate)

        return audio_fitted_length

    @staticmethod
    def add_audio_by_timestamp(audio_segment, target_audio, start, end):
        """
        Add audio to target audio by timestamp
        """
        target_audio[start:end] = audio_segment

    @staticmethod
    def save_video_with_new_audio(video_path, translated_audio_np, samplerate):
        # Convert audio to suitable format from numpy array
        translated_audio = moviepy.editor.AudioFileClip(translated_audio_np, fps=samplerate)

        # Load video
        video = moviepy.editor.VideoFileClip(video_path)

        # Change audio
        translated_video = video.set_audio(translated_audio)

        # Add '_translated' to filename
        translated_video_path = os.path.splitext(video_path)[0] + '_translated' + os.path.splitext(video_path)[1]

        # Save video
        translated_video.write_videofile(translated_video_path)

        return translated_video_path

    def run(self, params):
        """
        params: Dict(video_path, language, make_lipsync)
        """

        # Extract and save audio to a file
        audio_path = self.extract_and_save_audio_from_video(params['video_path'])

        # Split audio on voices and noise
        voices, noise = self.audio_separation_model(audio_path)

        # Split overlapping voices?

        # Voice to text + timestamps
        texts_and_timestamps = self.stt_model(voices)

        # Process each phrase separately
        translated_voices = np.zeros_like(voices)
        for text_and_timestamp in texts_and_timestamps:
            # Translate phrase
            translated_text = self.translation_model(text_and_timestamp["text"], text_and_timestamp["language"])

            # Get reference for voice cloning
            voice_reference = voices[text_and_timestamp["start"]:text_and_timestamp["end"]]

            # Speech to text
            translated_voice = self.tts_model(translated_text, voice_reference)

            # Fit to the length of initial voice
            translated_phrase_fitted = self.fit_audio_length(
                translated_voice,
                text_and_timestamp["end"] - text_and_timestamp["start"]
            )

            # Add voice using timestamps
            self.add_audio_by_timestamp(
                translated_phrase_fitted,
                translated_voices,
                text_and_timestamp["start"],
                text_and_timestamp["end"]
            )

        # Add noise back
        translated_audio = translated_phrases + noise

        # Lipsync?

        # Save video with new audio
        translated_video_path = self.save_video_with_new_audio(
            params['video_path'],
            translated_audio
        )

        return translated_video_path
