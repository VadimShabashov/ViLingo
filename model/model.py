from audio_separation.audio_separation import AudioSeparationModel
from stt.stt import STTModel
from translation.translation import TranslationModel
from tts.tts import TTSModel

import os
import torch
import numpy as np
import moviepy.editor
import librosa
from pathlib import Path
from typing import List
from scipy.io import wavfile
from pydub import AudioSegment
import soundfile as sf


class Model:
    NOISE_PATH = Path("noise.wav")
    TRANSLATED_AUDIO_PATH = Path("translated_audio_path.wav")
    TRANSLATED_VOICES_PATH = Path("translated_voices_path.wav")
    VOICE_REFERENCE_PATH = Path("voice_reference_path.wav")

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.audio_separation_model = AudioSeparationModel()
        self.stt_model = STTModel(self.device)
        self.translation_model = TranslationModel()
        self.tts_model = TTSModel(self.device)

    @staticmethod
    def extract_and_save_audio_from_video(video_path):
        """
        Extract audio from video, then save it to the file .wav
        """

        # Load the video
        audio = moviepy.editor.VideoFileClip(video_path).audio

        # Extract and save audio
        video_path_no_extension = os.path.splitext(video_path)[0]
        audio_path = video_path_no_extension + ".wav"
        audio.write_audiofile(audio_path)

        return audio_path, audio.fps

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
    def save_audio(audio, audio_path, samplerate):
        sf.write(audio_path, audio, samplerate)

    @staticmethod
    def mix_audio(files: List[Path], mixed_path: Path):
        mixed = AudioSegment.from_file(files[0])

        for file in files[1:]:
            next_audio = AudioSegment.from_file(file)
            mixed = mixed.overlay(next_audio)

        mixed.export(mixed_path, format="wav")

    @staticmethod
    def add_audio_by_timestamp(audio_segment, target_audio, start, end):
        """
        Add audio to target audio by timestamp
        """
        target_audio[start:end] = audio_segment

    @staticmethod
    def read_audio(audio_path):
        samplerate, data = wavfile.read(audio_path)

        if len(data.shape) == 1:
            return data, samplerate

        return data[:, 0], samplerate

    @staticmethod
    def save_video_with_new_audio(video_path, translated_audio_path):
        # Convert audio to suitable format from numpy array
        translated_audio = moviepy.editor.AudioFileClip(translated_audio_path)

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
        audio_path, samplerate = self.extract_and_save_audio_from_video(params['video_path'])

        # Split audio on voices and noise
        voices_path, drums_path, bass_path, other_path = self.audio_separation_model(audio_path)

        # Mix noise
        self.mix_audio(
            [
                drums_path,
                bass_path,
                other_path
            ],
            self.NOISE_PATH
        )

        # Read voices
        voices = self.read_audio(voices_path)

        # Voice to text + timestamps
        texts_and_timestamps = self.stt_model(voices)

        # Process each phrase separately
        translated_voices = np.zeros_like(voices)
        for text_and_timestamp in texts_and_timestamps:
            # Translate phrase
            translated_text = self.translation_model(text_and_timestamp["text"], text_and_timestamp["language"])

            # Get reference for voice cloning
            voice_reference = voices[text_and_timestamp["start"]:text_and_timestamp["end"]]

            # Save reference for voice cloning
            self.save_audio(voice_reference, self.VOICE_REFERENCE_PATH, samplerate)

            # Speech to text
            translated_voice = self.tts_model(translated_text, self.VOICE_REFERENCE_PATH)

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

        # Save audio to file
        self.save_audio(translated_voices, self.TRANSLATED_VOICES_PATH, samplerate)

        # Add noise back
        self.mix_audio(
            [
                self.TRANSLATED_VOICES_PATH,
                self.NOISE_PATH
            ],
            self.TRANSLATED_AUDIO_PATH
        )

        # Lipsync?

        # Save video with new audio
        translated_video_path = self.save_video_with_new_audio(
            params['video_path'],
            self.TRANSLATED_AUDIO_PATH
        )

        return translated_video_path
