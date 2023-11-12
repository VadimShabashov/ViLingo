from audio_separation.audio_separation import AudioSeparationModel
from stt.stt import STTModel
from translation.translation import TranslationModel
from tts.tts import TTSModel

import os
import json
import moviepy.editor
from pydub import AudioSegment


class Model:
    def __init__(self):
        self.audio_separation_model = AudioSeparationModel()
        self.stt_model = STTModel()
        self.translation_model = TranslationModel()
        self.tts_model = TTSModel()

    @staticmethod
    def extract_and_save_audio_from_video(video_path, audio_path):
        """
        Extract audio from video, then save it to the file .wav
        """

        # Load the video
        video = moviepy.editor.VideoFileClip(video_path)

        # Extract and save audio
        video.audio.write_audiofile(audio_path)

    @staticmethod
    def mix_audio(files, mixed_path):
        mixed = AudioSegment.from_file(files[0])

        for file in files[1:]:
            next_audio = AudioSegment.from_file(file)
            mixed = mixed.overlay(next_audio)

        mixed.export(mixed_path, format="wav")

    @staticmethod
    def add_audio_by_timestamp(audio_segment_path, target_audio_path, start_timestamp_ms):
        """
        Add audio to target audio by timestamp
        """

        # Load target audio file
        target_audio = AudioSegment.from_file(target_audio_path, format='wav')

        # Load audio segment to be inserted
        audio_segment = AudioSegment.from_file(audio_segment_path, format='wav')

        # Mix (overlay) the insert audio onto the main audio at the specified timestamp
        mixed_audio = target_audio.overlay(audio_segment, position=start_timestamp_ms)

        # Save mixed
        mixed_audio.export(target_audio_path, format="wav")

    @staticmethod
    def create_empty_sound(target_audio_path, silent_audio_path):
        # Get target audio
        target_audio = AudioSegment.from_file(target_audio_path, format='wav')

        # Create similar silent audio
        silent_audio = AudioSegment.silent(len(target_audio))

        # Save silent audio by path
        silent_audio.export(silent_audio_path, format="wav")

    @staticmethod
    def save_video_with_new_audio(video_path, translated_audio_path, translated_video_path):
        # Convert audio to suitable format from numpy array
        translated_audio = moviepy.editor.AudioFileClip(translated_audio_path)

        # Load video
        video = moviepy.editor.VideoFileClip(video_path)

        # Change audio
        translated_video = video.set_audio(translated_audio)

        # Save video
        translated_video.write_videofile(translated_video_path)

        return translated_video_path

    def run(self, params):
        """
        params: Dict(video_path, language)
        """

        # Create paths
        video_path = params['video_path']
        video_path_no_extension = os.path.splitext(video_path)[0]
        audio_path = video_path_no_extension + ".wav"
        translated_audio_path = video_path_no_extension + "_translated" + ".wav"
        translated_video_path = video_path_no_extension + "_translated" + ".mp4"
        translations_with_timestamps_file_path = video_path_no_extension + "_translated" + ".json"

        translated_voice_path = video_path_no_extension + "_translated_voice" + ".wav"
        translated_voices_path = video_path_no_extension + "_translated_voices" + ".wav"
        voice_reference_path = video_path_no_extension + "_voice_reference" + ".wav"

        # Extract and save audio to a file
        self.extract_and_save_audio_from_video(video_path, audio_path)

        # Split audio on voices and noise
        voices_path, drums_path, bass_path, other_path = self.audio_separation_model(audio_path)

        # Read voices
        voices = AudioSegment.from_file(voices_path, format='wav')

        # Voice to text + timestamps
        texts_and_timestamps, _ = self.stt_model(voices_path)

        # Save translations to file
        with open(translations_with_timestamps_file_path, "w", encoding='utf-8') as translations_with_timestamps_file:
            json.dump(texts_and_timestamps, translations_with_timestamps_file, ensure_ascii=False, indent=4)

        # Create empty sound for filling speakers later
        self.create_empty_sound(audio_path, translated_voices_path)

        # Process each phrase separately
        for text_and_timestamp in texts_and_timestamps:
            # Timestamp to audio indices
            start_timestamp_ms = int(text_and_timestamp["start"] * 1000)
            end_timestamp_ms = int(text_and_timestamp["end"] * 1000)

            # Translate phrase
            translated_text = self.translation_model(text_and_timestamp["text"], params["language"])

            # Get reference for voice cloning and save it to a file
            voice_reference = voices[start_timestamp_ms:end_timestamp_ms]
            voice_reference.export(voice_reference_path, format="wav")

            # Speech to text
            self.tts_model(translated_text, params["language"], voice_reference_path, translated_voice_path)

            # Add voice using timestamps
            self.add_audio_by_timestamp(
                translated_voice_path,
                translated_voices_path,
                start_timestamp_ms
            )

        # Add noise back
        self.mix_audio(
            [
                translated_voices_path,
                drums_path,
                bass_path,
                other_path
            ],
            translated_audio_path
        )

        # Save video with new audio
        translated_video_path = self.save_video_with_new_audio(
            video_path,
            translated_audio_path,
            translated_video_path
        )

        return translated_video_path
