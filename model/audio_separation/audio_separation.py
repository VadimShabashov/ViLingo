import logging
from pathlib import Path
from typing import List

import demucs.separate
import numpy as np
import soundfile as sf
from pydub import AudioSegment


class AudioSeparationModel:
    def __init__(self):
        self.output_dir = "separated"
        return

    def __call__(self, audio_path: str) -> (np.ndarray, np.ndarray):
        self.audio_path = audio_path

        voice_path = self.separate_voice()
        sounds_path = self.extract_other_sounds()

        return self.read_audio_files_to_arrays(voice_path, sounds_path)

    @staticmethod
    def read_audio_files_to_arrays(
        file_path_1: Path, file_path_2: Path
    ) -> (np.ndarray, np.ndarray):
        data1, _ = sf.read(file_path_1.as_posix())

        data2, _ = sf.read(file_path_2.as_posix())
        logging.info("Resulted np arrays are shape of:")
        logging.info(f"voice: {data1.shape}")
        logging.info(f"other: {data2.shape}")

        return data1, data2

    @staticmethod
    def mix_audio(files: List[Path], output_file: Path):
        mixed = AudioSegment.from_file(files[0])

        for file in files[1:]:
            next_audio = AudioSegment.from_file(file)
            mixed = mixed.overlay(next_audio)

        mixed.export(output_file.as_posix(), format="wav")

        print(f"Mixed audio saved as {output_file}")

    def separate_voice(self) -> Path:
        logging.info("Separating voice...")
        demucs.separate.main(["-o", self.output_dir, self.audio_path])
        logging.info("Separation is done!")

        input_filename = Path(self.audio_path).stem
        voice_path = Path(self.output_dir) / input_filename / "vocals.wav"

        return voice_path

    def extract_other_sounds(self) -> Path:
        logging.info("Extracting sounds...")
        input_filename = Path(self.audio_path).stem

        other_files = [
            Path(self.output_dir) / input_filename / "bass.wav",
            Path(self.output_dir) / input_filename / "drums.wav",
            Path(self.output_dir) / input_filename / "other.wav",
        ]

        sounds_path = Path(self.output_dir) / input_filename / "sounds.wav"

        self.mix_audio(other_files, sounds_path)

        return sounds_path
