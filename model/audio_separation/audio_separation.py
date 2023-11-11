import logging
from pathlib import Path
from typing import List

import demucs.separate


class AudioSeparationModel:
    def __init__(self):
        self.output_dir = "separated/htdemucs"

    def __call__(self, audio_path: str) -> List[Path]:
        self.audio_path = audio_path
        return self.separate_sounds()

    def separate_sounds(self) -> List[Path]:
        logging.info("Separating voice...")
        demucs.separate.main([self.audio_path])
        logging.info("Separation is done!")

        input_filename = Path(self.audio_path).stem
        result_files = [
            Path(self.output_dir) / input_filename / "vocals.wav",
            Path(self.output_dir) / input_filename / "bass.wav",
            Path(self.output_dir) / input_filename / "drums.wav",
            Path(self.output_dir) / input_filename / "other.wav",
        ]

        return result_files
