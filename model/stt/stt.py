import os
import whisperx
from collections import defaultdict


class STTModel:
    def __init__(self, compute_type="float16"):
        self.device = "cuda"
        self.model = whisperx.load_model("large-v2", self.device, compute_type=compute_type, language="ru")
        self.model_a, self.metadata = whisperx.load_align_model(language_code="ru", device=self.device)

    def __call__(self, voices_path, batch_size=16):
        """
        returns: list[dict(text:..., start:..., end:..., speaker: ...),...]
        """
        # if you have problem with array, save file and uncomment it
        voices = whisperx.load_audio(voices_path)
        result = self.model.transcribe(voices, batch_size=batch_size)
        result = whisperx.align(result["segments"],
                                self.model_a,
                                self.metadata,
                                voices,
                                self.device,
                                return_char_alignments=False)
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv("TOKEN"), device=self.device)
        diarize_segments = diarize_model(voices)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        res = []
        speaker_times = defaultdict(list)
        for segment in result["segments"]:
            list_of_words = segment["words"]
            segment.pop('words', None)
            speakers = []
            for i in list_of_words:
                if "speaker" in i:
                    speakers.append(i["speaker"])
            cur_speaker = max(speakers, key=speakers.count) if len(speakers) > 0 else "UNKNOWN"
            segment["speaker"] = cur_speaker
            speaker_times[cur_speaker].append((segment["start"], segment["end"]))
            res.append(segment)

        return res, speaker_times
