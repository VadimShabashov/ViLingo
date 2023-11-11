import whisperx


class STTModel:
    def __init__(self, device, compute_type="float16"):
        self.device = device
        self.model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="ru")
        self.model_a, self.metadata = whisperx.load_align_model(language_code="ru", device=device)

    def __call__(self, voice, batch_size=16):
        """
        returns: list[dict(text:..., start:..., end:...),...]
        """
        # if you have problem with array, save file and uncomment it
        # voice = whisperx.load_audio(audio_file)
        result = self.model.transcribe(voice, batch_size=batch_size)
        result = whisperx.align(result["segments"],
                                self.model_a,
                                self.metadata,
                                voice,
                                self.device,
                                return_char_alignments=False)
        res = []
        for segment in result["segments"]:
            segment.pop('words', None)
            res.append(segment)
        return res
