from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TranslationModel:
    def __init__(self):
        self.model_name = 'model_name = "facebook/nllb-200-distilled-600M'
        self.tokenizer = self.load_tokenizer()
        self.translation_model = self.load_model()

    def load_tokenizer(self, src_language='ru'):
        src_language = self.get_language(src_language)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, src_lang=src_language
        )
        return tokenizer

    def load_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    @staticmethod
    def get_language(language):
        language_mapping = {
            'ru': 'rus_Cyrl',  # русский
            'de': 'deu_Latn',  # немецкий
            'en': 'eng_Latn',  # английский
            'fr': 'fra_Latn',  # французский
            'it': 'ita_Latn',  # итальянский
            'es': 'spa_Latn',  # испанский
            'ja': 'jpn_Jpan',  # японский
            'zh': 'zho_Hans',  # китайский
            'pt': 'por_Latn',  # португальский
            'cs': 'ces_Latn',  # чешский
            'nl': 'nld_Latn',  # датский
            'pl': 'pol_Latn',  # польский
            'tr': 'tur_Latn'  # турецкий
        }

        if language not in language_mapping:
            raise ValueError("Language is not supported")

        return language_mapping[language]

    @staticmethod
    def preprocess_text(text):
        text = text.strip()
        stop = ['...', '.', '?', '!', '!!!', '?!', '!?', ';']

        if not text:
            return text

        if not text[-1] in stop:
            text = text + '.'

        return text

    def __call__(self, text, target_language):
        target_language = self.get_language(target_language)

        text = self.preprocess_text(text)
        inputs = self.tokenizer(text, return_tensors="pt")
        translated_tokens = self.translation_model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[target_language],
            max_length=512
        )
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        return translated_text
