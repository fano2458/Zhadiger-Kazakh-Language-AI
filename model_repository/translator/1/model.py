import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import torch


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device('cpu')
        self.languages = {'ru': 'rus_Cyrl', 'kk': 'kaz_Cyrl', 'en': 'eng_Latn', 'tr': 'tur_Latn'}
        self.load_model()

    def load_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained('/assets/translator/checkpoint').eval().to(self.device)
        self.kaz_tokenizer = AutoTokenizer.from_pretrained('/assets/translator/checkpoint', src_lang="kaz_Cyrl")
        self.eng_tokenizer = AutoTokenizer.from_pretrained('/assets/translator/checkpoint', src_lang="eng_Latn")

    def preprocess_text(self, texts, lang_type):
        tokenizer = self.kaz_tokenizer if lang_type == "kaz" else self.eng_tokenizer
        tokenized_inputs = tokenizer(texts, return_tensors="pt").to(self.device)
        return tokenized_inputs, tokenizer

    def translate(self, tokenized_inputs, tokenizer, trg_lang):
        with torch.no_grad():
            output = self.model.generate(**tokenized_inputs, 
                                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(self.languages[trg_lang]), max_length=1000)
        translated_sentence = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return translated_sentence

    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            texts = [el.decode() for el in texts][0]

            lang_type = pb_utils.get_input_tensor_by_name(request, "lang_type").as_numpy()
            trg_lang = pb_utils.get_input_tensor_by_name(request, "trg_lang").as_numpy()

            lang_type = [el.decode() for el in lang_type][0]
            trg_lang = [el.decode() for el in trg_lang][0]

            tokenized_inputs, tokenizer = self.preprocess_text(texts, lang_type)
            translated_sentence = self.translate(tokenized_inputs, tokenizer, trg_lang)

            output_tensor = pb_utils.Tensor("output", np.array([translated_sentence], dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
