import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
    
        self.model = AutoModelForSeq2SeqLM.from_pretrained('/assets/translator/checkpoint')
        self.kaz_tokenizer = AutoTokenizer.from_pretrained('/assets/translator/checkpoint', src_lang="kaz_Cyrl")
        self.eng_tokenizer = AutoTokenizer.from_pretrained('/assets/translator/checkpoint', src_lang="eng_Latn")
        
    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "TEXTS").as_numpy()
            texts = [el.decode() for el in texts][0]

            # print(texts)

            lang_type = pb_utils.get_input_tensor_by_name(request, "LANG_TYPE").as_numpy()
            trg_lang = pb_utils.get_input_tensor_by_name(request, "TRG_LANG").as_numpy()

            lang_type = [el.decode() for el in lang_type][0]
            trg_lang = [el.decode() for el in trg_lang][0]

            # print(lang_type, trg_lang)
            # with open("/assets/translator/log.txt", "w") as f:
            #     f.write(f"{lang_type} {trg_lang}")

            tokenizer = None
            if lang_type == "kaz":
                tokenizer = self.kaz_tokenizer
            elif lang_type == "eng":
                tokenizer = self.eng_tokenizer

            tokenized_inputs = tokenizer(texts, return_tensors="pt")
           
            output = self.model.generate(**tokenized_inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(trg_lang), max_length=1000)
            translated_sentence = tokenizer.batch_decode(output, skip_special_tokens=True)
            translated_sentence = translated_sentence[0]

            output_tensor = pb_utils.Tensor("OUTPUT", np.array([translated_sentence], dtype=np.object_))
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )

            responses.append(inference_response)
            
        return responses
