import re
import json
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForTokenClassification


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.load_labels()

    def load_model(self):
        model_checkpoint = "/assets/ner/checkpoint"
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(model_checkpoint).to(self.device)

    def load_labels(self):
        self.labels_dict = {0:"O", 1:"B-ADAGE", 2:"I-ADAGE", 3:"B-ART", 4:"I-ART", 5:"B-CARDINAL",
                6:"I-CARDINAL", 7:"B-CONTACT", 8:"I-CONTACT", 9:"B-DATE", 10:"I-DATE", 11:"B-DISEASE",
                12:"I-DISEASE", 13:"B-EVENT", 14:"I-EVENT", 15:"B-FACILITY", 16:"I-FACILITY",
                17:"B-GPE", 18:"I-GPE", 19:"B-LANGUAGE", 20:"I-LANGUAGE", 21:"B-LAW", 22:"I-LAW",
                23:"B-LOCATION", 24:"I-LOCATION", 25:"B-MISCELLANEOUS", 26:"I-MISCELLANEOUS",
                27:"B-MONEY", 28:"I-MONEY", 29:"B-NON_HUMAN", 30:"I-NON_HUMAN", 31:"B-NORP",
                32:"I-NORP", 33:"B-ORDINAL", 34:"I-ORDINAL", 35:"B-ORGANISATION", 36:"I-ORGANISATION",
                37:"B-PERCENTAGE", 38:"I-PERCENTAGE", 39:"B-PERSON", 40:"I-PERSON", 41:"B-POSITION",
                42:"I-POSITION", 43:"B-PRODUCT", 44:"I-PRODUCT", 45:"B-PROJECT", 46:"I-PROJECT",
                47:"B-QUANTITY", 48:"I-QUANTITY", 49:"B-TIME", 50:"I-TIME"}

    def preprocess_text(self, texts):
        tokenized_inputs = self.tokenizer(texts, return_tensors="pt").to(self.device)
        return tokenized_inputs

    def predict(self, tokenized_inputs):
        with torch.no_grad():
            output = self.model(**tokenized_inputs)
        predictions = np.argmax(output.logits.detach().cpu().numpy(), axis=2)
        return predictions

    def postprocess_predictions(self, texts, tokenized_inputs, predictions):
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        previous_word_idx = None
        labels = []

        for i, p in zip(word_ids, predictions[0]):
            if i is None or i == previous_word_idx:
                continue
            elif i != previous_word_idx:
                try:
                    labels.append(self.labels_dict[p][2:])
                except:
                    labels.append(self.labels_dict[p])
            previous_word_idx = i

        input_sent_tokens = re.findall(r"[\w’-]+|[.,#?!)(\]\[;:–—\"«№»/%&']", texts)
        assert len(input_sent_tokens) == len(labels), f"Mismatch between input token and label sizes! {len(input_sent_tokens), len(labels)}, input: {input_sent_tokens}labels: {labels}"
        result_dict = {}
        words = []
        classes = []
        for t, l in zip(input_sent_tokens, labels):
            result_dict[t] = l
            words.append(t)
            classes.append(l)

        result_dict['words'] = words
        result_dict['classes'] = classes

        return json.dumps(result_dict, indent=4)

    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            texts = [el.decode() for el in texts][0]

            tokenized_inputs = self.preprocess_text(texts)
            predictions = self.predict(tokenized_inputs)
            result = self.postprocess_predictions(texts, tokenized_inputs, predictions)

            output_tensor = pb_utils.Tensor("output", np.array(result, dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
    