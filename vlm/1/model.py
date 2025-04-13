import triton_python_backend_utils as pb_utils
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
import torch
import os
from huggingface_hub import login

torch.set_float32_matmul_precision('high')


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device('cpu')
        self.authenticate()
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
        self.model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it").to(self.device).eval()

    def authenticate(self):
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)
        else:
            print("WARNING: HUGGINGFACE_TOKEN environment variable not set. Authentication may fail for gated models.")

    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            images = pb_utils.get_input_tensor_by_name(request, "images").as_numpy()
            images = [el.decode() for el in images][0]
            texts = [el.decode() for el in texts][0]

            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inference_result = self.model.generate(**inputs)
            inference_result = self.processor.batch_decode(inference_result, skip_special_tokens=True)[0]

            output_tensor = pb_utils.Tensor("output", np.array([inference_result], dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
