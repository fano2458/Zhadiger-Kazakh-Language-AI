import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import io
from scipy.io import wavfile
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("/assets/tts/checkpoint")
        self.model = AutoModelForTextToWaveform.from_pretrained("/assets/tts/checkpoint").to(self.device)

    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "TEXTS").as_numpy()
            texts = [el.decode() for el in texts][0]

            inputs = self.tokenizer(texts, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model(**inputs).waveform

            # Convert the waveform tensor to a numpy array with the appropriate data type
            output_numpy = output.squeeze().cpu().numpy().astype(np.float32)

            with io.BytesIO() as wav_io:
                wavfile.write(wav_io, rate=self.model.config.sampling_rate, data=output_numpy)
                wav_bytes = wav_io.getvalue()

            # Create an output tensor
            output_tensor = pb_utils.Tensor("OUTPUT", np.frombuffer(wav_bytes, dtype=np.uint8))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )

            responses.append(inference_response)

        return responses
