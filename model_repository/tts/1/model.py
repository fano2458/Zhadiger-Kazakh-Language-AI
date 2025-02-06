import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import io
from scipy.io import wavfile
import numpy as np

torch.set_float32_matmul_precision('high')


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cpu")
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("/assets/tts/checkpoint")
        self.model = AutoModelForTextToWaveform.from_pretrained("/assets/tts/checkpoint").eval().to(self.device)

        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

    def preprocess_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt").to(self.device)
        return inputs

    @torch.no_grad()
    def generate_waveform(self, inputs):
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            output = self.model(**inputs).waveform
        return output

    def postprocess_waveform(self, waveform):
        output_numpy = waveform.squeeze().cpu().numpy().astype(np.float32)
        with io.BytesIO() as wav_io:
            wavfile.write(wav_io, rate=self.model.config.sampling_rate, data=output_numpy)
            wav_bytes = wav_io.getvalue()
        return wav_bytes

    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            texts = [el.decode() for el in texts][0]

            inputs = self.preprocess_text(texts)
            waveform = self.generate_waveform(inputs)
            wav_bytes = self.postprocess_waveform(waveform)

            output_tensor = pb_utils.Tensor("output", np.frombuffer(wav_bytes, dtype=np.uint8))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
