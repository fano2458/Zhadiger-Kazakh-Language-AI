import numpy as np
import triton_python_backend_utils as pb_utils
from vosk import Model, KaldiRecognizer
import wave
import json
import io
from scipy.io import wavfile
import base64


class TritonPythonModel:
    def initialize(self, args):
        self.model_path = "/assets/stt/checkpoint"  
        self.model = Model(self.model_path)

    def execute(self, requests):
        responses = []

        for request in requests:
            # Retrieve input audio waveform
            audio_input = pb_utils.get_input_tensor_by_name(request, "AUDIO").as_numpy()
            encoded_audio_waveform = audio_input[0].decode('utf-8')

            # Decode the base64 encoded audio waveform
            audio_waveform = base64.b64decode(encoded_audio_waveform)

            # Convert bytes to a file-like object
            audio_waveform_io = io.BytesIO(audio_waveform)
            rate, data = wavfile.read(audio_waveform_io)

            recognizer = KaldiRecognizer(self.model, rate)

            if recognizer.AcceptWaveform(data.tobytes()):
                result = json.loads(recognizer.Result())
                transcription = result.get("text", "")
            else:
                transcription = recognizer.PartialResult()

            output_tensor = pb_utils.Tensor(
                "OUTPUT",
                np.array([transcription], dtype=object)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )

            responses.append(inference_response)

        return responses
