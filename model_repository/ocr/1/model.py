import base64
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image as PIL_Image
from io import BytesIO
import sys
sys.path.append('/assets/ocr')

from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor


class TritonPythonModel:
    def initialize(self, args):
        self.load_models_and_processors()

    def load_models_and_processors(self):
        self.langs = ["kk", "ru", "en"]
        self.det_processor, self.det_model = load_det_processor(), load_det_model(device="cpu", dtype=torch.float32)
        self.rec_model, self.rec_processor = load_rec_model(device='cpu', dtype=torch.float32), load_rec_processor()

    def preprocess_image(self, image_base64):
        image_bytes = base64.b64decode(image_base64)
        image_stream = BytesIO(image_bytes)
        image = PIL_Image.open(image_stream)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        return image

    def predict(self, image):
        predictions = run_ocr(
            [image], [self.langs], 
            self.det_model, self.det_processor, 
            self.rec_model, self.rec_processor
        )
        return predictions

    def format_predictions(self, predictions):
        formatted_text = ""
        for result in predictions:
            text_lines = result.text_lines
            for line in text_lines:
                if line.confidence >= 0.50:
                    formatted_text += line.text + "\n"
        return formatted_text

    def execute(self, requests):
        responses = []

        for request in requests:
            base64_images = pb_utils.get_input_tensor_by_name(request, "IMAGES").as_numpy()
            image_base64 = base64_images[0].decode('utf-8')
            image = self.preprocess_image(image_base64)
            predictions = self.predict(image)
            formatted_text = self.format_predictions(predictions)

            output_tensor = pb_utils.Tensor("OUTPUT", np.array([formatted_text], dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
    