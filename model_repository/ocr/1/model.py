import base64
import pickle
import torch
import numpy as np
from argparse import Namespace
import triton_python_backend_utils as pb_utils
from PIL import Image as PIL_Image
from io import BytesIO
import sys
sys.path.append('/assets/ocr')
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.language_utils import convert_vector_idx2word
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor


class TritonPythonModel:
    def initialize(self, args):
        """Load models and processors during Triton model initialization."""
        self.langs = ["kk", "ru", "en"]
        self.det_processor, self.det_model = load_det_processor(), load_det_model()
        self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()

    def execute(self, requests):
        """Execute inference on input requests."""
        responses = []

        for request in requests:
            base64_images = pb_utils.get_input_tensor_by_name(request, "IMAGES").as_numpy()
            decoded_images = [base64.b64decode(img) for img in base64_images]

            formatted_texts = []
            for img_data in decoded_images:
                image_stream = BytesIO(img_data)
                image = PIL_Image.open(image_stream)

                if image.mode != 'RGB':
                    image = image.convert("RGB")

                predictions = run_ocr(
                    [image], [self.langs], 
                    self.det_model, self.det_processor, 
                    self.rec_model, self.rec_processor
                )

                formatted_text = ""
                for result in predictions:
                    text_lines = result.text_lines
                    for line in text_lines:
                        if line.confidence >= 0.50:  
                            formatted_text += line.text + "\n"

                formatted_texts.append(formatted_text)

            output_tensor = pb_utils.Tensor(
                "OUTPUT", 
                np.array(formatted_texts, dtype=np.object_)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses