import base64
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image as PIL_Image
from io import BytesIO
import fitz  # PyMuPDF
import os
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.det_processor, self.det_model = load_det_processor(), load_det_model(device=device, dtype=torch.float32)
        self.rec_processor, self.rec_model = load_rec_processor(), load_rec_model(device=device, dtype=torch.float32)

    def preprocess_image(self, image_base64):
        image_bytes = base64.b64decode(image_base64)
        image_stream = BytesIO(image_bytes)
        image = PIL_Image.open(image_stream)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        return image
    
    def pdf_to_images(self, pdf_bytes):
        """Convert PDF to individual images."""
        images = []
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = PIL_Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        pdf_document.close()
        return images

    def predict(self, images):
        predictions = []
        for image in images:
            result = run_ocr(
                [image], [self.langs], 
                self.det_model, self.det_processor, 
                self.rec_model, self.rec_processor
            )
            predictions.append(result)
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
            base64_data = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            data_base64 = base64_data[0].decode('utf-8')
            
            # Check if input is a PDF or an image
            if data_base64.startswith("JVBER"):
                pdf_bytes = base64.b64decode(data_base64)
                images = self.pdf_to_images(pdf_bytes)
            else:
                images = [self.preprocess_image(data_base64)]

            predictions = self.predict(images)
            formatted_text = self.format_predictions(predictions)

            output_tensor = pb_utils.Tensor("output", np.array(formated_outputs, dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
