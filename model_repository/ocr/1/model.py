import base64
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image as PIL_Image
from io import BytesIO
import fitz 
import sys
import gc
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
        device = torch.device("cuda")
        self.det_processor, self.det_model = load_det_processor(), load_det_model(device=device, dtype=torch.float16)
        self.rec_processor, self.rec_model = load_rec_processor(), load_rec_model(device=device, dtype=torch.float16)

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
        
        with open("/assets/ocr_pdf_log.txt", "a") as f:
            f.write(f"page count: {pdf_document.page_count}\n")

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = PIL_Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        with open("/assets/ocr_pdf_log.txt", "a") as f:
            f.write(f"obtained images: {len(images)}\n")  

        pdf_document.close()
        return images

    def predict(self, image):
        predictions = run_ocr(
            image, [self.langs], 
            self.det_model, self.det_processor, 
            self.rec_model, self.rec_processor
        )
        return predictions
    
    def clean_gpu_memory(self):
        """Clean up GPU memory by emptying cache and garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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
            base64_images = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            
            formated_outputs = []

            for base64_image in base64_images:
                image_base64 = base64_image.decode('utf-8')
                if image_base64.startswith("JVBER"):
                    with open("/assets/ocr_pdf_log.txt", "w") as f:
                        f.write("pdf")
                    pdf_bytes = base64.b64decode(image_base64)
                    images = self.pdf_to_images(pdf_bytes)

                    with open("/assets/ocr_pdf_log.txt", "a") as f:
                        f.write(f"images: {len(images)}\n")

                    for image in images:
                        if image.mode != 'RGB':
                            image = image.convert("RGB")
                        try:
                            predictions = self.predict([image])
                        except Exception as e:
                            with open("/assets/ocr_pdf_log.txt", "a") as f:
                                f.write(f"error: {e}\n")
                            predictions = []

                        with open("/assets/ocr_pdf_log.txt", "a") as f:
                            f.write(f"predictions: {len(predictions)}\n")

                        formatted_text = self.format_predictions(predictions)

                        with open("/assets/ocr_pdf_log.txt", "a") as f:
                            f.write(f"formatted text: {formatted_text}\n")

                        formated_outputs.append(formatted_text)
                        
                        # Clean GPU memory after processing each image
                        self.clean_gpu_memory()
                else:
                    images = self.preprocess_image(image_base64)
                    predictions = self.predict([images])
                    formatted_text = self.format_predictions(predictions)
                    formated_outputs.append(formatted_text)
                    
                    # Clean GPU memory after processing each image
                    self.clean_gpu_memory()

            output_tensor = pb_utils.Tensor("output", np.array(formated_outputs, dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses