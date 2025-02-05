import base64
import torch
import numpy as np
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils

import sys
sys.path.append('/assets/kazclip')
from text_encoder import TextTokenizer
from kazclip_model import KazClip


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = KazClip()
        self.model.load_state_dict(torch.load("/assets/kazclip/checkpoint/model.pt", map_location=self.device))
        self.model.eval().to(self.device)

        self.tokenizer = TextTokenizer()
        self.image_embeddings = torch.load("/assets/kazclip/precomputed_image_embeddings.pt", map_location=self.device)
        self.image_paths = torch.load("/assets/kazclip/image_paths.pt")

    def predict(self, texts):
        tokens = self.tokenizer(texts)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            _, text_features = self.model(None, tokens)

        text_features = F.normalize(text_features, dim=-1)
        scores = text_features @ self.image_embeddings.t()

        top5_indices = scores.squeeze().topk(5).indices
        top5_images = [self.image_paths[i] for i in top5_indices]

        encoded_images = []
        for image_path in top5_images:
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")
                encoded_images.append(encoded_image)

        return encoded_images

    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            texts = [el.decode() for el in texts][0]

            top5_images = self.predict(texts)

            output_tensor = pb_utils.Tensor("output", np.array([top5_images], dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
