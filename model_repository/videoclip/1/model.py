import base64
import torch
import numpy as np
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils

import sys
sys.path.append('/assets/videoclip')
from text_encoder import TextTokenizer
from videoclip_model import VideoCLIP

torch.set_float32_matmul_precision('high') 


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cpu"

        self.model = VideoCLIP()
        self.model.load_state_dict(torch.load("/assets/kazclip/checkpoint/model.pt", map_location=self.device))

        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.model.eval().to(self.device)

        self.tokenizer = TextTokenizer()
        self.image_embeddings = torch.load("/assets/videoclip/precomputed_frames_embeddings.pt", map_location=self.device)
        self.image_paths = torch.load("/assets/videoclip/frame_paths.pt")
   
    @torch.no_grad()
    def predict(self, texts):
        tokens = self.tokenizer(texts)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # with torch.autocast(device_type=self.device, dtype=torch.bfloat16):  # need to measure the performance to decide whether to use bf16 or fp16 on both CPU and GPU
        _, text_features = self.model(None, tokens)

        text_features = F.normalize(text_features, dim=-1)
        scores = text_features @ self.image_embeddings.t()

        top3_indices = scores.squeeze().topk(3).indices
        top3_images = [f"{self.image_paths[i]}" for i in top3_indices]

    
        return top3_images

    def execute(self, requests):
        responses = []

        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            texts = [el.decode() for el in texts][0]

            top3_images = self.predict(texts)

            output_tensor = pb_utils.Tensor("output", np.array([top3_images], dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
