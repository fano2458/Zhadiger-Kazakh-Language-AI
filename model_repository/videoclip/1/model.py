import os
import json
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
import clip

class TritonPythonModel:
    def initialize(self, args):
        # Config uses KIND_CPU, so force CPU to avoid accidental CUDA branches.
        self.device = "cpu"

        # Load CLIP on CPU
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Load precomputed embeddings (expects dicts with at least "embedding" and optional "path")
        embedding_folder = "/assets/videoclip/video_embeddings"
        self.embeddings_data = []
        if os.path.isdir(embedding_folder):
            for filename in os.listdir(embedding_folder):
                if filename.endswith(".pt"):
                    file_path = os.path.join(embedding_folder, filename)
                    meta = torch.load(file_path, map_location=self.device)
                    emb = meta["embedding"].to(self.device).float()
                    emb = emb / emb.norm(p=2)  # normalize for cosine similarity
                    meta["embedding"] = emb
                    if "path" not in meta:
                        meta["path"] = filename
                    self.embeddings_data.append(meta)
        # Optional: precompute a stacked tensor for faster dot products
        if self.embeddings_data:
            self.emb_matrix = torch.stack([m["embedding"] for m in self.embeddings_data], dim=0)  # [N, D]
        else:
            self.emb_matrix = None

    @torch.no_grad()
    def predict(self, text: str):
        # Encode text
        tokens = clip.tokenize([text]).to(self.device)
        text_feat = self.model.encode_text(tokens)[0].float()
        text_feat = text_feat / text_feat.norm(p=2)

        if self.emb_matrix is None or self.emb_matrix.numel() == 0:
            return [], []

        # Cosine similarity via dot product (both normalized)
        # sims: [N]
        sims = torch.mv(self.emb_matrix, text_feat).tolist()

        # Top-3 indices
        topk = min(3, len(sims))
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:topk]

        top_paths = [self.embeddings_data[i]["path"] for i in top_indices]
        top_scores = [float(sims[i]) for i in top_indices]
        return top_paths, top_scores

    def execute(self, requests):
        responses = []
        for request in requests:
            # Input "texts" is TYPE_STRING, dims [-1]. No batching at request level (max_batch_size=0).
            text_arr = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            # Handle bytes/str and potential vector length >1 by joining or taking first.
            texts = []
            for el in text_arr:
                if isinstance(el, (bytes, bytearray)):
                    texts.append(el.decode("utf-8"))
                else:
                    texts.append(str(el))
            # For this model we'll process the first string; adjust if you want multi-text queries.
            query = texts[0] if texts else ""

            top_paths, top_scores = self.predict(query)

            payload = json.dumps({"paths": top_paths, "scores": top_scores})
            # Triton STRING tensor => numpy array with dtype=object (each element is a str/bytes)
            output_tensor = pb_utils.Tensor("output", np.array([payload], dtype=object))

            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses
