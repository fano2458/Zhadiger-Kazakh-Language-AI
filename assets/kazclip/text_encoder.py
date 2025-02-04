import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class TextTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("kz-transformers/kaz-roberta-conversational")

    def __call__(self, x):
        return self.tokenizer(x, return_tensors="pt", padding="max_length", max_length=64, truncation=True)


class TextEncoder(nn.Module):
    def __init__(self, projection_dim=256):
        super(TextEncoder, self).__init__()
        config = AutoConfig.from_pretrained("kz-transformers/kaz-roberta-conversational")
        self.encoder = AutoModel.from_config(config)
        self.projection_head = nn.Linear(self.encoder.config.hidden_size, projection_dim)

    def forward(self, x):
        out = self.encoder(**x)
        text_features = out.pooler_output
        projection = self.projection_head(text_features)

        return projection
