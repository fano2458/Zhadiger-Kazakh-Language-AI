import sys
sys.path.append('/assets/kazclip')
from text_encoder import TextEncoder
from visual_encoder import VisualEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F


class KazClip(nn.Module):
    def __init__(self, projection_dim=256):
        super(KazClip, self).__init__()
        
        self.visual_encoder = VisualEncoder(projection_dim)
        self.text_encoder = TextEncoder(projection_dim)
        self.logit_scale = nn.Parameter(torch.ones(()))

    def forward(self, image, text):
        visual_features = None
        text_features = None
        if image is not None:
            visual_features = self.visual_encoder(image)
        if text is not None:
            text_features = self.text_encoder(text)

        return visual_features, text_features

    def compute_loss(self, visual_features, text_features):
        visual_features = F.normalize(visual_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute logits with scaling
        logits = visual_features @ text_features.t() * torch.exp(self.logit_scale)
        batch_size = logits.shape[0]
        ground_truth = torch.arange(batch_size, device=logits.device)

        loss = (F.cross_entropy(logits, ground_truth) + F.cross_entropy(logits.t(), ground_truth)) / 2
        return loss