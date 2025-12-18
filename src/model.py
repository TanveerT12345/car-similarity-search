import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    # This model converts an input car image into a fixed-length embedding vector.
    # The key idea: images of the same car class should end up close together in embedding space,
    # so we can do nearest-neighbor search (FAISS) to find "similar" cars.
    def __init__(self, embed_dim: int = 512):
        super().__init__()

        # Start from a pretrained ResNet50 backbone (ImageNet) so we don’t train from scratch.
        # Pretraining gives strong general visual features (edges/textures/shapes) that transfer well to cars.
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove the original ImageNet classification layer.
        # We want feature vectors, not 1000-class logits.
        in_dim = m.fc.in_features
        m.fc = nn.Identity()

        self.backbone = m

        # Projection head: maps ResNet features -> embedding dimension used for retrieval.
        # embed_dim must match what you use in training, indexing, and searching.
        self.head = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        # 1) Extract high-level visual features with ResNet
        feats = self.backbone(x)

        # 2) Project features into the retrieval embedding space
        emb = self.head(feats)

        # 3) L2-normalize so cosine similarity is consistent across images
        # (this makes nearest-neighbor re
::contentReference[oaicite:0]{index=0}
