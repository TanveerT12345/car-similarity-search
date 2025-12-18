import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        feats = self.backbone(x)
        emb = self.head(feats)
        return F.normalize(emb, p=2, dim=1)
