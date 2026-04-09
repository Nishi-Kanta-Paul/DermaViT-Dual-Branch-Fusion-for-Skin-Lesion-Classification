import torch
import timm
from DermaViT.model import DermaViT
print(DermaViT)
model = DermaViT(num_classes=7)
x = torch.randn(2, 3, 224, 224)
model(x)
