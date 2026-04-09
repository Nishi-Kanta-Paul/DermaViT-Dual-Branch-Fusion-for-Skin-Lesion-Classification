import torch
from DermaViT.model import DermaViT

def test():
    model = DermaViT(num_classes=7)
    x = torch.randn(2, 3, 224, 224)
    model(x)

if __name__ == "__main__":
    test()
