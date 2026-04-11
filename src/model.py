"""
DermaViT Model Architecture
Dual-branch fusion: EfficientNet-B0 (local) + Swin-T (global)
with Squeeze-and-Excitation channel attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import (
    NUM_CLASSES, DROPOUT, SE_REDUCTION_RATIO,
    LR_EFFICIENTNET, LR_SWIN, LR_FUSION
)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block with gated residual connection.
    
    Input: x of shape [B, C] (already pooled 1D features)
    Output: (sigmoid(FC2(ReLU(FC1(x)))) * x) + x
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced = max(channels // reduction_ratio, 1)
        self.fc1 = nn.Linear(channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced, channels, bias=True)

    def forward(self, x):
        # x: [B, C]
        w = F.relu(self.fc1(x))       # [B, C//r]
        w = torch.sigmoid(self.fc2(w))  # [B, C]
        return (w * x) + x             # gated residual


class DualScopeFusionBlock(nn.Module):
    """
    Dual-scope fusion: concatenate local and global features,
    then apply SE channel attention.
    
    F_cat = Concat(F_L, F_G)  → [B, 2048]
    F_fused = SE(F_cat)        → [B, 2048]
    """

    def __init__(self, input_dim: int = 2048, reduction_ratio: int = 16):
        super().__init__()
        self.se = SEBlock(input_dim, reduction_ratio)

    def forward(self, f_local, f_global):
        print(f"DEBUG: f_local dim={f_local.dim()} shape={f_local.shape}")
        print(f"DEBUG: f_global dim={f_global.dim()} shape={f_global.shape}")
        f_cat = torch.cat([f_local, f_global], dim=1)  # [B, 2048]
        f_fused = self.se(f_cat)                         # [B, 2048]
        return f_fused


class DermaViT(nn.Module):
    """
    DermaViT: Dual-branch skin lesion classification model.
    
    Branch 1 — EfficientNet-B0 (local features):
        Output: F_L ∈ R^1280
    
    Branch 2 — Swin-T (global features):
        Output: F_G ∈ R^768
    
    Fusion:
        DualScopeFusionBlock(2048) → F_fused ∈ R^2048
    
    Classifier:
        BatchNorm1d(2048) → Dropout(0.4) → Linear(2048, 7)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT,
                 se_reduction: int = SE_REDUCTION_RATIO, pretrained: bool = True):
        super().__init__()

        # ── Branch 1: EfficientNet-B0 (local features) ──
        self.efficientnet = timm.create_model(
            'efficientnet_b0', pretrained=pretrained
        )
        # Remove classifier and global pool — we'll do GAP manually
        self.efficientnet.classifier = nn.Identity()
        self.efficientnet.global_pool = nn.Identity()
        # EfficientNet-B0 outputs 1280 channels after conv_head

        # ── Branch 2: Swin-T (global features) ──
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=pretrained
        )
        # Remove classification head
        self.swin.head = nn.Identity()
        # Swin-T outputs 768-dim features

        # ── Fusion ──
        # EfficientNet: 1280 + Swin: 768 = 2048
        self.fusion = DualScopeFusionBlock(
            input_dim=1280 + 768, reduction_ratio=se_reduction
        )

        # ── Classifier Head ──
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        # Branch 1: EfficientNet-B0
        eff_features = self.efficientnet(x)  # [B, 1280, H, W] or [B, 1280*H*W]
        if eff_features.dim() == 4:
            f_local = F.adaptive_avg_pool2d(eff_features, 1).flatten(1)  # [B, 1280]
        elif eff_features.dim() == 3:
            f_local = eff_features.mean(dim=1)  # [B, 1280]
        else:
            f_local = eff_features  # [B, 1280] already pooled

        # Branch 2: Swin-T
        f_global = self.swin(x)  # [B, 7, 7, 768] or [B, N, 768]
        if f_global.dim() == 4:
            f_global = f_global.mean(dim=(1, 2))  # [B, 768]
        elif f_global.dim() == 3:
            f_global = f_global.mean(dim=1)  # [B, 768]

        # Fusion
        f_fused = self.fusion(f_local, f_global)  # [B, 2048]

        # Classifier
        logits = self.classifier(f_fused)  # [B, num_classes]
        return logits

    def get_param_groups(self):
        """Return parameter groups with differential learning rates."""
        return [
            {'params': self.efficientnet.parameters(), 'lr': LR_EFFICIENTNET},
            {'params': self.swin.parameters(), 'lr': LR_SWIN},
            {
                'params': list(self.fusion.parameters()) +
                          list(self.classifier.parameters()),
                'lr': LR_FUSION
            },
        ]
