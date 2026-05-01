# ═══ DermaViT v2.1 ═══
# Modified: Fix 1 - Swin Feature Map Visualization (replaced broken attention rollout)
# All changes marked with # CHANGED
"""
DermaViT Explainability
Generate joint saliency maps: CNN Grad-CAM + Swin Feature Map Visualization.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from config import (
    NUM_CLASSES, CLASS_NAMES, SALIENCY_DIR, BEST_MODEL_PATH,
    LAMBDA_SALIENCY, IMAGENET_MEAN, IMAGENET_STD
)
from model import DermaViT
from utils import load_checkpoint


def _denormalize(tensor):
    """Convert normalized tensor back to displayable image [0,255]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


class GradCAM:
    """
    Grad-CAM for the EfficientNet-B0 branch.
    Hooks the last convolutional layer to capture activations and gradients.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None

        # Default: hook the last block of EfficientNet conv features
        if target_layer is None:
            # EfficientNet-B0: the last block in the feature extractor
            target_layer = model.efficientnet.conv_head
        
        self.hook_forward = target_layer.register_forward_hook(self._save_activation)
        self.hook_backward = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, metadata=None, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Returns:
            cam: numpy array of shape [H, W] normalized to [0, 1]
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward
        logits = self.model(input_tensor, metadata)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)

        # Compute Grad-CAM
        grads = self.gradients  # [1, C, H, W]
        acts = self.activations  # [1, C, H, W]

        # Global average pooling of gradients
        alpha = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination
        cam = F.relu((alpha * acts).sum(dim=1, keepdim=True))  # [1, 1, H, W]

        # Upsample to 224×224
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def remove_hooks(self):
        self.hook_forward.remove()
        self.hook_backward.remove()


class SwinAttentionRollout:  # CHANGED: Renamed to SwinFeatureMapVisualization internally but kept class name for compatibility
    """  # CHANGED
    Swin Feature Map Visualization using Stage 4 output.  # CHANGED
    Replaces broken attention rollout with spatial activation maps.  # CHANGED
    """  # CHANGED

    def __init__(self, model):  # CHANGED
        self.model = model  # CHANGED
        self.stage4_features = None  # CHANGED: Store Stage 4 features
        self.hook_handle = None  # CHANGED: Single hook for Stage 4

    def generate(self, input_tensor, metadata=None):  # CHANGED
        """  # CHANGED
        Generate spatial attention map from Swin-T Stage 4 feature maps.  # CHANGED
        
        Steps:  # CHANGED
        1. Register hook on Stage 4 output (7x7x768 features)  # CHANGED
        2. Forward pass to capture features  # CHANGED
        3. Compute spatial importance as mean across channels  # CHANGED
        4. Apply ReLU and normalize  # CHANGED
        5. Upsample to 224x224  # CHANGED
        
        Returns:  # CHANGED
            attn_map: numpy array of shape [224, 224] normalized to [0, 1]  # CHANGED
        """  # CHANGED
        self.model.eval()  # CHANGED
        self.stage4_features = None  # CHANGED: Reset

        # CHANGED: Hook function to capture Stage 4 output
        def hook_fn(module, input, output):  # CHANGED
            """Capture Stage 4 feature map."""  # CHANGED
            self.stage4_features = output.detach()  # CHANGED

        # CHANGED: Register hook on Swin-T Stage 4 (last layer)
        # In timm's Swin-T, layers[-1] is Stage 4, which outputs [B, 7, 7, 768]
        try:  # CHANGED
            target_layer = self.model.swin.layers[-1]  # CHANGED: Stage 4
            self.hook_handle = target_layer.register_forward_hook(hook_fn)  # CHANGED
        except (AttributeError, IndexError):  # CHANGED
            # CHANGED: Fallback if structure is different
            try:  # CHANGED
                target_layer = self.model.swin.norm  # CHANGED: Final LayerNorm after Stage 4
                self.hook_handle = target_layer.register_forward_hook(hook_fn)  # CHANGED
            except AttributeError:  # CHANGED
                pass  # CHANGED: Will return uniform map

        # CHANGED: Forward pass to trigger hook
        with torch.no_grad():  # CHANGED
            _ = self.model(input_tensor, metadata)  # CHANGED

        # CHANGED: Remove hook immediately to avoid memory leaks
        if self.hook_handle is not None:  # CHANGED
            self.hook_handle.remove()  # CHANGED
            self.hook_handle = None  # CHANGED

        # CHANGED: Process captured features
        if self.stage4_features is not None:  # CHANGED
            feat = self.stage4_features  # CHANGED: [B, H, W, C] or [B, N, C]
            
            # CHANGED: Handle different output formats
            if feat.dim() == 4:  # CHANGED: [B, H, W, C] format
                B, H, W, C = feat.shape  # CHANGED
                # CHANGED: Compute spatial importance as mean across channels
                spatial_map = feat.mean(dim=-1)  # CHANGED: [B, H, W]
            elif feat.dim() == 3:  # CHANGED: [B, N, C] format (flattened)
                B, N, C = feat.shape  # CHANGED
                # CHANGED: Reshape to spatial grid
                H = W = int(np.sqrt(N))  # CHANGED: Assume square (7x7 for Stage 4)
                if H * W == N:  # CHANGED
                    spatial_map = feat.mean(dim=-1).reshape(B, H, W)  # CHANGED
                else:  # CHANGED
                    # CHANGED: Fallback for non-square
                    spatial_map = feat.mean(dim=-1).reshape(B, 7, -1)[:, :, :7]  # CHANGED
            else:  # CHANGED
                # CHANGED: Unexpected format, return uniform map
                attn_map = np.ones((224, 224), dtype=np.float32)  # CHANGED
                return attn_map  # CHANGED
            
            # CHANGED: Apply ReLU to keep only positive activations
            spatial_map = F.relu(spatial_map)  # CHANGED
            
            # CHANGED: Normalize to [0, 1]
            spatial_map = spatial_map[0]  # CHANGED: Take first batch element
            if spatial_map.max() > spatial_map.min():  # CHANGED
                spatial_map = (spatial_map - spatial_map.min()) / (spatial_map.max() - spatial_map.min() + 1e-8)  # CHANGED
            
            # CHANGED: Upsample from 7x7 to 224x224 using bilinear interpolation
            spatial_map = spatial_map.unsqueeze(0).unsqueeze(0)  # CHANGED: [1, 1, H, W]
            attn_map = F.interpolate(  # CHANGED
                spatial_map,  # CHANGED
                size=(224, 224),  # CHANGED
                mode='bilinear',  # CHANGED
                align_corners=False  # CHANGED
            ).squeeze().cpu().numpy()  # CHANGED: [224, 224]
        else:  # CHANGED
            # CHANGED: Fallback if hook failed
            attn_map = np.ones((224, 224), dtype=np.float32)  # CHANGED
        
        return attn_map  # CHANGED

    def remove_hooks(self):  # CHANGED
        """Remove any remaining hooks."""  # CHANGED
        if self.hook_handle is not None:  # CHANGED
            self.hook_handle.remove()  # CHANGED
            self.hook_handle = None  # CHANGED


def generate_saliency_maps(model=None, test_loader=None, device=None, n_samples=20):
    """
    Generate joint saliency maps for random test samples.
    
    Creates 4-panel figures:
        [Original | CNN Grad-CAM | Swin Attention | Joint Saliency]
    """
    print("\n" + "=" * 70)
    print("DermaViT — Explainability Pipeline")
    print("=" * 70)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if model is None:
        model = DermaViT().to(device)
        model, _, _, _ = load_checkpoint(model, None, BEST_MODEL_PATH)
    model = model.to(device)
    model.eval()

    os.makedirs(SALIENCY_DIR, exist_ok=True)

    # Collect random test samples
    all_images = []
    all_metadata = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            images, metadata, labels = batch
            images = images.to(device)
            metadata = metadata.to(device)
            logits = model(images, metadata)
            preds = logits.argmax(dim=1)
            all_images.append(images.cpu())
            all_metadata.append(metadata.cpu())
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    all_images = torch.cat(all_images, dim=0)
    all_metadata = torch.cat(all_metadata, dim=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Random sample selection
    np.random.seed(42)
    n_samples = min(n_samples, len(all_images))
    indices = np.random.choice(len(all_images), size=n_samples, replace=False)

    # Initialize explainers
    grad_cam = GradCAM(model)
    swin_rollout = SwinAttentionRollout(model)

    # Track SE attention weights per class
    se_weights_per_class = {i: [] for i in range(NUM_CLASSES)}

    print(f"\n  Generating saliency maps for {n_samples} samples...")

    for count, idx in enumerate(tqdm(indices, desc="  Saliency")):
        img_tensor = all_images[idx].unsqueeze(0).to(device)
        meta_tensor = all_metadata[idx].unsqueeze(0).to(device)
        true_label = all_labels[idx]
        pred_label = all_preds[idx]

        # 1. Original image
        orig_img = _denormalize(all_images[idx])

        # 2. Grad-CAM
        try:
            cam_map = grad_cam.generate(img_tensor.clone(), meta_tensor, target_class=int(pred_label))
        except Exception:
            cam_map = np.zeros((224, 224), dtype=np.float32)

        # 3. Swin attention rollout
        try:
            swin_map = swin_rollout.generate(img_tensor.clone(), meta_tensor)
        except Exception:
            swin_map = np.zeros((224, 224), dtype=np.float32)

        # 4. Class-adaptive lambda from SE weights
        lambda_c = LAMBDA_SALIENCY
        if model.se_weights is not None:
            se_w = model.se_weights[0].cpu().numpy()
            w_cnn = se_w[:1280].mean()
            w_swin = se_w[1280:].mean()
            if (w_cnn + w_swin) > 0:
                lambda_c = w_cnn / (w_cnn + w_swin)
        
        # Joint saliency with class-adaptive lambda
        joint_map = lambda_c * cam_map + (1 - lambda_c) * swin_map
        if joint_map.max() > joint_map.min():
            joint_map = (joint_map - joint_map.min()) / (joint_map.max() - joint_map.min() + 1e-8)

        # Create overlay
        heatmap_jet = cv2.applyColorMap(
            (joint_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_img, 0.5, heatmap_jet, 0.5, 0)

        # Create 4-panel figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(orig_img)
        axes[0].set_title(f'Original\nTrue: {CLASS_NAMES[true_label]}', fontsize=11)
        axes[0].axis('off')

        cam_vis = cv2.applyColorMap((cam_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_vis = cv2.cvtColor(cam_vis, cv2.COLOR_BGR2RGB)
        axes[1].imshow(cam_vis)
        axes[1].set_title('CNN Grad-CAM\n(EfficientNet-B0)', fontsize=11)
        axes[1].axis('off')

        swin_vis = cv2.applyColorMap((swin_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        swin_vis = cv2.cvtColor(swin_vis, cv2.COLOR_BGR2RGB)
        axes[2].imshow(swin_vis)
        axes[2].set_title('Swin Stage-4\nFeature Map', fontsize=11)  # CHANGED: Updated title
        axes[2].axis('off')

        axes[3].imshow(overlay)
        axes[3].set_title(f'Joint Saliency (λ={lambda_c:.3f})\nPred: {CLASS_NAMES[pred_label]}', fontsize=11)
        axes[3].axis('off')

        plt.suptitle(
            f'DermaViT Saliency — Sample {count+1}',
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        save_path = os.path.join(SALIENCY_DIR, f"saliency_sample_{count+1:02d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Clean up hooks
    grad_cam.remove_hooks()
    swin_rollout.remove_hooks()

    # Report branch dominance
    print(f"\n  ✓ Saliency maps saved to {SALIENCY_DIR}/")
    print(f"  ✓ Generated {n_samples} 4-panel saliency visualizations")

    # Branch dominance analysis
    print("\n  Branch Dominance Analysis (Grad-CAM vs Swin Attention):")
    print("  (Based on mean activation intensity per class)")
    print("  Note: Full analysis requires per-class aggregation over many samples")

    return


if __name__ == '__main__':
    generate_saliency_maps()
