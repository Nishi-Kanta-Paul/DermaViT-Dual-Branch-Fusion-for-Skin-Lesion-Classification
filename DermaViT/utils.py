"""
DermaViT Utilities
Seed setting, class weights, checkpointing, plotting, and reporting.
"""
import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_weights(labels, num_classes: int):
    """
    Compute inverse-frequency class weights: w_c = N / (C * n_c).
    
    Args:
        labels: array-like of integer labels
        num_classes: number of classes
    
    Returns:
        torch.FloatTensor of shape [num_classes]
    """
    labels = np.array(labels)
    N = len(labels)
    weights = []
    for c in range(num_classes):
        n_c = np.sum(labels == c)
        if n_c == 0:
            weights.append(1.0)
        else:
            weights.append(N / (num_classes * n_c))
    return torch.FloatTensor(weights)


def save_checkpoint(model, optimizer, epoch, val_f1, path):
    """Save model checkpoint with optimizer state, epoch, and best val F1."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
    }, path)
    print(f"  ✓ Checkpoint saved to {path} (epoch {epoch}, val_f1={val_f1:.4f})")


def load_checkpoint(model, optimizer, path):
    """
    Load model checkpoint.
    
    Returns:
        model, optimizer, epoch, best_val_f1
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_f1 = checkpoint['val_f1']
    print(f"  ✓ Checkpoint loaded from {path} (epoch {epoch}, val_f1={best_val_f1:.4f})")
    return model, optimizer, epoch, best_val_f1


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot loss and accuracy curves side by side and save as PNG."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='#2196F3', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', color='#F44336', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc', color='#2196F3', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', color='#F44336', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training curves saved to {save_path}")


def print_classification_report(y_true, y_pred, class_names):
    """Print sklearn classification_report with per-class P, R, F1."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)
    return report
