"""
DermaViT Evaluation
Load best checkpoint and run full evaluation on test set.
Generates classification report, confusion matrix, and ROC curves.
"""
import os
import numpy as np
import torch
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
)
from tqdm import tqdm

from config import (
    NUM_CLASSES, CLASS_NAMES, BEST_MODEL_PATH, RESULTS_DIR
)
from model import DermaViT
from utils import load_checkpoint


def evaluate(model=None, test_loader=None, device=None):
    """
    Full evaluation on test set.
    
    Args:
        model: trained DermaViT model (if None, loads from checkpoint)
        test_loader: test DataLoader
        device: torch device
    
    Returns:
        dict of summary metrics
    """
    print("\n" + "=" * 70)
    print("DermaViT — Evaluation Pipeline")
    print("=" * 70)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model from checkpoint if needed
    if model is None:
        model = DermaViT().to(device)
        model, _, _, _ = load_checkpoint(model, None, BEST_MODEL_PATH)
    model = model.to(device)
    model.eval()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Collect predictions ──
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Evaluating"):
            images = images.to(device, non_blocking=True)

            with autocast():
                logits = model(images)

            probs = torch.softmax(logits.float(), dim=1)
            _, preds = logits.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── 1. Classification Report ──
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES, digits=4
    )
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)

    # Save to file
    report_path = os.path.join(RESULTS_DIR, "evaluate_results.txt")
    with open(report_path, 'w') as f:
        f.write("DermaViT — Classification Report\n")
        f.write("=" * 70 + "\n")
        f.write(report)
    print(f"  ✓ Report saved to {report_path}")

    # ── 2. Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5,
        annot_kws={'size': 12}
    )
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('True', fontsize=13, fontweight='bold')
    ax.set_title('DermaViT — Confusion Matrix', fontsize=15, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved to {cm_path}")

    # ── 3. ROC Curves (One-vs-Rest) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#E91E63', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#FF5722']

    macro_auc_list = []
    for i in range(NUM_CLASSES):
        y_true_binary = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        macro_auc_list.append(roc_auc)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('DermaViT — ROC Curves (One-vs-Rest)', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curves saved to {roc_path}")

    # ── 4. Summary Metrics ──
    overall_acc = accuracy_score(all_labels, all_preds) * 100
    macro_prec = precision_score(all_labels, all_preds, average='macro') * 100
    macro_rec = recall_score(all_labels, all_preds, average='macro') * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    macro_auc_val = np.mean(macro_auc_list) * 100

    summary = {
        'accuracy': overall_acc,
        'precision': macro_prec,
        'recall': macro_rec,
        'f1_score': macro_f1,
        'auc': macro_auc_val,
    }

    print("\n┌─────────────────────────────────┐")
    print("│ DermaViT — Test Results         │")
    print("├─────────────────────────────────┤")
    print(f"│ Accuracy:  {overall_acc:.1f}%{' '*(19-len(f'{overall_acc:.1f}'))}│")
    print(f"│ Precision: {macro_prec:.1f}%{' '*(19-len(f'{macro_prec:.1f}'))}│")
    print(f"│ Recall:    {macro_rec:.1f}%{' '*(19-len(f'{macro_rec:.1f}'))}│")
    print(f"│ F1-score:  {macro_f1:.1f}%{' '*(19-len(f'{macro_f1:.1f}'))}│")
    print(f"│ AUC:       {macro_auc_val:.1f}%{' '*(19-len(f'{macro_auc_val:.1f}'))}│")
    print("└─────────────────────────────────┘")

    # Save summary to file
    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.txt")
    with open(summary_path, 'w') as f:
        f.write("DermaViT — Summary Metrics\n")
        f.write("=" * 40 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v:.2f}%\n")
    print(f"  ✓ Summary saved to {summary_path}")

    return summary


if __name__ == '__main__':
    evaluate()
