"""
DermaViT — Main Entry Point
Runs the complete pipeline: seed → data → model → train → evaluate → explain.
"""
import os
import sys
import time
import torch

from config import (
    SEED, BATCH_SIZE, NUM_EPOCHS, NUM_CLASSES,
    GROUNDTRUTH_CSV, IMAGE_DIR, OUTPUT_DIR,
    RESULTS_DIR, SALIENCY_DIR, BEST_MODEL_PATH,
    CLASS_NAMES
)
from utils import set_seed, load_checkpoint
from dataset import get_dataloaders
from model import DermaViT
from train import train
from evaluate import evaluate
from explainability import generate_saliency_maps


def main():
    """Run the complete DermaViT pipeline."""
    start_time = time.time()

    print("╔══════════════════════════════════════════════════╗")
    print("║       DermaViT — Skin Lesion Classification      ║")
    print("║   Dual-Branch Fusion: EfficientNet-B0 + Swin-T   ║")
    print("╚══════════════════════════════════════════════════╝")

    # ── Step 1: Set seed for reproducibility ──
    print("\n[Step 1/7] Setting random seed...")
    set_seed(SEED)
    print(f"  ✓ Seed set to {SEED}")

    # ── Step 2: Setup device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Step 2/7] Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {mem_gb:.1f} GB")

    # ── Step 3: Create output directories ──
    print("\n[Step 3/7] Creating directories...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SALIENCY_DIR, exist_ok=True)
    print(f"  ✓ Output: {OUTPUT_DIR}/")
    print(f"  ✓ Results: {RESULTS_DIR}/")
    print(f"  ✓ Saliency: {SALIENCY_DIR}/")

    # ── Step 4: Build & inspect model ──
    print("\n[Step 4/7] Building DermaViT model...")
    model = DermaViT().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    # Free the model — train() will build its own
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 5: Train ──
    print("\n[Step 5/7] Starting training...")
    model, test_loader = train()

    # Load best checkpoint for evaluation
    print("\n  Loading best checkpoint for evaluation...")
    model, _, _, best_f1 = load_checkpoint(model, None, BEST_MODEL_PATH)
    model = model.to(device)

    # ── Step 6: Evaluate on test set ──
    print("\n[Step 6/7] Evaluating on test set...")
    summary = evaluate(model=model, test_loader=test_loader, device=device)

    # ── Step 7: Generate explainability maps ──
    print("\n[Step 7/7] Generating saliency maps...")
    generate_saliency_maps(model=model, test_loader=test_loader, device=device, n_samples=20)

    # ── Final Summary ──
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print("\n")
    print("╔══════════════════════════════════════════════════╗")
    print("║       DermaViT — Final Results                  ║")
    print("╠══════════════════════════════════════════════════╣")
    acc_s = f"{summary['accuracy']:.1f}"
    prec_s = f"{summary['precision']:.1f}"
    rec_s = f"{summary['recall']:.1f}"
    f1_s = f"{summary['f1_score']:.1f}"
    auc_s = f"{summary['auc']:.1f}"
    time_s = f"{hours}h {minutes}m"
    print(f"║  Accuracy:  {acc_s}%{' '*(36-len(acc_s))}║")
    print(f"║  Precision: {prec_s}%{' '*(36-len(prec_s))}║")
    print(f"║  Recall:    {rec_s}%{' '*(36-len(rec_s))}║")
    print(f"║  F1-score:  {f1_s}%{' '*(36-len(f1_s))}║")
    print(f"║  AUC:       {auc_s}%{' '*(36-len(auc_s))}║")
    print(f"║  Time:      {time_s}{' '*(38-len(time_s))}║")
    print("╚══════════════════════════════════════════════════╝")

    print("\n✓ Pipeline complete! All outputs saved to:")
    print(f"  • Best model:     {BEST_MODEL_PATH}")
    print(f"  • Results:        {RESULTS_DIR}/")
    print(f"  • Saliency maps:  {SALIENCY_DIR}/")


if __name__ == '__main__':
    main()
