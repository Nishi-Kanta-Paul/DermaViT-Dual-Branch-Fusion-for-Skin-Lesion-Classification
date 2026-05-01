# ═══ DermaViT v2.1 ═══
# Modified: Fix 3 (multi-seed training), Fix 5 (external validation)
# All changes marked with # CHANGED
"""
DermaViT — Main Entry Point
Runs the complete pipeline: seed → data → model → train → evaluate → explain.
"""
import os
import sys
import time
import numpy as np  # CHANGED: Import for multi-seed aggregation
import torch

from config import (
    SEED, BATCH_SIZE, NUM_EPOCHS, NUM_CLASSES,
    GROUNDTRUTH_CSV, IMAGE_DIR, OUTPUT_DIR,
    RESULTS_DIR, SALIENCY_DIR, BEST_MODEL_PATH,
    CLASS_NAMES, ABLATION_MODE, DATA_ROOT,
    MULTI_SEED_MODE, TRAINING_SEEDS,  # CHANGED: Import multi-seed config (Fix 3)
    EXTERNAL_VALIDATION, EXTERNAL_DATA_PATH, EXTERNAL_METADATA_PATH, EXTERNAL_DATASET_NAME  # CHANGED: Import external validation config (Fix 5)
)
from utils import set_seed, load_checkpoint
from dataset import get_dataloaders
from model import DermaViT
from train import train, run_ablation_study
from evaluate import evaluate
from explainability import generate_saliency_maps
from external_validate import validate_external  # CHANGED: Import external validation (Fix 5)


def run_single_seed_pipeline(seed=SEED):  # CHANGED: Renamed and parameterized (Fix 3)
    """Run the complete DermaViT pipeline for a single seed."""  # CHANGED
    start_time = time.time()  # CHANGED
    
    # CHANGED: Use provided seed
    set_seed(seed)  # CHANGED

    print("=" * 50)
    print("      DermaViT — Skin Lesion Classification")
    print("  Dual-Branch Fusion: EfficientNet-B0 + Swin-T")
    print("=" * 50)

    # ── Step 1: Set seed for reproducibility ──
    print("\n[Step 1/7] Setting random seed...")
    # CHANGED: Already set at function start, just print
    print(f"  ✓ Seed set to {seed}")  # CHANGED

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

    # ── Step 5: Check ablation mode or train ──
    if ABLATION_MODE:
        print("\n[Step 5/7] Running ablation study...")
        print("  Loading data for ablation...")
        train_loader, val_loader, test_loader, class_weights = get_dataloaders(
            GROUNDTRUTH_CSV, IMAGE_DIR, BATCH_SIZE, SEED
        )
        best_config = run_ablation_study(train_loader, val_loader, class_weights, device)
        print("\n" + "=" * 50)
        print("Ablation study complete. Exiting.")
        print("=" * 50)
        return
    
    print("\n[Step 5/7] Starting training...")
    model, test_loader = train()

    # Load best checkpoint for evaluation
    print("\n  Loading best checkpoint for evaluation...")
    model, _, _, best_f1 = load_checkpoint(model, None, BEST_MODEL_PATH)
    model = model.to(device)

    # ── Step 6: Evaluate on test set ──
    print("\n[Step 6/7] Evaluating on test set...")
    masks_dir = os.path.join(DATA_ROOT, "masks")
    summary = evaluate(model=model, test_loader=test_loader, device=device, masks_dir=masks_dir)

    # ── Step 7: Generate explainability maps ──
    print("\n[Step 7/7] Generating saliency maps...")
    generate_saliency_maps(model=model, test_loader=test_loader, device=device, n_samples=20)
    
    # CHANGED: External validation (Fix 5)
    if EXTERNAL_VALIDATION and EXTERNAL_DATA_PATH and EXTERNAL_METADATA_PATH:  # CHANGED
        print("\n[Extra] Running external validation...")  # CHANGED
        try:  # CHANGED
            ext_summary = validate_external(  # CHANGED
                model_path=BEST_MODEL_PATH,  # CHANGED
                image_dir=EXTERNAL_DATA_PATH,  # CHANGED
                metadata_csv=EXTERNAL_METADATA_PATH,  # CHANGED
                dataset_name=EXTERNAL_DATASET_NAME,  # CHANGED
                output_dir=os.path.join(OUTPUT_DIR, "external"),  # CHANGED
                device=str(device)  # CHANGED
            )  # CHANGED
        except Exception as e:  # CHANGED
            print(f"  ⚠ External validation failed: {e}")  # CHANGED

    # ── Final Summary ──
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print("\n")
    print("=" * 50)
    print("       DermaViT — Final Results")
    print("-" * 50)
    acc_s = f"{summary['accuracy']:.1f}"
    top3_s = f"{summary.get('top3_accuracy', 0.0):.1f}"
    prec_s = f"{summary['precision']:.1f}"
    rec_s = f"{summary['recall']:.1f}"
    f1_s = f"{summary['f1_score']:.1f}"
    auc_s = f"{summary['auc']:.1f}"

    print(f"  Accuracy (Test) : {acc_s:>5}%")
    print(f"  Top-3 Accuracy  : {top3_s:>5}%")
    print(f"  Precision       : {prec_s:>5}%")
    print(f"  Recall          : {rec_s:>5}%")
    print(f"  F1-Score        : {f1_s:>5}%")
    print(f"  AUC (Macro)     : {auc_s:>5}%")
    if 'mean_dice' in summary:
        dice_s = f"{summary['mean_dice']:.4f}"
        print(f"  Mean Dice (XAI) : {dice_s:>5}")
    print("=" * 50)

    print("\n✓ Pipeline complete! All outputs saved to:")
    print(f"  • Best model:     {BEST_MODEL_PATH}")
    print(f"  • Results:        {RESULTS_DIR}/")
    print(f"  • Saliency maps:  {SALIENCY_DIR}/")
    
    return summary  # CHANGED: Return summary for multi-seed aggregation


# CHANGED: Multi-seed training wrapper (Fix 3)
def run_multi_seed_training():  # CHANGED
    """Run training across multiple seeds and aggregate results."""  # CHANGED
    print("\n" + "=" * 70)  # CHANGED
    print("DermaViT — Multi-Seed Training Mode")  # CHANGED
    print("=" * 70)  # CHANGED
    print(f"\n  Training seeds: {TRAINING_SEEDS}")  # CHANGED
    print(f"  Total runs: {len(TRAINING_SEEDS)}\n")  # CHANGED
    
    all_summaries = []  # CHANGED
    
    for seed_idx, seed in enumerate(TRAINING_SEEDS, 1):  # CHANGED
        print("\n" + "=" * 70)  # CHANGED
        print(f"SEED {seed_idx}/{len(TRAINING_SEEDS)}: {seed}")  # CHANGED
        print("=" * 70)  # CHANGED
        
        # CHANGED: Update paths for this seed
        seed_output_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")  # CHANGED
        os.makedirs(seed_output_dir, exist_ok=True)  # CHANGED
        
        # CHANGED: Temporarily override global paths
        import config  # CHANGED
        original_output_dir = config.OUTPUT_DIR  # CHANGED
        original_results_dir = config.RESULTS_DIR  # CHANGED
        original_saliency_dir = config.SALIENCY_DIR  # CHANGED
        original_best_model_path = config.BEST_MODEL_PATH  # CHANGED
        
        config.OUTPUT_DIR = seed_output_dir  # CHANGED
        config.RESULTS_DIR = os.path.join(seed_output_dir, "results")  # CHANGED
        config.SALIENCY_DIR = os.path.join(seed_output_dir, "saliency_maps")  # CHANGED
        config.BEST_MODEL_PATH = os.path.join(seed_output_dir, f"best_model_seed{seed}.pth")  # CHANGED
        
        os.makedirs(config.RESULTS_DIR, exist_ok=True)  # CHANGED
        os.makedirs(config.SALIENCY_DIR, exist_ok=True)  # CHANGED
        
        # CHANGED: Run single seed pipeline
        try:  # CHANGED
            summary = run_single_seed_pipeline(seed=seed)  # CHANGED
            all_summaries.append(summary)  # CHANGED
        except Exception as e:  # CHANGED
            print(f"  ⚠ Seed {seed} failed: {e}")  # CHANGED
            continue  # CHANGED
        
        # CHANGED: Restore original paths
        config.OUTPUT_DIR = original_output_dir  # CHANGED
        config.RESULTS_DIR = original_results_dir  # CHANGED
        config.SALIENCY_DIR = original_saliency_dir  # CHANGED
        config.BEST_MODEL_PATH = original_best_model_path  # CHANGED
    
    # CHANGED: Aggregate results across seeds
    if len(all_summaries) == 0:  # CHANGED
        print("\n⚠ No successful runs to aggregate")  # CHANGED
        return  # CHANGED
    
    print("\n" + "=" * 70)  # CHANGED
    print("Multi-Seed Results Aggregation")  # CHANGED
    print("=" * 70)  # CHANGED
    
    # CHANGED: Compute mean ± std for each metric
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'top3_accuracy']  # CHANGED
    aggregated = {}  # CHANGED
    
    for metric in metrics:  # CHANGED
        values = [s.get(metric, 0.0) for s in all_summaries]  # CHANGED
        aggregated[metric] = {  # CHANGED
            'mean': np.mean(values),  # CHANGED
            'std': np.std(values)  # CHANGED
        }  # CHANGED
    
    # CHANGED: Save aggregated results
    summary_path = os.path.join(OUTPUT_DIR, "results", "multi_seed_summary.txt")  # CHANGED
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)  # CHANGED
    
    with open(summary_path, 'w') as f:  # CHANGED
        f.write("DermaViT — Multi-Seed Training Summary\n")  # CHANGED
        f.write("=" * 70 + "\n\n")  # CHANGED
        f.write(f"Seeds: {TRAINING_SEEDS}\n")  # CHANGED
        f.write(f"Successful runs: {len(all_summaries)}/{len(TRAINING_SEEDS)}\n\n")  # CHANGED
        f.write("Results (mean ± std):\n")  # CHANGED
        f.write("-" * 70 + "\n")  # CHANGED
        for metric in metrics:  # CHANGED
            mean = aggregated[metric]['mean']  # CHANGED
            std = aggregated[metric]['std']  # CHANGED
            f.write(f"{metric.replace('_', ' ').title():<20}: {mean:>5.1f} ± {std:>4.1f}%\n")  # CHANGED
    
    print(f"\n  ✓ Multi-seed summary saved to {summary_path}")  # CHANGED
    
    # CHANGED: Print summary
    print("\n  Results (mean ± std):")  # CHANGED
    for metric in metrics:  # CHANGED
        mean = aggregated[metric]['mean']  # CHANGED
        std = aggregated[metric]['std']  # CHANGED
        print(f"    {metric.replace('_', ' ').title():<20}: {mean:>5.1f} ± {std:>4.1f}%")  # CHANGED
    
    print("\n" + "=" * 70)  # CHANGED
    print("✓ Multi-seed training complete!")  # CHANGED
    print("=" * 70)  # CHANGED


def main():  # CHANGED
    """Main entry point with multi-seed support."""  # CHANGED
    if MULTI_SEED_MODE:  # CHANGED
        run_multi_seed_training()  # CHANGED
    else:  # CHANGED
        run_single_seed_pipeline()  # CHANGED


if __name__ == '__main__':
    main()
