"""
DermaViT Training Loop
Complete training with AMP, differential LRs, cosine annealing,
early stopping, and checkpoint saving.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import (
    BATCH_SIZE, NUM_EPOCHS, WEIGHT_DECAY,
    GROUNDTRUTH_CSV, IMAGE_DIR, SEED,
    OUTPUT_DIR, RESULTS_DIR, BEST_MODEL_PATH,
    EARLY_STOPPING_PATIENCE, NUM_CLASSES
)
from utils import (
    set_seed, save_checkpoint, plot_training_curves
)
from dataset import get_dataloaders
from model import DermaViT


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Train for one epoch with mixed precision.
    
    Returns:
        avg_loss, accuracy (%)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """
    Validate model on val/test set.
    
    Returns:
        avg_loss, accuracy (%), macro_f1
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, macro_f1


def train(config=None):
    """
    Full training pipeline.
    
    Returns:
        model, test_loader (for downstream evaluation)
    """
    print("=" * 70)
    print("DermaViT — Training Pipeline")
    print("=" * 70)

    # Setup
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Data
    print("\n[1/4] Loading data...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        GROUNDTRUTH_CSV, IMAGE_DIR, BATCH_SIZE, SEED
    )

    # Model
    print("\n[2/4] Building model...")
    model = DermaViT().to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW(
        model.get_param_groups(),
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    print(f"\n[3/4] Training for {NUM_EPOCHS} epochs...")
    best_val_f1 = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )

        # Step scheduler
        scheduler.step()

        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - start_time

        # Print epoch summary
        print(
            f"  Epoch [{epoch}/{NUM_EPOCHS}] "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.1f}% | "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc:.1f}% ValF1={val_f1:.4f} | "
            f"{elapsed:.0f}s"
        )

        # Checkpoint saving (best val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_f1, BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  ⚠ Early stopping at epoch {epoch} "
                      f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Plot training curves
    print("\n[4/4] Saving training curves...")
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(RESULTS_DIR, "training_curves.png")
    )

    print(f"\n  ✓ Best Val F1: {best_val_f1:.4f}")
    print("  ✓ Training complete!")

    return model, test_loader


if __name__ == '__main__':
    train()
