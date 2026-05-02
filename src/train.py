# ═══ DermaViT v2.1 ═══
# Modified: Fix 2b (label smoothing), Fix 4a (ReduceLROnPlateau), Fix 4b (MixUp)
# All changes marked with # CHANGED
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
    EARLY_STOPPING_PATIENCE, NUM_CLASSES,
    ABLATION_MODE, SE_REDUCTION_RATIOS, DROPOUT_RATES,  # CHANGED
    LR_CNN_OPTIONS, LR_SWIN_OPTIONS, ABLATION_EPOCHS, ABLATION_PATIENCE,  # CHANGED
    LABEL_SMOOTHING, MIXUP_ALPHA, MIXUP_PROB  # CHANGED: Import new config values (Fix 2b, 4b)
)
from utils import (
    set_seed, save_checkpoint, plot_training_curves
)
from dataset import get_dataloaders
from model import DermaViT


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):  # CHANGED
    """  # CHANGED
    Train for one epoch with mixed precision and MixUp augmentation.  # CHANGED
    
    Returns:  # CHANGED
        avg_loss, accuracy (%)  # CHANGED
    """  # CHANGED
    model.train()  # CHANGED
    running_loss = 0.0  # CHANGED
    correct = 0  # CHANGED
    total = 0  # CHANGED

    pbar = tqdm(loader, desc="  Train", leave=False)  # CHANGED
    for batch in pbar:  # CHANGED
        images, metadata, labels = batch  # CHANGED: unpack metadata
        images = images.to(device, non_blocking=True)  # CHANGED
        metadata = metadata.to(device, non_blocking=True)  # CHANGED
        labels = labels.to(device, non_blocking=True)  # CHANGED

        optimizer.zero_grad()  # CHANGED
        
        # CHANGED: MixUp augmentation (Fix 4b)
        use_mixup = np.random.rand() < MIXUP_PROB  # CHANGED: Apply with probability
        if use_mixup:  # CHANGED
            # CHANGED: Sample lambda from Beta distribution
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)  # CHANGED
            batch_size = images.size(0)  # CHANGED
            index = torch.randperm(batch_size).to(device)  # CHANGED: Shuffle indices
            
            # CHANGED: Mix images and metadata
            mixed_images = lam * images + (1 - lam) * images[index]  # CHANGED
            mixed_metadata = lam * metadata + (1 - lam) * metadata[index]  # CHANGED
            labels_a, labels_b = labels, labels[index]  # CHANGED
            
            with autocast():  # CHANGED
                logits = model(mixed_images, mixed_metadata)  # CHANGED
                # CHANGED: MixUp loss
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)  # CHANGED
        else:  # CHANGED: Standard training
            with autocast():  # CHANGED
                logits = model(images, metadata)  # CHANGED: pass metadata
                loss = criterion(logits, labels)  # CHANGED

        scaler.scale(loss).backward()  # CHANGED
        scaler.step(optimizer)  # CHANGED
        scaler.update()  # CHANGED

        running_loss += loss.item() * images.size(0)  # CHANGED
        _, preds = logits.max(1)  # CHANGED
        correct += preds.eq(labels).sum().item()  # CHANGED: Count correct (use original labels)
        total += labels.size(0)  # CHANGED

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")  # CHANGED

    avg_loss = running_loss / total  # CHANGED
    accuracy = 100.0 * correct / total  # CHANGED
    return avg_loss, accuracy  # CHANGED


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
        for batch in tqdm(loader, desc="  Val  ", leave=False):  # CHANGED
            images, metadata, labels = batch  # CHANGED: unpack metadata
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)  # CHANGED
            labels = labels.to(device, non_blocking=True)

            with autocast():
                logits = model(images, metadata)  # CHANGED: pass metadata
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


def run_ablation_study(train_loader, val_loader, class_weights, device):  # CHANGED: new function
    """
    Run hyperparameter ablation study with grid search.
    
    Returns:
        best_config dict
    """
    import itertools
    import csv
    
    print("\n" + "=" * 70)
    print("DermaViT — Ablation Study")
    print("=" * 70)
    
    # Grid search combinations
    configs = list(itertools.product(
        SE_REDUCTION_RATIOS, DROPOUT_RATES, LR_CNN_OPTIONS, LR_SWIN_OPTIONS
    ))
    
    print(f"\n  Total configurations to test: {len(configs)}")
    print(f"  Epochs per config: {ABLATION_EPOCHS}")
    print(f"  Early stopping patience: {ABLATION_PATIENCE}\n")
    
    results = []
    best_val_f1 = 0.0
    best_config = None
    
    for idx, (se_ratio, dropout, lr_cnn, lr_swin) in enumerate(configs, 1):
        print(f"\n[Config {idx}/{len(configs)}] SE={se_ratio}, Dropout={dropout}, LR_CNN={lr_cnn}, LR_SWIN={lr_swin}")
        
        # Build model with current config
        model = DermaViT(dropout=dropout, se_reduction=se_ratio).to(device)
        
        # Optimizer with differential LRs
        optimizer = torch.optim.AdamW([
            {'params': model.efficientnet.parameters(), 'lr': lr_cnn},
            {'params': model.swin.parameters(), 'lr': lr_swin},
            {'params': list(model.fusion.parameters()) + list(model.classifier.parameters()), 'lr': lr_cnn}
        ], weight_decay=WEIGHT_DECAY)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        scaler = GradScaler()
        
        # Train for ablation epochs
        best_config_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(1, ABLATION_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device
            )
            val_loss, val_acc, val_f1 = validate(
                model, val_loader, criterion, device
            )
            scheduler.step()
            
            if val_f1 > best_config_val_f1:
                best_config_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= ABLATION_PATIENCE:
                    print(f"    Early stop at epoch {epoch}")
                    break
        
        # Record results
        results.append({
            'se_ratio': se_ratio,
            'dropout': dropout,
            'lr_cnn': lr_cnn,
            'lr_swin': lr_swin,
            'val_f1': best_config_val_f1,
            'val_acc': val_acc
        })
        
        print(f"    Best Val F1: {best_config_val_f1:.4f}, Val Acc: {val_acc:.1f}%")
        
        if best_config_val_f1 > best_val_f1:
            best_val_f1 = best_config_val_f1
            best_config = results[-1].copy()
        
        # Clear GPU cache
        del model, optimizer, scheduler, criterion, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results to CSV
    ablation_csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    with open(ablation_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['se_ratio', 'dropout', 'lr_cnn', 'lr_swin', 'val_f1', 'val_acc'])
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "=" * 70)
    print("Ablation Study Complete")
    print("=" * 70)
    print(f"\nBest Configuration:")
    print(f"  SE Reduction Ratio: {best_config['se_ratio']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  LR CNN: {best_config['lr_cnn']}")
    print(f"  LR Swin: {best_config['lr_swin']}")
    print(f"  Val F1: {best_config['val_f1']:.4f}")
    print(f"  Val Acc: {best_config['val_acc']:.1f}%")
    print(f"\nResults saved to: {ablation_csv_path}")
    
    return best_config


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
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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

    # Scheduler (cosine + plateau)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    # CHANGED: Add ReduceLROnPlateau as secondary scheduler (Fix 4a)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # CHANGED
        optimizer, mode='max', factor=0.5, patience=5  # CHANGED
    )  # CHANGED

    # CHANGED: Loss with class weights and label smoothing (Fix 2b)
    criterion = nn.CrossEntropyLoss(  # CHANGED
        weight=class_weights.to(device),  # CHANGED
        label_smoothing=LABEL_SMOOTHING  # CHANGED: Add label smoothing
    )  # CHANGED

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

        # Step schedulers
        scheduler.step()  # CHANGED: Cosine scheduler
        plateau_scheduler.step(val_f1)  # CHANGED: Plateau scheduler (Fix 4a)

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
