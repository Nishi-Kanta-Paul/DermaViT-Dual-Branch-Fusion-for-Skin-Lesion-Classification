"""
Baseline Training Script for ResNet-50
"""
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import timm
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import (
    BATCH_SIZE, NUM_EPOCHS, WEIGHT_DECAY, LR_BASE,
    GROUNDTRUTH_CSV, IMAGE_DIR, SEED,
    OUTPUT_DIR, EARLY_STOPPING_PATIENCE, NUM_CLASSES
)
from utils import set_seed, save_checkpoint, plot_training_curves
from dataset import get_dataloaders

# Model specific setup
MODEL_NAME = "EfficientNet-B2"
MODEL_DIR = os.path.join(OUTPUT_DIR, MODEL_NAME)
RESULTS_DIR = os.path.join(MODEL_DIR, "results")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")


def build_model():
    model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=NUM_CLASSES)
    return model


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:  
        images, metadata, labels = batch  
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
        correct += logits.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")
    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Val  ", leave=False):  
            images, metadata, labels = batch  
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            preds = logits.max(1)[1]
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, 100.0 * correct / total, f1_score(all_labels, all_preds, average='macro')


def get_evaluate_script_logic(model, test_loader, device):
    """Inline evaluation to match DermaViT evaluate.py output requirements."""
    import numpy as np
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Test"):  # CHANGED
            images, metadata, labels = batch  # CHANGED
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
            all_preds.extend(logits.max(1)[1].cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    # Summary
    overall_acc = accuracy_score(all_labels, all_preds) * 100
    macro_prec = precision_score(all_labels, all_preds, average='macro') * 100
    macro_rec = recall_score(all_labels, all_preds, average='macro') * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    macro_auc_list = []
    for i in range(NUM_CLASSES):
        y_true_binary = (all_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, all_probs[:, i])
        macro_auc_list.append(auc(fpr, tpr))
    macro_auc_val = np.mean(macro_auc_list) * 100

    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.txt")
    with open(summary_path, 'w') as f:
        f.write(f"{MODEL_NAME} — Summary Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"accuracy: {overall_acc:.2f}%\n")
        f.write(f"precision: {macro_prec:.2f}%\n")
        f.write(f"recall: {macro_rec:.2f}%\n")
        f.write(f"f1_score: {macro_f1:.2f}%\n")
        f.write(f"auc: {macro_auc_val:.2f}%\n")
    print(f"✓ Saved test results to {summary_path}")


def main():
    print(f"=== Baseline: {MODEL_NAME} ===")
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(GROUNDTRUTH_CSV, IMAGE_DIR, BATCH_SIZE, SEED)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = GradScaler()

    best_val_f1, patience_counter = 0.0, 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_f1, BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(RESULTS_DIR, "training_curves.png"))
    
    # Load best model and evaluate
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])    
    print("\nRunning final test evaluation...")
    get_evaluate_script_logic(model, test_loader, device)

if __name__ == '__main__':
    main()
