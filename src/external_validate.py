# ═══ DermaViT v2.1 ═══
# Modified: Fix 5 - External validation support
# All changes marked with # CHANGED
"""
DermaViT External Validation
Validate trained model on external datasets (e.g., ISIC 2019).
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, CLASS_NAMES
from model import DermaViT


# CHANGED: External dataset class
class ExternalDataset(Dataset):  # CHANGED
    """  # CHANGED
    External dataset loader with same preprocessing as training.  # CHANGED
    Supports ISIC 2019 and other external datasets.  # CHANGED
    """  # CHANGED
    
    def __init__(self, image_ids, labels, image_dir, metadata_df=None):  # CHANGED
        """  # CHANGED
        Args:  # CHANGED
            image_ids: list of image IDs  # CHANGED
            labels: list of integer labels (mapped to HAM10000 classes)  # CHANGED
            image_dir: path to image directory  # CHANGED
            metadata_df: DataFrame with age, sex, localization (optional)  # CHANGED
        """  # CHANGED
        self.image_ids = image_ids  # CHANGED
        self.labels = labels  # CHANGED
        self.image_dir = image_dir  # CHANGED
        self.metadata_df = metadata_df  # CHANGED
        
        # CHANGED: Preprocessing pipeline (CLAHE + normalize, NO augmentation)
        self.transform = A.Compose([  # CHANGED
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # CHANGED
            ToTensorV2(),  # CHANGED
        ])  # CHANGED
        
        # CHANGED: Metadata defaults
        self.age_mean = 0.0  # CHANGED
        self.age_std = 1.0  # CHANGED
        self.age_median = 0.0  # CHANGED
        self.sex_categories = ['male', 'female', 'unknown']  # CHANGED
        self.localization_categories = [  # CHANGED
            'abdomen', 'back', 'chest', 'ear', 'face', 'foot', 'genital',  # CHANGED
            'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'upper extremity', 'acral', 'unknown'  # CHANGED
        ]  # CHANGED
    
    def set_metadata_stats(self, age_mean, age_std, age_median, sex_categories, localization_categories):  # CHANGED
        """Set metadata normalization statistics."""  # CHANGED
        self.age_mean = age_mean  # CHANGED
        self.age_std = age_std  # CHANGED
        self.age_median = age_median  # CHANGED
        self.sex_categories = sex_categories  # CHANGED
        self.localization_categories = localization_categories  # CHANGED
    
    def _get_metadata_vector(self, image_id):  # CHANGED
        """Extract and encode metadata (same as training)."""  # CHANGED
        age_norm = 0.0  # CHANGED
        sex_onehot = [0.0, 0.0, 1.0]  # CHANGED: default unknown
        loc_onehot = [0.0] * len(self.localization_categories)  # CHANGED
        
        if self.metadata_df is not None and image_id in self.metadata_df.index:  # CHANGED
            row = self.metadata_df.loc[image_id]  # CHANGED
            
            if 'age' in row and pd.notna(row['age']):  # CHANGED
                age_norm = (row['age'] - self.age_mean) / self.age_std  # CHANGED
            else:  # CHANGED
                age_norm = (self.age_median - self.age_mean) / self.age_std  # CHANGED
            
            if 'sex' in row and pd.notna(row['sex']):  # CHANGED
                sex_val = str(row['sex']).lower()  # CHANGED
                if sex_val in self.sex_categories:  # CHANGED
                    sex_idx = self.sex_categories.index(sex_val)  # CHANGED
                    sex_onehot = [0.0] * len(self.sex_categories)  # CHANGED
                    sex_onehot[sex_idx] = 1.0  # CHANGED
            
            if 'localization' in row and pd.notna(row['localization']):  # CHANGED
                loc_val = str(row['localization']).lower()  # CHANGED
                if loc_val in self.localization_categories:  # CHANGED
                    loc_idx = self.localization_categories.index(loc_val)  # CHANGED
                    loc_onehot = [0.0] * len(self.localization_categories)  # CHANGED
                    loc_onehot[loc_idx] = 1.0  # CHANGED
        
        metadata_vec = [age_norm] + sex_onehot + loc_onehot  # CHANGED
        return torch.FloatTensor(metadata_vec)  # CHANGED
    
    def __len__(self):  # CHANGED
        return len(self.image_ids)  # CHANGED
    
    def __getitem__(self, idx):  # CHANGED
        image_id = self.image_ids[idx]  # CHANGED
        label = self.labels[idx]  # CHANGED
        
        # CHANGED: Load image (try multiple extensions)
        img_path = None  # CHANGED
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:  # CHANGED
            candidate = os.path.join(self.image_dir, f"{image_id}{ext}")  # CHANGED
            if os.path.exists(candidate):  # CHANGED
                img_path = candidate  # CHANGED
                break  # CHANGED
        
        if img_path is None:  # CHANGED
            raise FileNotFoundError(f"Image not found: {image_id}")  # CHANGED
        
        image = cv2.imread(img_path)  # CHANGED
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # CHANGED
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # CHANGED
        
        # CHANGED: Apply CLAHE per-channel (same as training)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CHANGED
        for c in range(3):  # CHANGED
            image[:, :, c] = clahe.apply(image[:, :, c])  # CHANGED
        
        # CHANGED: Apply transform (normalize only, no augmentation)
        augmented = self.transform(image=image)  # CHANGED
        image = augmented['image']  # CHANGED
        
        metadata_vec = self._get_metadata_vector(image_id)  # CHANGED
        
        return image, metadata_vec, label  # CHANGED


# CHANGED: Main external validation function
def validate_external(  # CHANGED
    model_path: str,  # CHANGED
    image_dir: str,  # CHANGED
    metadata_csv: str,  # CHANGED
    dataset_name: str = "ISIC2019",  # CHANGED
    output_dir: str = "outputs/external/",  # CHANGED
    device: str = "cuda"  # CHANGED
) -> dict:  # CHANGED
    """  # CHANGED
    Validate trained DermaViT model on external dataset.  # CHANGED
    
    Args:  # CHANGED
        model_path: Path to trained model checkpoint  # CHANGED
        image_dir: Path to external dataset images  # CHANGED
        metadata_csv: Path to external metadata CSV  # CHANGED
        dataset_name: Name of external dataset (e.g., "ISIC2019")  # CHANGED
        output_dir: Directory to save results  # CHANGED
        device: Device to run inference on  # CHANGED
    
    Returns:  # CHANGED
        dict: Summary metrics  # CHANGED
    """  # CHANGED
    print("\n" + "=" * 70)  # CHANGED
    print(f"DermaViT — External Validation on {dataset_name}")  # CHANGED
    print("=" * 70)  # CHANGED
    
    os.makedirs(output_dir, exist_ok=True)  # CHANGED
    device = torch.device(device if torch.cuda.is_available() else 'cpu')  # CHANGED
    print(f"\n  Device: {device}")  # CHANGED
    
    # CHANGED: Load model
    print(f"\n  Loading model from {model_path}...")  # CHANGED
    model = DermaViT().to(device)  # CHANGED
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # CHANGED
    model.load_state_dict(checkpoint['model_state_dict'])  # CHANGED
    model.eval()  # CHANGED
    print("  ✓ Model loaded")  # CHANGED
    
    # CHANGED: Load external metadata
    print(f"\n  Loading metadata from {metadata_csv}...")  # CHANGED
    df = pd.read_csv(metadata_csv)  # CHANGED
    print(f"  ✓ Loaded {len(df)} samples")  # CHANGED
    
    # CHANGED: Class mapping for ISIC 2019 → HAM10000
    # ISIC 2019 has 8 classes, HAM10000 has 7. Map and exclude SCC.
    class_mapping = {  # CHANGED
        'MEL': 0, 'NV': 1, 'BCC': 2, 'AK': 3, 'AKIEC': 3,  # CHANGED: AK maps to AKIEC
        'BKL': 4, 'DF': 5, 'VASC': 6, 'SCC': -1  # CHANGED: SCC excluded
    }  # CHANGED
    
    # CHANGED: Filter and map labels
    valid_samples = []  # CHANGED
    for _, row in df.iterrows():  # CHANGED
        if 'diagnosis' in row or 'dx' in row:  # CHANGED
            dx = str(row.get('diagnosis', row.get('dx', ''))).upper()  # CHANGED
            if dx in class_mapping and class_mapping[dx] != -1:  # CHANGED
                valid_samples.append((row['image'], class_mapping[dx]))  # CHANGED
    
    if len(valid_samples) == 0:  # CHANGED
        print("  ⚠ No valid samples found after class mapping")  # CHANGED
        return {}  # CHANGED
    
    image_ids, labels = zip(*valid_samples)  # CHANGED
    print(f"  ✓ {len(valid_samples)} valid samples (excluded SCC)")  # CHANGED
    
    # CHANGED: Prepare metadata DataFrame
    metadata_df = None  # CHANGED
    if 'age' in df.columns or 'sex' in df.columns or 'localization' in df.columns:  # CHANGED
        meta_cols = [c for c in ['age', 'sex', 'localization'] if c in df.columns]  # CHANGED
        if meta_cols and 'image' in df.columns:  # CHANGED
            metadata_df = df.set_index('image')[meta_cols]  # CHANGED
    
    # CHANGED: Create dataset and dataloader
    dataset = ExternalDataset(list(image_ids), list(labels), image_dir, metadata_df)  # CHANGED
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)  # CHANGED
    
    # CHANGED: Run inference
    print("\n  Running inference...")  # CHANGED
    all_preds = []  # CHANGED
    all_labels = []  # CHANGED
    all_probs = []  # CHANGED
    
    with torch.no_grad():  # CHANGED
        for batch in tqdm(loader, desc="  Inference"):  # CHANGED
            images, metadata, labels_batch = batch  # CHANGED
            images = images.to(device)  # CHANGED
            metadata = metadata.to(device)  # CHANGED
            
            with autocast():  # CHANGED
                logits = model(images, metadata)  # CHANGED
            
            probs = F.softmax(logits, dim=1)  # CHANGED
            preds = logits.argmax(dim=1)  # CHANGED
            
            all_preds.extend(preds.cpu().numpy())  # CHANGED
            all_labels.extend(labels_batch.numpy())  # CHANGED
            all_probs.extend(probs.cpu().numpy())  # CHANGED
    
    all_preds = np.array(all_preds)  # CHANGED
    all_labels = np.array(all_labels)  # CHANGED
    all_probs = np.array(all_probs)  # CHANGED
    
    # CHANGED: Compute metrics
    print("\n  Computing metrics...")  # CHANGED
    accuracy = accuracy_score(all_labels, all_preds) * 100  # CHANGED
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100  # CHANGED
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100  # CHANGED
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100  # CHANGED
    
    # CHANGED: Compute AUC
    auc_scores = []  # CHANGED
    for i in range(NUM_CLASSES):  # CHANGED
        y_true_binary = (all_labels == i).astype(int)  # CHANGED
        if y_true_binary.sum() > 0:  # CHANGED: Only if class exists
            fpr, tpr, _ = roc_curve(y_true_binary, all_probs[:, i])  # CHANGED
            auc_scores.append(auc(fpr, tpr))  # CHANGED
    macro_auc = np.mean(auc_scores) * 100 if auc_scores else 0.0  # CHANGED
    
    # CHANGED: Top-3 accuracy
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:]  # CHANGED
    top3_correct = [all_labels[i] in top3_preds[i] for i in range(len(all_labels))]  # CHANGED
    top3_acc = np.mean(top3_correct) * 100  # CHANGED
    
    # CHANGED: Save results
    results_path = os.path.join(output_dir, f"{dataset_name}_results.txt")  # CHANGED
    with open(results_path, 'w') as f:  # CHANGED
        f.write(f"DermaViT — External Validation on {dataset_name}\n")  # CHANGED
        f.write("=" * 70 + "\n\n")  # CHANGED
        f.write(f"Accuracy:       {accuracy:.2f}%\n")  # CHANGED
        f.write(f"Precision:      {precision:.2f}%\n")  # CHANGED
        f.write(f"Recall:         {recall:.2f}%\n")  # CHANGED
        f.write(f"F1-Score:       {f1:.2f}%\n")  # CHANGED
        f.write(f"AUC (Macro):    {macro_auc:.2f}%\n")  # CHANGED
        f.write(f"Top-3 Accuracy: {top3_acc:.2f}%\n")  # CHANGED
    print(f"  ✓ Results saved to {results_path}")  # CHANGED
    
    # CHANGED: Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)  # CHANGED
    plt.figure(figsize=(10, 8))  # CHANGED
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)  # CHANGED
    plt.title(f'Confusion Matrix — {dataset_name}', fontsize=14, fontweight='bold')  # CHANGED
    plt.ylabel('True Label', fontsize=12)  # CHANGED
    plt.xlabel('Predicted Label', fontsize=12)  # CHANGED
    cm_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")  # CHANGED
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')  # CHANGED
    plt.close()  # CHANGED
    print(f"  ✓ Confusion matrix saved to {cm_path}")  # CHANGED
    
    # CHANGED: Print summary
    print("\n" + "=" * 70)  # CHANGED
    print(f"External Validation Complete — {dataset_name}")  # CHANGED
    print("=" * 70)  # CHANGED
    print(f"  Accuracy:       {accuracy:.2f}%")  # CHANGED
    print(f"  Precision:      {precision:.2f}%")  # CHANGED
    print(f"  Recall:         {recall:.2f}%")  # CHANGED
    print(f"  F1-Score:       {f1:.2f}%")  # CHANGED
    print(f"  AUC (Macro):    {macro_auc:.2f}%")  # CHANGED
    print(f"  Top-3 Accuracy: {top3_acc:.2f}%")  # CHANGED
    print("=" * 70)  # CHANGED
    
    return {  # CHANGED
        'accuracy': accuracy,  # CHANGED
        'precision': precision,  # CHANGED
        'recall': recall,  # CHANGED
        'f1_score': f1,  # CHANGED
        'auc': macro_auc,  # CHANGED
        'top3_accuracy': top3_acc  # CHANGED
    }  # CHANGED
