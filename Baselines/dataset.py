"""
DermaViT Dataset
HAM10000 dataset loading with CLAHE preprocessing, albumentations transforms,
and stratified train/val/test split.
"""
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    ONEHOT_COLUMNS, IMAGE_DIR, GROUNDTRUTH_CSV,
    NUM_CLASSES
)

# Optional CPU setting fallback
try:
    from config import DEBUG_SUBSET
except ImportError:
    DEBUG_SUBSET = False

from utils import get_class_weights


class HAM10000Dataset(Dataset):
    """
    HAM10000 Skin Lesion Dataset.
    
    Reads GroundTruth.csv (one-hot encoded), converts to integer labels,
    applies CLAHE per-channel, and then albumentations transforms.
    """

    def __init__(self, image_ids, labels, image_dir, transform=None):
        """
        Args:
            image_ids: list of image IDs (e.g., 'ISIC_0024306')
            labels: list of integer labels
            image_dir: path to image directory
            transform: albumentations transform pipeline
        """
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]

        # Load image
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to target size
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Apply CLAHE per-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for c in range(3):
            image[:, :, c] = clahe.apply(image[:, :, c])

        # Apply albumentations transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


def get_transforms(phase: str):
    """
    Get albumentations transform pipeline.
    
    Args:
        phase: 'train', 'val', or 'test'
    
    Returns:
        albumentations.Compose
    """
    if phase == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5
            ),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5
            ),
            A.CoarseDropout(
                max_holes=8, max_height=16, max_width=16,
                min_holes=1, min_height=8, min_width=8,
                fill_value=0, p=0.2
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:  # val or test
        return A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


def get_dataloaders(groundtruth_csv, image_dir, batch_size, seed):
    """
    Create stratified train/val/test dataloaders.
    
    Split: 80% train, 10% val, 10% test (stratified by class).
    
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    # Load metadata
    df = pd.read_csv(groundtruth_csv)

    if DEBUG_SUBSET:
        print("\n  [CPU DEBUG MODE] Subsetting dataset to 140 images to save memory/time.")
        df = df.head(140)

    # Convert one-hot to integer labels
    image_ids = df['image'].values.tolist()
    onehot_values = df[ONEHOT_COLUMNS].values
    labels = np.argmax(onehot_values, axis=1).tolist()

    stratify_target = None if DEBUG_SUBSET else labels
    # Stratified split: 80% train, 20% temp
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        image_ids, labels,
        test_size=0.2,
        stratify=stratify_target,
        random_state=seed
    )

    # Stratified split: 50% val, 50% test from temp (→ 10% each overall)
    stratify_temp = None if DEBUG_SUBSET else temp_labels
    val_ids, test_ids, val_labels, test_labels = train_test_split(
        temp_ids, temp_labels,
        test_size=0.5,
        stratify=stratify_temp,
        random_state=seed
    )

    print(f"  Dataset splits: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Class distribution in training set
    train_label_arr = np.array(train_labels)
    for i, name in enumerate(ONEHOT_COLUMNS):
        count = np.sum(train_label_arr == i)
        print(f"    {name}: {count} ({100 * count / len(train_labels):.1f}%)")

    # Compute class weights from training set
    class_weights = get_class_weights(train_labels, NUM_CLASSES)
    print(f"  Class weights: {class_weights.tolist()}")

    # Create datasets
    train_dataset = HAM10000Dataset(
        train_ids, train_labels, image_dir, transform=get_transforms('train')
    )
    val_dataset = HAM10000Dataset(
        val_ids, val_labels, image_dir, transform=get_transforms('val')
    )
    test_dataset = HAM10000Dataset(
        test_ids, test_labels, image_dir, transform=get_transforms('test')
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_weights
