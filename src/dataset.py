"""
DermaViT Dataset
Skin lesion dataset loading with CLAHE preprocessing, albumentations transforms,
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
import torch  # CHANGED: for metadata tensor creation

# Optional CPU setting fallback
try:
    from config import DEBUG_SUBSET
except ImportError:
    DEBUG_SUBSET = False

from utils import get_class_weights

# CHANGED: Metadata processing utilities
def _prepare_metadata_encoders(df):
    """
    Prepare metadata statistics and encoders from training data.
    Returns: age_mean, age_std, sex_categories, localization_categories
    """
    age_mean, age_std = 0.0, 1.0
    sex_categories = ['male', 'female', 'unknown']
    localization_categories = []
    
    if df is not None and 'age' in df.columns:
        valid_ages = df['age'].dropna()
        if len(valid_ages) > 0:
            age_mean = valid_ages.mean()
            age_std = valid_ages.std() if valid_ages.std() > 0 else 1.0
    
    if df is not None and 'localization' in df.columns:
        localization_categories = sorted(df['localization'].dropna().unique().tolist())
        if len(localization_categories) == 0:
            localization_categories = ['unknown']  # fallback
    else:
        # Default HAM10000 localization sites
        localization_categories = [
            'abdomen', 'back', 'chest', 'ear', 'face', 'foot', 'genital',
            'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'upper extremity', 'acral', 'unknown'
        ]
    
    return age_mean, age_std, sex_categories, localization_categories


class SkinLesionDataset(Dataset):
    """
    Skin lesion dataset.
    
    Reads metadata.csv (one-hot encoded), converts to integer labels,
    applies CLAHE per-channel, and then albumentations transforms.
    """

    def __init__(self, image_ids, labels, image_dir, transform=None, metadata_df=None):  # CHANGED: added metadata_df
        """
        Args:
            image_ids: list of image IDs (e.g., 'ISIC_0024306')
            labels: list of integer labels
            image_dir: path to image directory
            transform: albumentations transform pipeline
            metadata_df: DataFrame with age, sex, localization columns (optional)  # CHANGED
        """
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform
        self.metadata_df = metadata_df  # CHANGED
        
        # CHANGED: Initialize metadata statistics
        self.age_mean = 0.0
        self.age_std = 1.0
        self.age_median = 0.0
        self.sex_categories = ['male', 'female', 'unknown']
        self.localization_categories = [
            'abdomen', 'back', 'chest', 'ear', 'face', 'foot', 'genital',
            'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'upper extremity', 'acral', 'unknown'
        ]

    def set_metadata_stats(self, age_mean, age_std, age_median, sex_categories, localization_categories):  # CHANGED
        """Set metadata normalization statistics."""
        self.age_mean = age_mean
        self.age_std = age_std
        self.age_median = age_median
        self.sex_categories = sex_categories
        self.localization_categories = localization_categories

    def _get_metadata_vector(self, image_id):  # CHANGED: new method
        """
        Extract and encode metadata for a given image_id.
        Returns: torch.FloatTensor of shape (19,)
        """
        # Initialize with zeros
        age_norm = 0.0
        sex_onehot = [0.0, 0.0, 1.0]  # default: unknown
        loc_onehot = [0.0] * len(self.localization_categories)
        
        if self.metadata_df is not None and image_id in self.metadata_df.index:
            row = self.metadata_df.loc[image_id]
            
            # Age normalization
            if 'age' in row and pd.notna(row['age']):
                age_norm = (row['age'] - self.age_mean) / self.age_std
            else:
                # Missing age: use median
                age_norm = (self.age_median - self.age_mean) / self.age_std
            
            # Sex one-hot encoding
            if 'sex' in row and pd.notna(row['sex']):
                sex_val = str(row['sex']).lower()
                if sex_val in self.sex_categories:
                    sex_idx = self.sex_categories.index(sex_val)
                    sex_onehot = [0.0] * len(self.sex_categories)
                    sex_onehot[sex_idx] = 1.0
            
            # Localization one-hot encoding
            if 'localization' in row and pd.notna(row['localization']):
                loc_val = str(row['localization']).lower()
                if loc_val in self.localization_categories:
                    loc_idx = self.localization_categories.index(loc_val)
                    loc_onehot = [0.0] * len(self.localization_categories)
                    loc_onehot[loc_idx] = 1.0
        
        # Concatenate: age (1) + sex (3) + localization (15) = 19
        metadata_vec = [age_norm] + sex_onehot + loc_onehot
        return torch.FloatTensor(metadata_vec)

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

        # CHANGED: Extract metadata vector
        metadata_vec = self._get_metadata_vector(image_id)

        return image, metadata_vec, label  # CHANGED: return metadata alongside image


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
    
    # CHANGED: Prepare metadata dataframe if columns exist
    metadata_df = None
    if 'age' in df.columns or 'sex' in df.columns or 'localization' in df.columns:
        meta_cols = [c for c in ['age', 'sex', 'localization'] if c in df.columns]
        if meta_cols and 'image' in df.columns:
            metadata_df = df.set_index('image')[meta_cols]

    if DEBUG_SUBSET:
        print("\n  [CPU DEBUG MODE] Subsetting dataset to 140 images to save memory/time.")
        # Ensure at least 20 per class if possible, or just slice
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

    # CHANGED: Compute metadata statistics from training set
    age_mean, age_std, age_median = 0.0, 1.0, 0.0
    sex_categories = ['male', 'female', 'unknown']
    localization_categories = [
        'abdomen', 'back', 'chest', 'ear', 'face', 'foot', 'genital',
        'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'upper extremity', 'acral', 'unknown'
    ]
    
    if metadata_df is not None:
        train_metadata = metadata_df.loc[metadata_df.index.isin(train_ids)]
        if 'age' in train_metadata.columns:
            valid_ages = train_metadata['age'].dropna()
            if len(valid_ages) > 0:
                age_mean = valid_ages.mean()
                age_std = valid_ages.std() if valid_ages.std() > 0 else 1.0
                age_median = valid_ages.median()
        
        if 'localization' in train_metadata.columns:
            unique_locs = sorted(train_metadata['localization'].dropna().unique().tolist())
            if len(unique_locs) > 0:
                localization_categories = unique_locs
    
    print(f"  Metadata stats: age_mean={age_mean:.1f}, age_std={age_std:.1f}, n_localizations={len(localization_categories)}")

    # Create datasets
    train_dataset = SkinLesionDataset(
        train_ids, train_labels, image_dir, transform=get_transforms('train'), metadata_df=metadata_df  # CHANGED
    )
    val_dataset = SkinLesionDataset(
        val_ids, val_labels, image_dir, transform=get_transforms('val'), metadata_df=metadata_df  # CHANGED
    )
    test_dataset = SkinLesionDataset(
        test_ids, test_labels, image_dir, transform=get_transforms('test'), metadata_df=metadata_df  # CHANGED
    )
    
    # CHANGED: Set metadata statistics for all datasets
    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.set_metadata_stats(age_mean, age_std, age_median, sex_categories, localization_categories)

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
