"""
Baseline configuration
Reuses DermaViT dataset/training logic but drops fusion-specific details.
"""
import os

# ────────────────── Image & Training ──────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7
NUM_EPOCHS = 50
LR_BASE = 1e-4  # Standard AdamW LR for baselines
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15

# ────────────────── Normalization ──────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ────────────────── Class Definitions ──────────────────
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
ONEHOT_COLUMNS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# ────────────────── Paths ──────────────────
# Data root sits one level above
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HAM10000"))
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
GROUNDTRUTH_CSV = os.path.join(DATA_ROOT, "GroundTruth.csv")

OUTPUT_DIR = "outputs"
# Subdirectories will be created per model (e.g. outputs/ResNet-50/)

SEED = 42
