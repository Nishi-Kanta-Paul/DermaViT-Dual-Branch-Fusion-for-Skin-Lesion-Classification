"""
DermaViT Configuration
Automatically detects GPU to set optimal memory/training bounds.
"""
import os
import torch

# ────────────────── AUTO HARDWARE DETECTION ──────────────────
HAS_GPU = torch.cuda.is_available()

if not HAS_GPU:
    print("[CONFIG] No GPU detected! Switching to CPU-friendly hyper-parameters (DEBUG_SUBSET=True, BATCH_SIZE=2, EPOCHS=1).")

DEBUG_SUBSET = not HAS_GPU  # If True, subset dataset to 140 images

# ────────────────── Image & Training ──────────────────
IMG_SIZE = 224
BATCH_SIZE = 32 if HAS_GPU else 2
NUM_CLASSES = 7
NUM_EPOCHS = 50 if HAS_GPU else 1

LR_EFFICIENTNET = 1e-4
LR_SWIN = 5e-5
LR_FUSION = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
SE_REDUCTION_RATIO = 16
LAMBDA_SALIENCY = 0.5  
EARLY_STOPPING_PATIENCE = 15 if HAS_GPU else 1

# ────────────────── Normalization ──────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ────────────────── Class Definitions ──────────────────
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
LABEL_MAP = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}

# Column order in GroundTruth.csv (one-hot encoded)
ONEHOT_COLUMNS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# ────────────────── Paths ──────────────────
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HAM10000"))
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
GROUNDTRUTH_CSV = os.path.join(DATA_ROOT, "GroundTruth.csv")

OUTPUT_DIR = "outputs"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
SALIENCY_DIR = os.path.join(OUTPUT_DIR, "saliency_maps")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

SEED = 42
