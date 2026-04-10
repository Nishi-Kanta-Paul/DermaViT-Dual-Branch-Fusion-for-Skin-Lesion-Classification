"""
Baseline configuration
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

LR_BASE = 1e-4  # Standard AdamW LR for baselines
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15 if HAS_GPU else 1

# ────────────────── Normalization ──────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ────────────────── Class Definitions ──────────────────
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
ONEHOT_COLUMNS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# ────────────────── Paths ──────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "HAM10000")
DATA_ROOT = os.path.abspath(os.getenv("DERMAVIT_DATA_ROOT", DEFAULT_DATA_ROOT))
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
GROUNDTRUTH_CSV = os.path.join(DATA_ROOT, "GroundTruth.csv")

OUTPUT_DIR = os.path.abspath(os.getenv("DERMAVIT_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs")))

SEED = 42
