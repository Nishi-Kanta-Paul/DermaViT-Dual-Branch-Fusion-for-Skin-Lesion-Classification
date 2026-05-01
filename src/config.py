# ═══ DermaViT v2.1 ═══
# Modified: Clinical weights, label smoothing, MixUp, multi-seed, external validation
# All changes marked with # CHANGED
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

# CHANGED: Ablation study mode
ABLATION_MODE = False  # Set to True to run hyperparameter ablation study

# CHANGED: Multi-seed training mode (Fix 3)
MULTI_SEED_MODE = False  # Set to True to run training across multiple seeds
TRAINING_SEEDS = [42, 0, 1, 7, 123]  # CHANGED: Seeds for multi-seed training

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

# CHANGED: Label smoothing for loss function (Fix 2b)
LABEL_SMOOTHING = 0.1  # CHANGED

# CHANGED: MixUp augmentation (Fix 4b)
MIXUP_ALPHA = 0.2  # CHANGED: Beta distribution parameter for MixUp
MIXUP_PROB = 0.3  # CHANGED: Probability of applying MixUp per batch

# CHANGED: Ablation search spaces
SE_REDUCTION_RATIOS = [8, 16, 32]
DROPOUT_RATES = [0.3, 0.4, 0.5]
LR_CNN_OPTIONS = [1e-4, 5e-5]
LR_SWIN_OPTIONS = [5e-5, 1e-5]
ABLATION_EPOCHS = 20 if HAS_GPU else 1
ABLATION_PATIENCE = 5 if HAS_GPU else 1

# ────────────────── Normalization ──────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ────────────────── Class Definitions ──────────────────
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
LABEL_MAP = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}

# CHANGED: Clinical weight multipliers (Fix 2a)
# Multiply computed class weights by these factors to emphasize clinically critical classes
CLINICAL_WEIGHT_MULTIPLIER = {  # CHANGED
    'MEL': 2.5,    # CHANGED: malignant — highest cost
    'BCC': 1.5,    # CHANGED: malignant
    'AKIEC': 1.5,  # CHANGED: pre-malignant
    'NV': 1.0,     # CHANGED: benign
    'BKL': 1.0,    # CHANGED: benign
    'DF': 1.0,     # CHANGED: benign
    'VASC': 1.0    # CHANGED: benign
}  # CHANGED

# Column order in metadata.csv (one-hot encoded)
ONEHOT_COLUMNS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# ────────────────── Paths ──────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_ROOT = os.path.abspath(os.getenv("DERMAVIT_DATA_ROOT", DEFAULT_DATA_ROOT))
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
GROUNDTRUTH_CSV = os.path.join(DATA_ROOT, "metadata.csv")

OUTPUT_DIR = os.path.abspath(os.getenv("DERMAVIT_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs")))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
SALIENCY_DIR = os.path.join(OUTPUT_DIR, "saliency_maps")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

SEED = 42

# CHANGED: External validation settings (Fix 5)
EXTERNAL_VALIDATION = False  # CHANGED: Set to True to run external validation
EXTERNAL_DATA_PATH = ""  # CHANGED: Path to external dataset images
EXTERNAL_METADATA_PATH = ""  # CHANGED: Path to external metadata CSV
EXTERNAL_DATASET_NAME = "ISIC2019"  # CHANGED: Name of external dataset
