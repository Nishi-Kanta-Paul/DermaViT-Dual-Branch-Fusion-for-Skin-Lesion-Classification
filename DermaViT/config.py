"""
DermaViT Configuration
All hyperparameters and paths in one place.
"""
import os

# ────────────────── Image & Training ──────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7
NUM_EPOCHS = 50
LR_EFFICIENTNET = 1e-4
LR_SWIN = 5e-5
LR_FUSION = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
SE_REDUCTION_RATIO = 16
LAMBDA_SALIENCY = 0.5  # weight for joint saliency map
EARLY_STOPPING_PATIENCE = 15

# ────────────────── Normalization ──────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ────────────────── Class Definitions ──────────────────
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
LABEL_MAP = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}

# Column order in GroundTruth.csv (one-hot encoded)
ONEHOT_COLUMNS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

CLASS_FULL_NAMES = {
    'MEL':   'Melanoma',
    'NV':    'Melanocytic Nevi',
    'BCC':   'Basal Cell Carcinoma',
    'AKIEC': 'Actinic Keratoses',
    'BKL':   'Benign Keratosis',
    'DF':    'Dermatofibroma',
    'VASC':  'Vascular Lesions',
}

# ────────────────── Paths ──────────────────
DATA_ROOT = "HAM10000"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
GROUNDTRUTH_CSV = os.path.join(DATA_ROOT, "GroundTruth.csv")
OUTPUT_DIR = "outputs"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
SALIENCY_DIR = os.path.join(OUTPUT_DIR, "saliency_maps")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

SEED = 42
