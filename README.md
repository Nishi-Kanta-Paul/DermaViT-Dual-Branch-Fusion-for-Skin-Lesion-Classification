# DermaViT: Dual-Branch Fusion for Skin Lesion Classification

This repository contains the official implementation of **DermaViT**, a deep learning architecture designed for highly accurate and explainable 7-class skin lesion classification using the HAM10000 dataset. 

DermaViT leverages a **dual-branch fusion architecture** combining the local feature extraction capabilities of **EfficientNet-B0** with the global context modeling of **Swin Transformer (Swin-T)**. The branches are fused using a customized Squeeze-and-Excitation (SE) attention block to dynamically reweight spatial and semantic features.

## 🚀 Features

- **Dual-Branch Architecture**: EfficientNet-B0 (CNN) + Swin-T (Vision Transformer)
- **SE-Based Fusion**: Cross-attention feature consolidation mechanism
- **Complete Pipeline**: Includes pre-processing (CLAHE), dynamic augmentations (Albumentations), mixed-precision training (AMP), and thorough evaluation metrics.
- **Explainability Module**: Joint visualization combining CNN Grad-CAM with Swin-T Attention Rollout.
- **Evaluation Workbench**: Direct head-to-head evaluation scripts against state-of-the-art baselines (ResNet-50, EfficientNet-B2, ViT-B/16, Swin-T).

---

## 📂 Project Structure

```text
DermaViT/
├── README.md
├── requirements.txt
├── .gitignore
├── baselines/               # Baseline model training and comparison scripts
│   ├── compare_results.py
│   ├── config.py
│   ├── dataset.py
│   ├── train_efficientnet_b2.py
│   ├── train_resnet50.py
│   ├── train_swin_t.py
│   ├── train_vit_b16.py
│   └── utils.py
├── data/                    # Dataset metadata, images, and segmentation masks
│   ├── metadata.csv
│   ├── images/
│   └── masks/
├── experiments/             # Per-model artifacts (checkpoints/logs/results)
│   ├── derma_vit/
│   ├── efficientnet_b2/
│   ├── resnet50/
│   ├── swin_t/
│   └── vit_b16/
├── notebooks/
├── outputs/
├── scripts/
│   └── train.sh
└── src/                     # Main DermaViT training/evaluation/inference code
    ├── config.py
    ├── dataset.py
    ├── evaluate.py
    ├── explainability.py
    ├── inference.py
    ├── main.py
    ├── model.py
    ├── train.py
    └── utils.py
```

---

## 🛠️ Setup & Installation

**1. Create a Python environment & install requirements:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Prepare the dataset in `data/`:**
Use the current repository layout and ensure the following paths exist:
- `data/metadata.csv` (metadata/labels file)
- `data/images/` (all dermoscopy images)
- `data/masks/` (segmentation masks, if used)

**3. Quick sanity check (optional):**
```bash
ls data
```
You should see at least: `images`, `masks`, and `metadata.csv`.

---

## 🏃‍♂️ Running the Models

## ⚡ Google Colab Fast Iteration (No Re-Upload of Dataset)

If you make frequent code changes, do not zip/upload the full workspace every time.

### One-Time Setup (Drive)
1. Upload dataset once to Drive, for example:
    - `/content/drive/MyDrive/DermaViT_Research/data/`
2. Keep outputs persistent in Drive as well, for example:
    - `/content/drive/MyDrive/DermaViT_Research/outputs/`

### Recommended Code Sync Strategy
Use Git for code sync (small changes = only `git pull` in Colab):

```bash
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!git clone <your-repo-url> DermaViT_Workspace || true
%cd /content/DermaViT_Workspace
!git pull
```

### Run with Persistent Dataset + Output Paths
This project now supports environment-variable based paths:
- `DERMAVIT_DATA_ROOT`
- `DERMAVIT_OUTPUT_DIR`

```bash
%cd /content/DermaViT_Workspace
!pip install -r DermaViT/requirements.txt

import os
os.environ["DERMAVIT_DATA_ROOT"] = "/content/drive/MyDrive/DermaViT_Research/data"
os.environ["DERMAVIT_OUTPUT_DIR"] = "/content/drive/MyDrive/DermaViT_Research/outputs"

!chmod +x run_all.sh
!./run_all.sh
```

Now dataset upload is one-time only. For later code edits, just push locally and `git pull` in Colab.

### 1. Train and Evaluate the Proposed DermaViT
Run the complete pipeline—this will train the model, evaluate test metrics (Acc, Precision, Recall, F1, AUC), and generate both plots and 4-panel explainability saliency maps.

```bash
python DermaViT/main.py
```
All outputs (model checkpoints, ROC curves, confusion matrices, and saliency maps) are saved in `DermaViT/outputs/`.

### 2. Train Standard Baselines
To evaluate the DermaViT architecture against standard state-of-the-art implementations, run the baseline scripts. They utilize the exact same dataset splits, learning environments, and early-stopping mechanisms to ensure a fair comparison.

```bash
python Baselines/train_resnet50.py
python Baselines/train_efficientnet_b2.py
python Baselines/train_vit_b16.py
python Baselines/train_swin_t.py
```

### 3. Generate Performance Comparison Table
Once the baselines and the DermaViT model have completed training/testing, generate the benchmarking table matching the research specifications:

```bash
python Baselines/compare_results.py
```

---

## 📊 Evaluation & Metrics Collected

Our unified evaluation script automatically logs the following metrics for all models:
- **Accuracy (%)**
- **Precision (Macro-averaged)**
- **Recall (Macro-averaged)**
- **F1 Score (Macro-averaged)**
- **AUC (Macro-averaged One-vs-Rest)**

---

## 🔬 Explainable AI (XAI)
DermaViT incorporates a native XAI system ensuring its medical predictions are interpretable.
The `explainability.py` module correlates:
1. **CNN Grad-CAM** tracking spatial borders via the EfficientNet branch.
2. **Attention Rollout** mapping global contextual similarities via the Swin-T branch.
3. **Joint Saliency** overlapping both perspectives dynamically onto the original dermoscopic image.
