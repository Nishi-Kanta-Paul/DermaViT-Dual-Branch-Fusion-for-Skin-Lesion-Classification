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
DermaViT_Workspace/
├── HAM10000/                # Dataset directory (Images & GroundTruth.csv)
├── DermaViT/                # Our proposed architecture & main implementation
│   ├── config.py            # Core hyperparameters
│   ├── dataset.py           # Dataloaders with stratified splits & augmentation
│   ├── model.py             # Dual-branch Fusion Model (EfficientNet + Swin)
│   ├── train.py             # Optimized training loop (Cosine Annealing + AMP)
│   ├── evaluate.py          # Metric calculations & ROC curve generation
│   ├── explainability.py    # Grad-CAM & Attention joint saliency mapping
│   ├── main.py              # Single-file pipeline orchestration
│   └── requirements.txt     # Python dependencies
└── Baselines/               # Standard architectures for fair comparison
    ├── config.py            # Comparative hyperparameters
    ├── dataset.py           # Shared data pipeline
    ├── utils.py             # Shared utilities
    ├── train_resnet50.py
    ├── train_efficientnet_b2.py
    ├── train_vit_b16.py
    ├── train_swin_t.py
    └── compare_results.py   # Aggregates evaluation logs into comparison tables
```

---

## 🛠️ Setup & Installation

**1. Create a Python environment & install requirements:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r DermaViT/requirements.txt
```

**2. Prepare the HAM10000 Dataset:**
Place your downloaded dataset in the root folder such that:
- `HAM10000/GroundTruth.csv` exists (One-hot encoded label columns).
- `HAM10000/images/` contains all `ISIC_XXXXXXX.jpg` image files.

---

## 🏃‍♂️ Running the Models

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
