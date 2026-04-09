#!/bin/bash
set -e
echo "Starting DermaViT Main Pipeline..."
python3 DermaViT/main.py

echo "Starting Baselines..."
python3 Baselines/train_resnet50.py
python3 Baselines/train_efficientnet_b2.py
python3 Baselines/train_vit_b16.py
python3 Baselines/train_swin_t.py

echo "Comparing Results..."
python3 Baselines/compare_results.py
