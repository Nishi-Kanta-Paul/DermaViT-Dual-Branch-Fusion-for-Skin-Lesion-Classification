#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Project root: $SCRIPT_DIR"
echo "DERMAVIT_DATA_ROOT=${DERMAVIT_DATA_ROOT:-$SCRIPT_DIR/HAM10000}"
echo "DERMAVIT_OUTPUT_DIR=${DERMAVIT_OUTPUT_DIR:-$SCRIPT_DIR/outputs}"

echo "Starting DermaViT Main Pipeline..."
python3 DermaViT/main.py

echo "Starting Baselines..."
python3 Baselines/train_resnet50.py
python3 Baselines/train_efficientnet_b2.py
python3 Baselines/train_vit_b16.py
python3 Baselines/train_swin_t.py

echo "Comparing Results..."
python3 Baselines/compare_results.py
