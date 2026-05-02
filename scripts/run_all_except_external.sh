#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================"
echo "DermaViT v2.1 - Complete Pipeline"
echo "(All modes except External Validation)"
echo "========================================"
echo ""
echo "Project root: $SCRIPT_DIR"
echo "DERMAVIT_DATA_ROOT=${DERMAVIT_DATA_ROOT:-$SCRIPT_DIR/data}"
echo "DERMAVIT_OUTPUT_DIR=${DERMAVIT_OUTPUT_DIR:-$SCRIPT_DIR/outputs}"
echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 1: Normal Training (Single Seed)
# ═══════════════════════════════════════════════════════════════
echo "========================================"
echo "STEP 1/5: Normal Training (Single Seed)"
echo "========================================"
echo "Running: python src/main.py"
python src/main.py
echo "✓ Normal training complete"
echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 2: Multi-Seed Training
# ═══════════════════════════════════════════════════════════════
echo "========================================"
echo "STEP 2/5: Multi-Seed Training (5 Seeds)"
echo "========================================"
echo "Enabling MULTI_SEED_MODE..."
sed -i 's/MULTI_SEED_MODE = False/MULTI_SEED_MODE = True/' src/config.py
echo "Running: python src/main.py (multi-seed)"
python src/main.py
echo "✓ Multi-seed training complete"
echo "Disabling MULTI_SEED_MODE..."
sed -i 's/MULTI_SEED_MODE = True/MULTI_SEED_MODE = False/' src/config.py
echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 3: Ablation Study
# ═══════════════════════════════════════════════════════════════
echo "========================================"
echo "STEP 3/5: Hyperparameter Ablation Study"
echo "========================================"
echo "Enabling ABLATION_MODE..."
sed -i 's/ABLATION_MODE = False/ABLATION_MODE = True/' src/config.py
echo "Running: python src/main.py (ablation)"
python src/main.py
echo "✓ Ablation study complete"
echo "Disabling ABLATION_MODE..."
sed -i 's/ABLATION_MODE = True/ABLATION_MODE = False/' src/config.py
echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 4: Baseline Models
# ═══════════════════════════════════════════════════════════════
echo "========================================"
echo "STEP 4/5: Training Baseline Models"
echo "========================================"
echo "Training ResNet-50..."
python baselines/train_resnet50.py
echo "✓ ResNet-50 complete"

echo "Training EfficientNet-B2..."
python baselines/train_efficientnet_b2.py
echo "✓ EfficientNet-B2 complete"

echo "Training ViT-B/16..."
python baselines/train_vit_b16.py
echo "✓ ViT-B/16 complete"

echo "Training Swin-T..."
python baselines/train_swin_t.py
echo "✓ Swin-T complete"
echo ""

# ═══════════════════════════════════════════════════════════════
# STEP 5: Compare Results
# ═══════════════════════════════════════════════════════════════
echo "========================================"
echo "STEP 5/5: Comparing All Results"
echo "========================================"
echo "Running: python baselines/compare_results.py"
python baselines/compare_results.py
echo "✓ Comparison complete"
echo ""

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
echo "========================================"
echo "✓ COMPLETE PIPELINE FINISHED"
echo "========================================"
echo ""
echo "Generated Outputs:"
echo "  • outputs/best_model.pth (single seed)"
echo "  • outputs/seed_*/best_model_seed*.pth (multi-seed)"
echo "  • outputs/results/multi_seed_summary.txt"
echo "  • outputs/ablation_results.csv"
echo "  • outputs/results/ (evaluation & saliency)"
echo "  • outputs/ResNet-50/summary_metrics.txt"
echo "  • outputs/EfficientNet-B2/summary_metrics.txt"
echo "  • outputs/ViT-B16/summary_metrics.txt"
echo "  • outputs/Swin-T/summary_metrics.txt"
echo "  • outputs/comparison_table.md"
echo ""
echo "⚠️  NOTE: External Validation was SKIPPED as requested"
echo ""
echo "Check all results in: ${DERMAVIT_OUTPUT_DIR:-$SCRIPT_DIR/outputs}"
echo "========================================"
