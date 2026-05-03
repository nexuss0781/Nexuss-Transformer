#!/bin/bash
# ============================================================================
# Blank Slate Training Script for Ethiopian Religious Texts
# ============================================================================
# 
# This script trains a decoder-only transformer from scratch (blank slate)
# on two datasets:
#   1. Synaxarium - Ethiopian Orthodox daily readings (366 entries)
#   2. Canon Biblical - Amharic-English parallel Bible corpus (31,920 verses)
#
# The trained model will have these religious texts as frozen base knowledge.
# ============================================================================

set -e

echo "============================================================================"
echo "BLANK SLATE TRAINING - Ethiopian Religious Texts"
echo "============================================================================"
echo ""
echo "Datasets:"
echo "  - Synaxarium: 366 daily readings (Ethiopian Orthodox)"
echo "  - Canon Biblical: 31,920 Amharic-English verses"
echo ""
echo "Model Configuration:"
echo "  - Architecture: Small (~25M parameters)"
echo "  - Vocabulary: Character-level (~379 tokens)"
echo "  - Layers: 6 transformer blocks"
echo "  - Attention heads: 8"
echo "  - Embedding dimension: 512"
echo "  - Max sequence length: 512"
echo ""
echo "Training Configuration:"
echo "  - Epochs: 10"
echo "  - Effective batch size: 16"
echo "  - Learning rate: 1e-3 with linear decay"
echo "  - Warmup: 5% of steps"
echo "  - Checkpointing: Every 100 steps"
echo ""
echo "Output Directory: ./outputs/blank_slate_ethiopian_religious/"
echo "============================================================================"
echo ""

# Run training
python3 train_blank_slate.py

echo ""
echo "============================================================================"
echo "TRAINING COMPLETED!"
echo "============================================================================"
echo ""
echo "Model saved to: ./outputs/blank_slate_ethiopian_religious/"
echo ""
echo "Next steps:"
echo "  1. Evaluate the model on held-out test data"
echo "  2. Freeze the base knowledge weights"
echo "  3. Fine-tune on downstream tasks if needed"
echo "  4. Deploy for inference"
echo "============================================================================"
