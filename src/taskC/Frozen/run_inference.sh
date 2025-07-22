#!/bin/bash

# Script for running Cross-Attention model inference
# Usage: ./run_inference.sh <results_dir> <terms_file> <output_dir> [custom_threshold]

set -e

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <results_dir> <terms_file> <output_dir> [custom_threshold]"
    echo ""
    echo "Example:"
    echo "  $0 results/20250707_192128_MatOnto_ep10_lr5e-06_bs16_eval10_seed42 \\"
    echo "     /path/to/terms.txt \\"
    echo "     inference_results/MatOnto_test"
    echo ""
    echo "Arguments:"
    echo "  results_dir      - Training results folder"
    echo "  terms_file       - .txt file with terms"
    echo "  output_dir       - Folder to save results"
    echo "  custom_threshold - Optional custom threshold (default from training)"
    exit 1
fi

RESULTS_DIR="$1"
TERMS_FILE="$2"
OUTPUT_DIR="$3"
CUSTOM_THRESHOLD="${4:-}"

# Check if files exist
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Results folder not found: $RESULTS_DIR"
    exit 1
fi

if [ ! -f "$TERMS_FILE" ]; then
    echo "❌ Terms file not found: $TERMS_FILE"
    exit 1
fi

if [ ! -f "$RESULTS_DIR/best_results.json" ]; then
    echo "❌ best_results.json file not found in $RESULTS_DIR"
    exit 1
fi

# Create output folder
mkdir -p "$OUTPUT_DIR"

# Activate environment
echo "🔄 Activating environment..."
source activate rah_python312_cuda124 || source activate rah_11_cu12.4_torch || {
    echo "❌ Failed to activate environment"
    exit 1
}

# Launch information
echo "🚀 Starting Cross-Attention model inference"
echo "📁 Training results: $RESULTS_DIR"
echo "📝 Terms file: $TERMS_FILE"
echo "💾 Output folder: $OUTPUT_DIR"

# Count number of terms
TERM_COUNT=$(wc -l < "$TERMS_FILE")
echo "📊 Number of terms: $TERM_COUNT"

# Estimate execution time
MATRIX_SIZE=$((TERM_COUNT * TERM_COUNT))
echo "🔢 Matrix size: $TERM_COUNT x $TERM_COUNT = $MATRIX_SIZE cells"

if [ $TERM_COUNT -gt 1000 ]; then
    echo "⚠️  Large number of terms - process may take a long time"
fi

# Create log file
LOG_FILE="$OUTPUT_DIR/inference_$(date +%Y%m%d_%H%M%S).log"
echo "📋 Log will be saved to: $LOG_FILE"

# Build command
CMD="python inference_cross_attention.py \
    --results_dir=\"$RESULTS_DIR\" \
    --terms_file=\"$TERMS_FILE\" \
    --output_dir=\"$OUTPUT_DIR\" \
    --log_file=\"$LOG_FILE\" \
    --embedding_batch_size=32 \
    --prediction_batch_size=64"

# Add custom threshold if specified
if [ -n "$CUSTOM_THRESHOLD" ]; then
    CMD="$CMD --custom_threshold=$CUSTOM_THRESHOLD"
    echo "🎯 Using custom threshold: $CUSTOM_THRESHOLD"
fi

echo ""
echo "🚀 Starting inference..."
echo "Command: $CMD"
echo ""

# Run with time measurement
START_TIME=$(date +%s)

eval $CMD

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅ Inference completed successfully!"
echo "⏱️  Execution time: $DURATION seconds"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "📊 Created files:"
echo "   📈 prediction_matrix.npy      - Prediction matrix"
echo "   🔗 predicted_pairs.json       - Child-parent pairs"
echo "   📝 terms.json                 - Terms list"
echo "   📋 inference_metadata.json    - Inference metadata"
echo "   📊 prediction_distribution.png - Distribution plot"
echo "   📄 $(basename "$LOG_FILE")                  - Execution log"
echo ""

# Show brief statistics
if [ -f "$OUTPUT_DIR/inference_metadata.json" ]; then
    echo "📈 Brief statistics:"
    python -c "
import json
with open('$OUTPUT_DIR/inference_metadata.json', 'r') as f:
    meta = json.load(f)
info = meta['inference_info']
print(f'   Terms: {info[\"n_terms\"]}')
print(f'   Found pairs: {info[\"n_pairs\"]}')
print(f'   Used threshold: {info[\"threshold_used\"]:.3f}')
print(f'   Positive pairs percentage: {info[\"n_pairs\"]/(info[\"n_terms\"]*(info[\"n_terms\"]-1))*100:.2f}%')
"
fi

echo ""
echo "🎉 Done! Use predicted_pairs.json file for further processing." 