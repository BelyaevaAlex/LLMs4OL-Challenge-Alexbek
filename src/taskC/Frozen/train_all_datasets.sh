#!/bin/bash

# Script for training Cross-Attention model on all datasets

# Setup paths (environment should be activated beforehand)
BASE_DIR="/home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge"
SCRIPT_DIR="${BASE_DIR}/src/taskC/method_v5_hm"

# Change to working directory
cd "$SCRIPT_DIR"

# List of all datasets
DATASETS=("DOID" "FoodOn" "MatOnto" "OBI" "PO" "PROCO" "SchemaOrg" "SWEET")

# Default training parameters
DEFAULT_ARGS="--epochs 5 --batch_size 32 --eval_every 100 --save_every 500"

# If arguments are provided, use them instead of default parameters
if [ $# -gt 0 ]; then
    TRAINING_ARGS="$@"
else
    TRAINING_ARGS="$DEFAULT_ARGS"
fi

echo "============================================="
echo "Training Cross-Attention Model on All Datasets"
echo "============================================="
echo "Datasets: ${DATASETS[@]}"
echo "Training args: $TRAINING_ARGS"
echo "============================================="

# Counters for statistics
TOTAL_DATASETS=${#DATASETS[@]}
SUCCESS_COUNT=0
FAILED_DATASETS=()

# Training on each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "[$((SUCCESS_COUNT + ${#FAILED_DATASETS[@]} + 1))/$TOTAL_DATASETS] Starting training on $dataset..."
    echo ""
    
    # Start training
    ./train_single_dataset.sh "$dataset" $TRAINING_ARGS
    
    # Check result
    if [ $? -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✅ $dataset: SUCCESS"
    else
        FAILED_DATASETS+=("$dataset")
        echo "❌ $dataset: FAILED"
    fi
done

echo ""
echo "============================================="
echo "Training Summary"
echo "============================================="
echo "Total datasets: $TOTAL_DATASETS"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: ${#FAILED_DATASETS[@]}"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo "Failed datasets: ${FAILED_DATASETS[@]}"
fi

echo "============================================="

# Return error code if there are failures
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    exit 1
else
    echo "🎉 All datasets trained successfully!"
    exit 0
fi 