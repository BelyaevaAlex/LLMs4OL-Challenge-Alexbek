#!/bin/bash

# Script for training Cross-Attention model on a single dataset

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_name> [additional_args...]"
    echo "Available datasets: DOID, FoodOn, MatOnto, OBI, PO, PROCO, SchemaOrg, SWEET"
    echo "Example: $0 DOID --epochs 10 --batch_size 32"
    exit 1
fi

DATASET_NAME=$1
shift  # Remove first argument, pass the rest to python script

# Setup paths (environment should be activated beforehand)
BASE_DIR="/home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge"
DATA_DIR="${BASE_DIR}/2025/TaskC-TaxonomyDiscovery/${DATASET_NAME}/train"
SCRIPT_DIR="${BASE_DIR}/src/taskC/method_v5_hm"
OUTPUT_DIR="${SCRIPT_DIR}/results_pool"  # Base folder, unique subfolder will be created automatically

# Check if dataset exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory not found: $DATA_DIR"
    exit 1
fi

# Check required files
ENTITIES_FILE="${DATA_DIR}/${DATASET_NAME,,}_train_types_embeddings_pool.json"
RELATIONS_FILE="${DATA_DIR}/${DATASET_NAME,,}_train_pairs.json"

echo "USING POOLING DATASET!!!"

if [ ! -f "$ENTITIES_FILE" ]; then
    echo "Error: Entities file not found: $ENTITIES_FILE"
    exit 1
fi

if [ ! -f "$RELATIONS_FILE" ]; then
    echo "Error: Relations file not found: $RELATIONS_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch information
echo "============================================="
echo "Training Cross-Attention Model"
echo "============================================="
echo "Dataset: $DATASET_NAME"
echo "Entities file: $ENTITIES_FILE"
echo "Relations file: $RELATIONS_FILE"
echo "Output directory: $OUTPUT_DIR (unique experiment folder will be created)"
echo "Additional args: $@"
echo "============================================="

# Change to working directory
cd "$SCRIPT_DIR"

# Start training
python train_cross_attention.py \
    --entities_path "$ENTITIES_FILE" \
    --relations_path "$RELATIONS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_name "$DATASET_NAME" \
    "$@"

# Check result
if [ $? -eq 0 ]; then
    echo "============================================="
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "============================================="
else
    echo "============================================="
    echo "Training failed!"
    echo "Check logs for details."
    echo "============================================="
    exit 1
fi 