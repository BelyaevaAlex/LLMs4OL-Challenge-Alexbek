#!/bin/bash

# Run inference for term-to-type task with RAG support
# Usage: ./run_inference_base.sh [domain] [data_type] [model_path]

set -e

# Default parameters
DOMAIN=${1:-"ecology"}  # engineering, scholarly, ecology
DATA_TYPE=${2:-"test"}      # train, test
MODEL_PATH=${3:-"Qwen/Qwen2.5-14B-Instruct"}  # Path to model

# Paths
BASE_DIR="./LLMs4OL-Challenge-AlexBek"
INPUT_DIR="${BASE_DIR}/2025/TaskA-Text2Onto-TFIDF-with_scores+/${DOMAIN}/${DATA_TYPE}"
INPUT_FILE="${INPUT_DIR}/terms2types.json"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo "Please make sure the data has been prepared with RAG examples."
    exit 1
fi

echo "=== Term-to-Type Inference ==="
echo "Domain: $DOMAIN"
echo "Data type: $DATA_TYPE"
echo "Model: $MODEL_PATH"
echo "Input: $INPUT_FILE"
echo "================================"

# Run inference with RAG
python -m src.taskA.method_v3_t2.inference \
    --model-path "$MODEL_PATH" \
    --input "$INPUT_FILE" \
    --use-rag \
    --random-few-shot 5 \
    --use-structured-output \
    --seed 42

echo "Inference completed!" 