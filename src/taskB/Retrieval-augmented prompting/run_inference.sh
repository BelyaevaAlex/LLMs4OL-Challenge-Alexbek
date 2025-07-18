#!/bin/bash

# TaskB-TermTyping Inference Script
# Usage: ./run_inference.sh <domain> <data_type> <model_path> [additional_args]
# Example: ./run_inference.sh SWEET test microsoft/Qwen2.5-14B-Instruct --few-shot-amount 5

set -e

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <domain> <data_type> <model_path> [additional_args]"
    echo "Example: $0 SWEET test Qwen/Qwen2.5-14B-Instruct --few-shot-amount 5"
    echo ""
    echo "Available domains: SWEET, MatOnto, OBI"
    echo "Available data types: train, test"
    exit 1
fi

DOMAIN=$1
DATA_TYPE=$2
MODEL_PATH=$3
shift 3  # Remove first 3 arguments, keep the rest as additional args

# Validate domain
case $DOMAIN in
    SWEET|MatOnto|OBI)
        ;;
    *)
        echo "Error: Invalid domain '$DOMAIN'. Available domains: SWEET, MatOnto, OBI"
        exit 1
        ;;
esac

# Validate data type
case $DATA_TYPE in
    train|test)
        ;;
    *)
        echo "Error: Invalid data type '$DATA_TYPE'. Available types: train, test"
        exit 1
        ;;
esac

# Set input file based on domain and data type
BASE_PATH="2025/TaskB-TermTyping"

if [ "$DATA_TYPE" = "train" ]; then
    INPUT_FILE="$BASE_PATH/$DOMAIN/train/term_typing_train_data.json"
elif [ "$DATA_TYPE" = "test" ]; then
    INPUT_FILE="$BASE_PATH/$DOMAIN/test/terms2types.json"
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

echo "=== TaskB-TermTyping Inference ==="
echo "Domain: $DOMAIN"
echo "Data type: $DATA_TYPE"
echo "Model: $MODEL_PATH"
echo "Input file: $INPUT_FILE"
echo "Additional args: $@"
echo "=================================="

# Run inference with RAG support by default
CUDA_VISIBLE_DEVICES=0 python -m src.taskB.method_v1_rag.inference \
    --model-path "$MODEL_PATH" \
    --input "$INPUT_FILE" \
    --domain "$DOMAIN" \
    --use-rag \
    --use-structured-output \
    --seed 42 \
    --batch-size 16 \
    "$@"

echo "Inference completed successfully!" 