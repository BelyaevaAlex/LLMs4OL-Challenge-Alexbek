#!/bin/bash

MODEL_PATH="qwen/Qwen2.5-14B-Instruct"
INPUT_PATH="2025/TaskA-Text2Onto-Processed/engineering/train/docs2terms.jsonl"

echo "Starting inference with automatic output filename generation..."

# New version with automatic output filename generation
python -m src.taskA.method_1_doc2type.inference \
    --model-path $MODEL_PATH \
    --input $INPUT_PATH \
    --random-few-shot 3 \
    --use-rag \
    --use-tfidf \
    --use-structured-output

echo "✓ Inference completed. Output filename was generated automatically."