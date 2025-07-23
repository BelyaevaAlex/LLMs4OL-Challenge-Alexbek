#!/bin/bash

# Simple Method v5 Inference Script
# Quick run on all domains

echo "🚀 Starting Method v5 inference on all domains..."

# Activate environment
source ~/.bashrc
conda activate rah_11_cu12.4_torch

# Go to project root and create results directory
cd ../../../
mkdir -p results/method_v5_inference

echo "📊 Running ecology domain..."
python -m src.taskA.method_v5.inference \
    --model-path qwen/Qwen2.5-14B-Instruct \
    --input 2025/TaskA-Text2Onto-TFIDF-with-RAG/text2onto_ecology_test_documents_with_rag.jsonl \
    --output results/method_v5_inference/ecology_results.json \
    --use-tfidf \
    --seed 42

echo "🏗️ Running engineering domain..."
python -m src.taskA.method_v5.inference \
    --model-path qwen/Qwen2.5-14B-Instruct \
    --input 2025/TaskA-Text2Onto-TFIDF-with-RAG/text2onto_engineering_test_documents_with_rag.jsonl \
    --output results/method_v5_inference/engineering_results.json \
    --use-tfidf \
    --seed 42

echo "📚 Running scholarly domain..."
python -m src.taskA.method_v5.inference \
    --model-path qwen/Qwen2.5-14B-Instruct \
    --input 2025/TaskA-Text2Onto-TFIDF-with-RAG/text2onto_scholarly_test_documents_with_rag.jsonl \
    --output results/method_v5_inference/scholarly_results.json \
    --use-tfidf \
    --seed 42

echo "🎉 All domains completed! Results in results/method_v5_inference/" 