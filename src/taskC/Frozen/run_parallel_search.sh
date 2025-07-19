#!/bin/bash

# Script for running parallel parameter search
# Uses GPU 2 and 8 parallel processes

# Activate environment
source /home/jovyan/.mlspace/envs/rah_python312_cuda124/lib/python3.12/venv/scripts/common/activate

# Setup paths
BASE_DIR="/home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge"
SCRIPT_DIR="${BASE_DIR}/src/taskC/method_v5_hm"

# Change to working directory
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_name> [--gpu N] [--eval_every N] [--save_every N] [--num_experiments N] [--num_processes N] [--seed N]"
    echo ""
    echo "Available datasets:"
    echo "  DOID      - Disease Ontology (662MB)"
    echo "  FoodOn    - Food Ontology (2.1GB)"
    echo "  MatOnto   - Materials Ontology (44MB)"
    echo "  OBI       - Ontology for Biomedical Investigations (287MB)"
    echo "  PO        - Plant Ontology (98MB)"
    echo "  PROCO     - Process Ontology (53MB)"
    echo "  SchemaOrg - Schema.org (47MB)"
    echo "  SWEET     - Semantic Web for Earth and Environmental Terminology (510MB)"
    echo ""
    echo "Examples:"
    echo "  $0 SchemaOrg"
    echo "  $0 DOID --gpu 1 --eval_every 20 --save_every 100"
    echo "  $0 MatOnto --gpu 2 --num_experiments 16 --num_processes 4"
    echo ""
    echo "Default parameters:"
    echo "  --gpu 2                (use GPU 2)"
    echo "  --eval_every 50        (evaluate every 50 steps)"
    echo "  --save_every 250       (save every 250 steps)"
    echo "  --num_experiments 180  (run 180 unique experiments)"
    echo "  --num_processes 8      (use 8 parallel processes)"
    echo "  --seed 42              (random seed)"
    echo ""
    echo "Search space (180 unique combinations):"
    echo "  lr: [1e-5, 5e-5, 1e-4, 5e-4]         (4 values)"
    echo "  epochs: [3, 5, 8, 10, 12]             (5 values)"
    echo "  batch_size: [16, 32, 64]              (3 values)"
    echo "  num_attention_heads: [8, 16, 32]      (3 values)"
    echo ""
    echo "Fixed parameters:"
    echo "  test_size: 0.2"
    echo "  dataset_strategy: single"
    echo "  sampling_strategy: balanced"
    echo "  positive_ratio: 1.0"
    echo "  use_qwen3: false"
    exit 1
fi

DATASET_NAME=$1
shift  # Remove first argument

# Parse --gpu argument
GPU_NUM=2  # Default value

# Check if --gpu is in arguments
for arg in "$@"; do
    if [[ "$arg" == "--gpu" ]]; then
        # Get next argument as GPU number
        for ((i=1; i<=$#; i++)); do
            if [[ "${!i}" == "--gpu" ]]; then
                next_i=$((i+1))
                if [[ $next_i -le $# ]]; then
                    GPU_NUM="${!next_i}"
                    break
                fi
            fi
        done
        break
    fi
done

# Launch logging
echo "============================================="
echo "🚀 Parallel Parameter Search"
echo "============================================="
echo "Dataset: $DATASET_NAME"
echo "GPU: $GPU_NUM"
echo "Additional parameters: $@"
echo "Start time: $(date)"
echo "============================================="

# Check if script file exists
if [ ! -f "train_parallel_search.py" ]; then
    echo "❌ Error: train_parallel_search.py file not found"
    exit 1
fi

# Check if dataset exists
DATA_DIR="${BASE_DIR}/2025/TaskC-TaxonomyDiscovery/${DATASET_NAME}/train"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Dataset folder not found: $DATA_DIR"
    echo "Available datasets:"
    ls "${BASE_DIR}/2025/TaskC-TaxonomyDiscovery/" 2>/dev/null || echo "  Dataset folder not found"
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 Checking GPU..."
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -5
    echo ""
else
    echo "⚠️ nvidia-smi not found, skipping GPU check"
fi

# Start parallel search
echo "🚀 Starting parallel search..."
export CUDA_VISIBLE_DEVICES=$GPU_NUM
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python train_parallel_search.py "$DATASET_NAME" "$@"

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "✅ Parallel search completed successfully!"
    echo "============================================="
    echo "End time: $(date)"
    echo "Results saved to: results/parallel_search_${DATASET_NAME}_*"
    echo ""
    echo "To view results:"
    echo "  - Final report: final_search_report.json"
    echo "  - Experiment parameters: experiments_params.json"
    echo "  - Result folders: results/parallel_search_${DATASET_NAME}_*/"
    echo "============================================="
else
    echo ""
    echo "============================================="
    echo "❌ Parallel search failed!"
    echo "============================================="
    echo "End time: $(date)"
    echo "Check logs for detailed information"
    echo "============================================="
    exit 1
fi 