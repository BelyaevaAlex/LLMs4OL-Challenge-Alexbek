#!/bin/bash

# Method v5 Inference Script for All Domains
# Runs inference on ecology, engineering, and scholarly domains with RAG data

set -e  # Exit on any error

# Configuration
MODEL_PATH="qwen/Qwen2.5-14B-Instruct"
CONDA_ENV="rah_11_cu12.4_torch"
DATA_DIR="../../../2025/TaskA-Text2Onto-TFIDF-with-RAG"
OUTPUT_DIR="../../../results/method_v5_inference"
SEED=42

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to run inference for a domain
run_inference() {
    local domain=$1
    local input_file="$DATA_DIR/text2onto_${domain}_test_documents_with_rag.jsonl"
    local output_file="$OUTPUT_DIR/method_v5_${domain}_results.json"
    
    log "Starting inference for $domain domain..."
    log "Input: $input_file"
    log "Output: $output_file"
    
    # Check if input file exists
    if [ ! -f "$input_file" ]; then
        log "ERROR: Input file not found: $input_file"
        return 1
    fi
    
    # Run inference (from method_v5 directory)
    cd ../../../
    
    # Adjust paths for root directory
    local root_input_file="2025/TaskA-Text2Onto-TFIDF-with-RAG/text2onto_${domain}_test_documents_with_rag.jsonl"
    local root_output_file="results/method_v5_inference/method_v5_${domain}_results.json"
    
    # Create output directory from root
    mkdir -p "results/method_v5_inference"
    
    python -m src.taskA.method_v5.inference \
        --model-path "$MODEL_PATH" \
        --input "$root_input_file" \
        --output "$root_output_file" \
        --use-tfidf \
        --seed "$SEED"
    cd src/taskA/method_v5/
    
    if [ $? -eq 0 ]; then
        log "✅ Successfully completed inference for $domain"
        log "Results saved to: $output_file"
    else
        log "❌ Failed inference for $domain"
        return 1
    fi
}

# Main execution
main() {
    log "🚀 Starting Method v5 inference for all domains"
    log "Model: $MODEL_PATH"
    log "Conda environment: $CONDA_ENV"
    log "Data directory: $DATA_DIR"
    log "Output directory: $OUTPUT_DIR"
    log "Seed: $SEED"
    echo
    
    # Activate conda environment
    log "Activating conda environment: $CONDA_ENV"
    source ~/.bashrc
    conda activate "$CONDA_ENV"
    
    # Check if outlines is installed
    log "Checking outlines installation..."
    python -c "import outlines; print('✅ outlines is available')" || {
        log "❌ outlines not found. Installing..."
        pip install outlines
    }
    
    # Array of domains to process
    domains=("engineering" "scholarly") # "ecology"
    
    # Track results
    successful_domains=()
    failed_domains=()
    
    # Process each domain
    for domain in "${domains[@]}"; do
        echo
        log "📊 Processing domain: $domain"
        echo "----------------------------------------"
        
        if run_inference "$domain"; then
            successful_domains+=("$domain")
        else
            failed_domains+=("$domain")
        fi
        
        echo "----------------------------------------"
    done
    
    # Summary
    echo
    log "📋 INFERENCE SUMMARY"
    log "===================="
    log "Total domains processed: ${#domains[@]}"
    log "Successful: ${#successful_domains[@]} (${successful_domains[*]})"
    log "Failed: ${#failed_domains[@]} (${failed_domains[*]})"
    
    if [ ${#failed_domains[@]} -eq 0 ]; then
        log "🎉 All domains completed successfully!"
        log "Results available in: $OUTPUT_DIR"
    else
        log "⚠️  Some domains failed. Check logs above for details."
        exit 1
    fi
}

# Check if we're in the method_v5 directory
if [ ! -f "inference.py" ]; then
    log "❌ Error: Please run this script from the method_v5 directory"
    log "Expected to find: inference.py in current directory"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    log "❌ Error: Data directory not found: $DATA_DIR"
    log "Please ensure RAG data is available"
    exit 1
fi

# Run main function
main "$@" 