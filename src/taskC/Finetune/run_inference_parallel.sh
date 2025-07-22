#!/bin/bash

# Array of domains
domains=("DOID" "MatOnto" "OBI" "PO" "PROCO" "SWEET" "SchemaOrg" "FoodOn")

# Number of available GPUs
NUM_GPUS=4

# Activate required environment
source /home/jovyan/miniconda3/etc/profile.d/conda.sh
conda activate rah_11_cu12.4_torch

# Navigate to required directory
cd /home/jovyan/rahmatullaev/rand_exps/LLMs4OL-Challenge/src/taskC/method_v6_hm

# Log file for tracking
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Starting inference for ${#domains[@]} domains on $NUM_GPUS GPUs"
echo "Domains: ${domains[@]}"

# Function to run inference for domain
run_inference() {
    local domain=$1
    local gpu_id=$2
    local log_file="$LOG_DIR/inference_${domain}_gpu${gpu_id}.log"
    
    echo "Starting domain $domain on GPU $gpu_id (log: $log_file)"
    
    # Export environment variable for GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Start inference
    python inference_all.py "$domain" --device cuda > "$log_file" 2>&1
    
    # Check result
    if [ $? -eq 0 ]; then
        echo "✅ Domain $domain (GPU $gpu_id) completed successfully"
    else
        echo "❌ Domain $domain (GPU $gpu_id) failed with error"
    fi
}

# Run in batches of NUM_GPUS domains
for ((i=0; i<${#domains[@]}; i+=NUM_GPUS)); do
    echo ""
    echo "=== Batch $((i/NUM_GPUS + 1)) ==="
    
    # Start processes for current batch
    for ((j=0; j<NUM_GPUS && (i+j)<${#domains[@]}; j++)); do
        domain=${domains[i+j]}
        gpu_id=$j
        run_inference "$domain" "$gpu_id" &
    done
    
    # Wait for all processes in batch to complete
    wait
    
    echo "Batch $((i/NUM_GPUS + 1)) completed"
done

echo ""
echo "🎉 All domains processed!"
echo "Check logs in folder $LOG_DIR"

# Show final statistics
echo ""
echo "=== FINAL STATISTICS ==="
for domain in "${domains[@]}"; do
    log_file="$LOG_DIR/inference_${domain}_gpu*.log"
    if ls $log_file 1> /dev/null 2>&1; then
        echo -n "📊 $domain: "
        if grep -q "completed successfully" $log_file; then
            echo "✅ SUCCESS"
        else
            echo "❌ ERROR"
        fi
    else
        echo "📊 $domain: ❓ NOT FOUND"
    fi
done

echo ""
echo "To view logs use:"
echo "tail -f $LOG_DIR/inference_<domain>_gpu<id>.log" 