#!/bin/bash

# MAE Continue Pretraining (MAE-CP) Batch Experiments
# This script runs MAE-CP across multiple datasets and sample sizes

set -e  # Exit on error

# Configuration
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

# Hardware configuration
NUM_GPUS=1
NUM_WORKERS=8
PRECISION="32"

# Training configuration
EPOCHS=100
STEPS_PER_EPOCH=16 
BATCH_SIZE=128
LR=1.5e-4
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=10

# Model configuration
MODEL_SIZE="base"  # Options: base, large, huge
PRETRAINED_SOURCE="facebook/vit-mae-base"
MASK_RATIO=0.75

# Output directory
OUTPUT_DIR="/root/output"
DATA_ROOT="/root/data"

# Dataset and sample size configurations
# Format: "dataset_name:num_classes"
DATASETS=(
    "galaxy10_decals:10"
    "fgvc_aircraft:100"
    "food101:101"
    "bloodmnist:8"
    "pathmnist:9"
    "dermamnist:7"
    "octmnist:4"
    "pneumoniamnist:2"
    "retinamnist:5"
    "breastmnist:2"
    "tissuemnist:8"
    "organamnist:11"
    "organcmnist:11"
    "organsmnist:11"
)

# Sample sizes for few-shot experiments
SAMPLE_SIZES=(
    10
    50
    100
    250
    500
    1000
    2500
    5000
    10000
)

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local limit_data=$2
    local dataset_name=$(echo $dataset | cut -d: -f1)
    local num_classes=$(echo $dataset | cut -d: -f2)
    
    # Set limit_data argument
    if [ "$limit_data" = "full" ]; then
        limit_arg=""
        exp_suffix="full"
    else
        limit_arg="--limit_data $limit_data"
        exp_suffix="${limit_data}"
    fi
    
    # Experiment name
    EXP_NAME="${dataset_name}_mae${MODEL_SIZE}_${exp_suffix}"
    
    echo "================================================"
    echo "Starting experiment: $EXP_NAME"
    echo "Dataset: $dataset_name"
    echo "Sample size: $limit_data"
    echo "Num classes: $num_classes"
    echo "Total steps: $((EPOCHS * STEPS_PER_EPOCH))"
    echo "================================================"
    
    # Run training
    python core/mae_cp_train.py \
        --dataset "$dataset_name" \
        --data_root "$DATA_ROOT" \
        $limit_arg \
        --model_size "$MODEL_SIZE" \
        --pretrained \
        --pretrained_source "$PRETRAINED_SOURCE" \
        --mask_ratio "$MASK_RATIO" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --steps_per_epoch "$STEPS_PER_EPOCH" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --num_workers "$NUM_WORKERS" \
        --precision "$PRECISION" \
        --devices "$NUM_GPUS" \
        --output_dir "$OUTPUT_DIR" \
        --exp_name "$EXP_NAME"
    
    echo "✓ Completed: $EXP_NAME"
    echo ""
}

# Main loop
echo "========================================"
echo "MAE Continue Pretraining - Batch Run"
echo "========================================"
echo "Datasets: ${#DATASETS[@]}"
echo "Sample sizes: ${#SAMPLE_SIZES[@]}"
echo "Total experiments: $((${#DATASETS[@]} * ${#SAMPLE_SIZES[@]}))"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"
echo ""

# Track progress
total_experiments=$((${#DATASETS[@]} * ${#SAMPLE_SIZES[@]}))
current_experiment=0

# Iterate through datasets and sample sizes
for dataset in "${DATASETS[@]}"; do
    for sample_size in "${SAMPLE_SIZES[@]}"; do
        current_experiment=$((current_experiment + 1))
        
        echo ""
        echo "========================================"
        echo "Progress: $current_experiment / $total_experiments"
        echo "========================================"
        
        # Run experiment (with error handling)
        if run_experiment "$dataset" "$sample_size"; then
            echo "✓ Success"
        else
            echo "✗ Failed: $dataset with $sample_size samples"
        fi
    done
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

