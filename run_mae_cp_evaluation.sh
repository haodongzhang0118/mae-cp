#!/bin/bash

# Simple MAE-CP Evaluation Script

set -e  # Exit on error

# Default configuration
OUTPUT_ROOT="/root/output"
DATA_ROOT="/root/data"
OUTPUT_CSV="mae_cp_metrics.csv"
MODEL_SIZE="base"  # base, large, huge
CHECKPOINT_TYPE="best_f1"  # best_f1 or last

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        --output_csv) OUTPUT_CSV="$2"; shift 2 ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --checkpoint_type) CHECKPOINT_TYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run evaluation
echo "Starting MAE-CP evaluation..."
echo "Output: ${OUTPUT_CSV}"
echo ""

python "${SCRIPT_DIR}/test/collect_mae_cp_metrics.py" \
    --output_root "${OUTPUT_ROOT}" \
    --data_root "${DATA_ROOT}" \
    --model_size "${MODEL_SIZE}" \
    --checkpoint_type "${CHECKPOINT_TYPE}" \
    --output_csv "${OUTPUT_CSV}"

echo ""
echo "Done! Results saved to: ${OUTPUT_CSV}"
