#!/bin/bash

# This script runs CurvaDion experiments with different RMMC thresholds
# All experiments use the same initialization for fair comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/threshold_sweep.yaml"
INIT_FILE="$SCRIPT_DIR/shared_init.pkl"
RESULTS_DIR="$SCRIPT_DIR/results"
JSON_LOG_DIR="$SCRIPT_DIR/logs"

# Thresholds to sweep
THRESHOLDS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ $NUM_GPUS -eq 0 ]; then
    echo "Error: No GPUs detected!"
    exit 1
fi

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$JSON_LOG_DIR"

echo "================================================"
echo "CurvaDion Threshold Sensitivity Sweep"
echo "================================================"
echo "Config file: $CONFIG_FILE"
echo "Initialization file: $INIT_FILE"
echo "Results directory: $RESULTS_DIR"
echo "JSON logs directory: $JSON_LOG_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Thresholds: ${THRESHOLDS[@]}"
echo "Note: Checkpointing disabled - metrics only"
echo "================================================"

# Step 1: Create shared initialization
if [ ! -f "$INIT_FILE" ]; then
    echo ""
    echo "Step 1: Creating shared initialization..."
    echo "================================================"
    cd "$PROJECT_ROOT"
    torchrun --standalone --nproc_per_node=$NUM_GPUS train.py \
        --config "$CONFIG_FILE" \
        --save_init "$INIT_FILE"

    echo "Initialization saved to $INIT_FILE"
else
    echo ""
    echo "Step 1: Using existing initialization at $INIT_FILE"
    echo "================================================"
fi

# Step 2: Run experiments for each threshold
echo ""
echo "Step 2: Running threshold experiments..."
echo "================================================"

for threshold in "${THRESHOLDS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running experiment with threshold = $threshold"
    echo "----------------------------------------"

    # Format threshold for directory name (replace . with _)
    threshold_str=$(echo $threshold | sed 's/\./_/g')

    # Run training
    cd "$PROJECT_ROOT"
    torchrun --standalone --nproc_per_node=$NUM_GPUS train.py \
        --config "$CONFIG_FILE" \
        --init_from "$INIT_FILE" \
        --rmmc_threshold $threshold \
        --json_log_dir "$JSON_LOG_DIR" \
        --wandb_job_name "threshold_${threshold_str}"

    echo "Completed threshold = $threshold"
    echo "Logs saved to: $JSON_LOG_DIR"
done

echo ""
echo "================================================"
echo "All experiments completed!"
echo "================================================"
echo "Results are saved in:"
echo "  - JSON logs: $JSON_LOG_DIR"
echo "  - WandB project: curvadion-threshold-sensitivity"
echo ""
echo "To analyze results:"
echo "  1. Check JSON files: $JSON_LOG_DIR"
echo "  2. Run: python analyze_results.py"
echo "  3. View WandB dashboard: https://wandb.ai"
echo "================================================"
