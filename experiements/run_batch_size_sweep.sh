#!/bin/bash
#SBATCH --job-name=curvadion_batch_sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=experiements/slurm_logs/batch_sweep_%j.out
#SBATCH --error=experiements/slurm_logs/batch_sweep_%j.err

################################################################################
# CurvaDion Batch Size Sensitivity Sweep
#
# Runs Dion (baseline) and CurvaDion across different batch sizes:
#   - 512, 1024, 2048, 4096, 8192
#
# All experiments use the same shared_init.pkl for fair comparison.
# Results logged to experiements/logs/ and WandB project: curvadion-batch-sensitivity
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "=========================================="
echo "CurvaDion Batch Size Sensitivity Sweep"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo ""

# Navigate to project directory
cd /mnt/weka/home/hjcpuro/bhav_stuff/CurvaDion

# Configuration
NUM_GPUS=8
CONFIG_FILE="experiements/configs/base_sensitivity.yaml"
INIT_FILE="experiements/shared_init_batch_sweep.pkl"
JSON_LOG_DIR="experiements/logs"
WANDB_PROJECT="curvadion-batch-sensitivity"

# Batch sizes to test
BATCH_SIZES=(512 1024 2048 4096 8192)

# Create log directory if it doesn't exist
mkdir -p "$JSON_LOG_DIR"
mkdir -p "experiements/slurm_logs"

################################################################################
# Step 1: Create shared initialization (if it doesn't exist)
################################################################################

if [ ! -f "$INIT_FILE" ]; then
    echo "=========================================="
    echo "Creating shared initialization: $INIT_FILE"
    echo "=========================================="
    echo ""

    srun torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        train.py \
            --config "$CONFIG_FILE" \
            --save_init "$INIT_FILE"

    echo ""
    echo "Shared initialization created successfully!"
    echo "File: $INIT_FILE"
    echo "Size: $(du -h $INIT_FILE | cut -f1)"
    echo ""
else
    echo "=========================================="
    echo "Using existing initialization: $INIT_FILE"
    echo "Size: $(du -h $INIT_FILE | cut -f1)"
    echo "=========================================="
    echo ""
fi

################################################################################
# Step 2: Run Dion baseline for all batch sizes
################################################################################

echo "=========================================="
echo "Running Dion Baseline Experiments"
echo "=========================================="
echo ""

for batch_size in "${BATCH_SIZES[@]}"; do
    echo "--------------------------------------"
    echo "Dion Baseline - Batch Size: $batch_size"
    echo "--------------------------------------"
    echo ""

    # Calculate device_batch_size to maintain reasonable gradient accumulation
    # while avoiding OOM for large batch sizes
    if [ $batch_size -le 1024 ]; then
        device_batch_size=32
    elif [ $batch_size -le 2048 ]; then
        device_batch_size=32
    elif [ $batch_size -le 4096 ]; then
        device_batch_size=64
    else
        # For batch_size=8192, use smaller device_batch_size to avoid OOM
        device_batch_size=64
    fi

    echo "  Global batch size: $batch_size"
    echo "  Device batch size: $device_batch_size"
    echo "  Gradient accumulation: $((batch_size / (device_batch_size * NUM_GPUS)))"
    echo ""

    srun torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        train.py \
            --config "$CONFIG_FILE" \
            --init_from "$INIT_FILE" \
            --optimizer dion_simple \
            --inv_rank_fraction 8 \
            --batch_size "$batch_size" \
            --device_batch_size "$device_batch_size" \
            --wandb_project_name "$WANDB_PROJECT" \
            --wandb_job_name "bs=${batch_size}" \
            --json_log_dir "$JSON_LOG_DIR"

    echo ""
    echo "Completed: Dion baseline with batch_size=$batch_size"
    echo ""
done

################################################################################
# Step 3: Run CurvaDion for all batch sizes
################################################################################

echo "=========================================="
echo "Running CurvaDion Experiments"
echo "=========================================="
echo ""

for batch_size in "${BATCH_SIZES[@]}"; do
    echo "--------------------------------------"
    echo "CurvaDion - Batch Size: $batch_size"
    echo "--------------------------------------"
    echo ""

    # Calculate device_batch_size (same logic as above)
    # while avoiding OOM for large batch sizes
    if [ $batch_size -le 1024 ]; then
        device_batch_size=32
    elif [ $batch_size -le 2048 ]; then
        device_batch_size=32
    elif [ $batch_size -le 4096 ]; then
        device_batch_size=64
    else
        # For batch_size=8192, use smaller device_batch_size to avoid OOM
        device_batch_size=64
    fi

    echo "  Global batch size: $batch_size"
    echo "  Device batch size: $device_batch_size"
    echo "  Gradient accumulation: $((batch_size / (device_batch_size * NUM_GPUS)))"
    echo ""

    srun torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        train.py \
            --config "$CONFIG_FILE" \
            --init_from "$INIT_FILE" \
            --optimizer curvadion \
            --rmmc_threshold 0.1 \
            --inv_rank_fraction 8 \
            --batch_size "$batch_size" \
            --device_batch_size "$device_batch_size" \
            --wandb_project_name "$WANDB_PROJECT" \
            --wandb_job_name "bs=${batch_size}" \
            --json_log_dir "$JSON_LOG_DIR"

    echo ""
    echo "Completed: CurvaDion with batch_size=$batch_size"
    echo ""
done

################################################################################
# Summary
################################################################################

echo "=========================================="
echo "Batch Size Sweep Complete!"
echo "=========================================="
echo ""
echo "End time: $(date)"
echo ""
echo "Summary:"
echo "  - Tested batch sizes: ${BATCH_SIZES[*]}"
echo "  - Total experiments: $((2 * ${#BATCH_SIZES[@]})) (Dion + CurvaDion)"
echo "  - Shared initialization: $INIT_FILE"
echo "  - JSON logs: $JSON_LOG_DIR"
echo "  - WandB project: $WANDB_PROJECT"
echo ""
echo "JSON log files created:"
ls -lh "$JSON_LOG_DIR"/*batch* 2>/dev/null || echo "  (No files found - check for errors)"
echo ""
echo "=========================================="
