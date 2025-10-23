#!/bin/bash
#SBATCH --job-name=curvadion_rank_sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=experiements/slurm_logs/rank_sweep_%j.out
#SBATCH --error=experiements/slurm_logs/rank_sweep_%j.err

################################################################################
# CurvaDion Rank Fraction Sensitivity Sweep
#
# Runs Dion (baseline) and CurvaDion across different rank fractions:
#   - 0.0625 (1/16)
#   - 0.125  (1/8)
#   - 0.25   (1/4)
#
# All experiments use the same shared_init.pkl for fair comparison.
# Results logged to experiements/logs/ and WandB project: curvadion-rank-fraction-sensitivity
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "=========================================="
echo "CurvaDion Rank Fraction Sensitivity Sweep"
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
INIT_FILE="experiements/shared_init_rank_sweep.pkl"
JSON_LOG_DIR="experiements/logs"
WANDB_PROJECT="curvadion-rank-fraction-sensitivity"

# Inverse rank fractions to test (rank = 1/inv_rank_fraction)
INV_RANK_FRACTIONS=(16 8 4)

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
# Step 2: Run Dion baseline for all rank fractions
################################################################################

echo "=========================================="
echo "Running Dion Baseline Experiments"
echo "=========================================="
echo ""

for inv_rank in "${INV_RANK_FRACTIONS[@]}"; do
    echo "--------------------------------------"
    echo "Dion Baseline - inv_rank_fraction=$inv_rank (rank = 1/$inv_rank)"
    echo "--------------------------------------"
    echo ""

    srun torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        train.py \
            --config "$CONFIG_FILE" \
            --init_from "$INIT_FILE" \
            --optimizer dion_simple \
            --inv_rank_fraction "$inv_rank" \
            --wandb_project_name "$WANDB_PROJECT" \
            --wandb_job_name "rank_sweep" \
            --json_log_dir "$JSON_LOG_DIR"

    echo ""
    echo "Completed: Dion baseline with inv_rank_fraction=$inv_rank"
    echo ""
done

################################################################################
# Step 3: Run CurvaDion for all rank fractions
################################################################################

echo "=========================================="
echo "Running CurvaDion Experiments"
echo "=========================================="
echo ""

for inv_rank in "${INV_RANK_FRACTIONS[@]}"; do
    echo "--------------------------------------"
    echo "CurvaDion - inv_rank_fraction=$inv_rank (rank = 1/$inv_rank)"
    echo "--------------------------------------"
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
            --inv_rank_fraction "$inv_rank" \
            --wandb_project_name "$WANDB_PROJECT" \
            --wandb_job_name "rank_sweep" \
            --json_log_dir "$JSON_LOG_DIR"

    echo ""
    echo "Completed: CurvaDion with inv_rank_fraction=$inv_rank"
    echo ""
done

################################################################################
# Summary
################################################################################

echo "=========================================="
echo "Rank Fraction Sweep Complete!"
echo "=========================================="
echo ""
echo "End time: $(date)"
echo ""
echo "Summary:"
echo "  - Tested inverse rank fractions: ${INV_RANK_FRACTIONS[*]}"
echo "  - Corresponding rank fractions: 1/16, 1/8, 1/4"
echo "  - Total experiments: $((2 * ${#INV_RANK_FRACTIONS[@]})) (Dion + CurvaDion)"
echo "  - Shared initialization: $INIT_FILE"
echo "  - JSON logs: $JSON_LOG_DIR"
echo "  - WandB project: $WANDB_PROJECT"
echo ""
echo "JSON log files created:"
ls -lh "$JSON_LOG_DIR"/*invrank* 2>/dev/null || echo "  (No files found - check for errors)"
echo ""
echo "=========================================="
