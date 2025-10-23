#!/bin/bash
#SBATCH --job-name=diloco_sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00
#SBATCH --output=threshold_sensitivity/slurm_logs/%j_diloco_sweep.out
#SBATCH --error=threshold_sensitivity/slurm_logs/%j_diloco_sweep.err

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of GPUs: 8"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Set working directory
cd /mnt/weka/home/hjcpuro/bhav_stuff/CurvaDion

# Configuration
INIT_FILE="threshold_sensitivity/shared_init.pkl"
JSON_LOG_DIR="threshold_sensitivity/logs"

# Verify initialization file exists
if [ ! -f "$INIT_FILE" ]; then
    echo "ERROR: Initialization file not found: $INIT_FILE"
    exit 1
fi

# Create log directory
mkdir -p "$JSON_LOG_DIR"
mkdir -p threshold_sensitivity/slurm_logs

# Array of H values to test
H_VALUES=(5 10 50 250)

# Run experiments sequentially
for H in "${H_VALUES[@]}"; do
    CONFIG_FILE="threshold_sensitivity/diloco_h${H}.yaml"

    echo ""
    echo "=========================================="
    echo "Running DiLoCo with H=$H"
    echo "=========================================="
    echo "Config: $CONFIG_FILE"
    echo "Initialization: $INIT_FILE"
    echo "JSON logs: $JSON_LOG_DIR"
    echo "Start time: $(date)"
    echo "=========================================="
    echo ""

    # Verify config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        echo "Skipping H=$H"
        continue
    fi

    # Run training with torchrun
    srun torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=8 \
        train.py \
        --config "$CONFIG_FILE" \
        --init_from "$INIT_FILE" \
        --json_log_dir "$JSON_LOG_DIR"

    EXIT_CODE=$?

    echo ""
    echo "=========================================="
    echo "H=$H completed with exit code: $EXIT_CODE"
    echo "Completion time: $(date)"
    echo "=========================================="
    echo ""

    # If training failed, continue to next H value
    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: Training with H=$H failed, continuing to next value"
    fi

    # Small delay between runs
    sleep 5
done

echo ""
echo "=========================================="
echo "All experiments completed"
echo "End time: $(date)"
echo "=========================================="
