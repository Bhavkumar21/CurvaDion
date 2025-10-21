# CurvaDion: Adaptive Communication Optimizer

**CurvaDion** is an adaptive communication optimizer for distributed training that reduces synchronization overhead by only syncing when momentum changes significantly.

## Key Innovation

Based on Dion (https://arxiv.org/abs/2504.05295), CurvaDion introduces an **adaptive communication schedule** controlled by the RMMC (Relative Momentum Magnitude Change) metric:

```
RMMC_ℓ(t) = |‖M_ℓ(t)‖ - ‖M_ℓ(t-1)‖| / ‖M_ℓ(t-1)‖
```

**Decision logic per step:**
1. Calculate RMMC for each layer
2. All-reduce a boolean flag: does any GPU need sync? **(4 bytes)**
3. If `max(RMMC) > threshold`: Full gradient sync
4. Else: Local updates only (models diverge slightly, reconverge at next sync)

## Communication Savings

**Per step communication:**
- **Stable training** (RMMC < threshold): **4 bytes** (just the flag)
- **Unstable training** (RMMC > threshold): **4 bytes + gradient all-reduces**

**Compared to standard DDP:**
- Standard DDP: Gradient all-reduce every step
- CurvaDion: Gradient all-reduce only when needed


## Installation

```bash
cd ~/workspace/CurvaDion
pip install -r requirements.txt
```
Download pretokenized FineWeb dataset:
```bash
python data/cached_fineweb10B.py 30
```

## Quick Start (DDP with 2 GPUs)

```bash
# Train GPT-small with CurvaDion
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/curvadion_160m.yaml
```

## Configuration

Key hyperparameters in `configs/curvadion_160m.yaml`:

```yaml
optimizer: curvadion
rmmc_threshold: 0.1              # Lower = more syncs, higher = more comm savings
rank_fraction: 0.125              # 1/8 rank for Dion low-rank updates
mixed_precision: true             # Use bfloat16 for optimizer states
replicate_mesh_grad_sync: true   # REQUIRED: Disables DDP auto-sync, lets optimizer control sync
```

**⚠️ IMPORTANT:** `replicate_mesh_grad_sync: true` is **required** for CurvaDion to work correctly. Without it, PyTorch DDP will automatically synchronize gradients every step, making CurvaDion's adaptive sync mechanism meaningless.

### Tuning `rmmc_threshold`

- **Too low (< 0.05)**: Syncs too often, minimal communication savings
- **Too high (> 0.3)**: Models may diverge, poor convergence
- **Sweet spot:** 0.1 - 0.2

## Tracked Metrics

CurvaDion automatically tracks and logs:

- `curvadion/sync_rate`: Fraction of steps that triggered full sync
- `curvadion/max_rmmc`: Current RMMC value
- `curvadion/avg_max_rmmc`: Average RMMC over training
- `curvadion/recent_bytes`: Bytes transferred in last step
- `curvadion/total_bytes_MB`: Total communication in MB
- `curvadion/avg_bytes_per_step`: Average bytes per step

## Implementation Details

### DDP-Only Design

This implementation is **simplified for DDP only** (no FSDP/TP support). Key design choices:

1. **Natural reconvergence**: No special divergence correction, models naturally reconverge
2. **Static threshold**: Single hyperparameter throughout training
3. **Boolean flag only**: Minimal overhead for sync decision
4. **Byte tracking**: Precise accounting of all communication

### Optimizer Structure

- **`curvadion.py`**: Main adaptive optimizer
  - Tracks momentum norms per layer
  - Calculates RMMC each step
  - Routes to sync or local update
  - Tracks bytes transferred

- **`dion_simple.py`**: Used for local updates (0 communication)

- **`scalar_opts.py`**: AdamW/Lion for non-matrix parameters


## Directory Structure

```
CurvaDion/
├── optimizers/
│   ├── curvadion.py           # Adaptive communication optimizer
│   ├── dion_simple.py         # Local Dion updates
│   ├── dion.py                # Full Dion (not used)
│   ├── scalar_opts.py         # AdamW/Lion
│   └── opt_utils.py           # Utilities
├── models/
│   ├── gpt_model.py           # GPT architecture
│   └── gpt_utils.py           # Data loaders
├── configs/
│   └── curvadion_160m.yaml    # Config file
├── train.py                   # Training script
├── requirements.txt
└── README.md
```
