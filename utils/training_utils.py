"""
Training infrastructure for CurvaDion.

This module provides:
- Distributed training initialization (DDP and DeviceMesh)
- Optimizer initialization with parameter groups
- Checkpoint management (save/load with atomic operations)
- JSON logging
- Model initialization save/load for reproducible experiments
- Learning rate scheduling
"""

import argparse
import json
import math
import os
import pickle
import shutil
import tempfile
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional

from models.gpt_model import GPT
from models.gpt_utils import DistributedDataLoader
from optimizers import CurvaDion, DionSimple, DionMixedPrecisionConfig
from optimizers.diloco_wrapper import DiLoCoWrapper
from utils.config import Hyperparameters, print0


def init_distributed(dp_size, fs_size, tp_size) -> Optional[DeviceMesh]:
    """
    Initialize DeviceMesh or ProcessGroup for distributed training.

    If all mesh dimensions are None, we default to using DDP (DistributedDataParallel).
    Otherwise, we create a 3D device mesh for advanced parallelism (DP, FSDP, TP).

    Args:
        dp_size: Data parallel size (replicate dimension)
        fs_size: Fully sharded data parallel size
        tp_size: Tensor parallel size

    Returns:
        DeviceMesh if using device mesh, None if using DDP

    Raises:
        AssertionError: If environment variables are not set or mesh dimensions don't match world_size
    """
    assert torch.distributed.is_available(), "Distributed must be available"

    # Check that environment variables are set
    assert all(
        var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    ), "This script must be launched using the 'torchrun' command."
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set global master process flag
    import utils.config
    utils.config.MASTER_PROCESS = (rank == 0)

    # Initialize process group first - NCCL will handle CUDA initialization properly
    mesh_dims = (dp_size, fs_size, tp_size)
    if all(d is None for d in mesh_dims):
        # If no mesh dimensions given, initialize process group for DDP
        device_mesh = None
        dist.init_process_group(backend="nccl")

        # Set device AFTER init_process_group
        torch.cuda.set_device(local_rank)

        print0("=" * 80)
        print0("Distributed training initialized with DDP")
        print0(f"World size: {world_size}")

    else:
        # Use device mesh for distributed training
        # All mesh dimensions must be specified
        assert all(
            d is not None for d in mesh_dims
        ), f"All mesh dimensions (dp_size, fs_size, tp_size) must be specified, but got ({dp_size}, {fs_size}, {tp_size})"

        # Check if we have the right number of GPUs
        total_gpus = dp_size * fs_size * tp_size
        assert world_size == total_gpus, (
            f"World size {world_size} does not match expected size {total_gpus} "
            f"(DP {dp_size}, FS {fs_size}, TP {tp_size})"
        )
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(dp_size, fs_size, tp_size),
            mesh_dim_names=("dp", "fs", "tp"),
        )

        print0("=" * 80)
        print0("Distributed training initialized with DeviceMesh")
        print0(f"World size: {world_size}")
        print0(f"DP size: {dp_size}")
        print0(f"FS size: {fs_size}")
        print0(f"TP size: {tp_size}")
        print0(device_mesh)

    return device_mesh


def init_optimizer(
    model: GPT,
    device_mesh: Optional[DeviceMesh],
    ddp_model: Optional[DDP],
    hp: Hyperparameters,
    cli_args: argparse.Namespace,
):
    """
    Initialize optimizer with parameter groups.

    Creates different parameter groups for:
    - Matrix parameters (transformer layers): Use Dion with low-rank updates
    - Embedding parameters: Use scalar optimizer (Lion/AdamW) with no weight decay
    - LM head parameters: Use scalar optimizer with scaled learning rate

    Args:
        model: The raw GPT model (unwrapped)
        device_mesh: Device mesh if using FSDP/TP, None for DDP
        ddp_model: DDP-wrapped model if using DDP, None otherwise
        hp: Hyperparameters configuration
        cli_args: CLI arguments

    Returns:
        Initialized optimizer (CurvaDion or DionSimple)

    Raises:
        ValueError: If optimizer or scalar_opt is invalid, or if replicate_mesh_grad_sync is misconfigured
    """
    # Check that we have a valid scalar optimizer
    if hp.scalar_opt not in ["adamw", "lion"]:
        raise ValueError(f"Unrecognized scalar optimizer: {hp.scalar_opt}")

    # Separate the model's parameters based on their types
    matrix_params = list(model.transformer.h.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())

    # Matrix params use optimizer default settings
    param_groups = [dict(params=matrix_params)]

    # Add additional param groups with the necessary configurations for scalar params
    param_groups.append(
        dict(
            params=embedding_params,
            algorithm=hp.scalar_opt,
            lr=hp.lr,  # no LR adjustment for embedding parameters
            betas=(0.95, 0.98),
            weight_decay=0,  # no weight decay for embedding parameters
        )
    )
    param_groups.append(
        dict(
            params=lm_head_params,
            algorithm=hp.scalar_opt,
            lr=hp.lr / math.sqrt(hp.model_dim),  # scale LR for lm_head
            betas=(0.95, 0.98),
            weight_decay=0,  # no weight decay for lm_head parameters
        )
    )

    # Get process group for distributed optimizers
    if ddp_model is not None:
        process_group = ddp_model.process_group
    else:
        # No DDP model - must be using device mesh or diloco without DDP
        if device_mesh is not None:
            raise NotImplementedError("Device mesh support not yet implemented in this path")
        # For diloco_dion without DDP, use the default world process group
        process_group = dist.group.WORLD if dist.is_initialized() else None

    # CurvaDion and diloco_dion REQUIRE replicate_mesh_grad_sync=True to work correctly
    # Without it, DDP will sync gradients automatically, making the optimizer's sync schedule meaningless
    if hp.optimizer in ["curvadion", "diloco_dion"]:
        if not hp.replicate_mesh_grad_sync:
            raise ValueError(
                f"{hp.optimizer} requires replicate_mesh_grad_sync=True to function correctly.\n"
                f"Without it, PyTorch DDP will sync gradients every step automatically,\n"
                f"making the optimizer's adaptive sync mechanism meaningless.\n"
                f"Please add 'replicate_mesh_grad_sync: true' to your config file."
            )

    if hp.mixed_precision:
        dion_mixed_precision_config = DionMixedPrecisionConfig(
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            Q_dtype=torch.bfloat16,
        )
    else:
        dion_mixed_precision_config = None

    if hp.optimizer == "curvadion":
        print0(f"CurvaDion rank fraction: {hp.rank_fraction}")
        print0(f"CurvaDion RMMC threshold: {hp.rmmc_threshold}")
        print0(f"CurvaDion mixed precision: {hp.mixed_precision}")
        print0(f"CurvaDion debug: {cli_args.curvadion_debug}")
        opt = CurvaDion(
            param_groups,
            process_group=process_group,
            rmmc_threshold=hp.rmmc_threshold,
            track_metrics=True,
            debug=cli_args.curvadion_debug,
            lr=hp.lr,
            mu=hp.mu,
            rank_fraction=hp.rank_fraction,
            weight_decay=hp.weight_decay,
            mixed_precision_config=dion_mixed_precision_config,
        )

    elif hp.optimizer == "dion_simple":
        print0(f"DionSimple rank fraction: {hp.rank_fraction}")
        print0(f"DionSimple mixed precision: {hp.mixed_precision}")
        opt = DionSimple(
            param_groups,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            rank=round(hp.rank_fraction * hp.model_dim),
            mixed_precision_config=dion_mixed_precision_config,
        )

    elif hp.optimizer == "diloco_dion":
        print0(f"DiLoCo with Dion rank fraction: {hp.rank_fraction}")
        print0(f"DiLoCo sync every: {hp.sync_every} steps")
        print0(f"DiLoCo mixed precision: {hp.mixed_precision}")

        # Create inner DionSimple optimizer
        inner_opt = DionSimple(
            param_groups,
            lr=hp.lr,
            mu=hp.mu,
            weight_decay=hp.weight_decay,
            rank=round(hp.rank_fraction * hp.model_dim),
            mixed_precision_config=dion_mixed_precision_config,
        )

        # Wrap with DiLoCoWrapper for periodic synchronization
        opt = DiLoCoWrapper(
            inner_optimizer=inner_opt,
            model=model,  # Pass the raw model for parameter sync
            replicate_mesh=process_group,
            num_inner_steps=hp.sync_every,
        )

    else:
        raise ValueError(f"Unsupported optimizer: {hp.optimizer}. Only 'curvadion', 'dion_simple', and 'diloco_dion' are supported.")

    return opt


def get_lr_schedule_fn(hp: Hyperparameters):
    """
    Create learning rate schedule function with 3 phases.

    The schedule has:
    1. Linear warmup from 0 to 1 (first warmup_ratio of training)
    2. Constant at 1 (middle of training)
    3. Linear warmdown from 1 to 0 (last warmdown_ratio of training)

    Args:
        hp: Hyperparameters with num_iterations, warmup_ratio, warmdown_ratio

    Returns:
        Function that takes iteration and returns LR multiplier
    """
    warmup_iters = round(hp.warmup_ratio * hp.num_iterations)
    warmdown_iters = round(hp.warmdown_ratio * hp.num_iterations)

    def get_lr(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= hp.num_iterations - warmdown_iters:
            return 1.0
        else:
            return (hp.num_iterations - it) / warmdown_iters

    return get_lr


class CheckpointManager:
    """
    Manages distributed checkpoint save/load with atomic operations.

    Handles:
    - Model and optimizer state
    - DataLoader positions
    - Training step and WandB ID
    - Atomic save (temp directory then move)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DistributedDataLoader,
        val_loader: DistributedDataLoader,
        wandb_id: Optional[str] = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            model: Model to checkpoint (DDP or FSDP wrapped)
            optimizer: Optimizer to checkpoint
            train_loader: Training dataloader (to save position)
            val_loader: Validation dataloader (to save position)
            wandb_id: WandB run ID for resume
        """
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb_id = wandb_id
        self.step = None
        self.DEFAULT_NAME = "checkpoint"

    def _get_state_dict(self) -> dict:
        """
        Bundle all checkpoint state into a dictionary.

        Uses get_state_dict() to standardize state dicts regardless of sharding.

        Returns:
            Dictionary with model, optimizer, dataloaders, step, wandb_id
        """
        model_state, opt_state = get_state_dict(self.model, self.optimizer)
        state_dict = {
            "model": model_state,
            "optimizer": opt_state,
            "train_loader": self.train_loader.state_dict(),
            "val_loader": self.val_loader.state_dict(),
            "step": self.step,
            "wandb_id": self.wandb_id,
        }
        return state_dict

    def save(self, name: Optional[str] = None, step: Optional[int] = None):
        """
        Save checkpoint atomically to shared filesystem.

        Uses temp directory approach:
        1. All ranks save to temp dir
        2. Rank 0 moves temp dir to final location (atomic)

        Args:
            name: Checkpoint name (default: "checkpoint")
            step: Current training step
        """
        assert self.checkpoint_dir, "Checkpoint directory must be specified"
        self.step = step
        name = name or self.DEFAULT_NAME
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        print0(f"Saving checkpoint to {checkpoint_path}")

        # Save to a temporary subdirectory first
        tmpdir = None
        if dist.get_rank() == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            tmpdir = tempfile.mkdtemp(dir=self.checkpoint_dir)

        # Broadcast tmpdir from rank 0 to all ranks
        obj_list = [tmpdir]
        dist.broadcast_object_list(obj_list, src=0)
        tmpdir = obj_list[0]

        # Save the checkpoint
        state_dict = self._get_state_dict()
        dcp.save(state_dict, checkpoint_id=tmpdir)
        dist.barrier()

        if dist.get_rank() == 0:
            # Delete any existing checkpoint with the same name
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            elif os.path.isdir(checkpoint_path):
                shutil.rmtree(checkpoint_path, ignore_errors=True)
            # Move the checkpoint to the final location
            shutil.move(tmpdir, checkpoint_path)
        dist.barrier()

    def load(self, name: Optional[str] = None, allow_missing: bool = False):
        """
        Load checkpoint from shared filesystem.

        Args:
            name: Checkpoint name (default: "checkpoint")
            allow_missing: If True, silently skip if checkpoint doesn't exist

        Raises:
            FileNotFoundError: If checkpoint doesn't exist and allow_missing=False
        """
        assert self.checkpoint_dir, "Checkpoint directory must be specified"
        name = name or self.DEFAULT_NAME
        checkpoint_path = os.path.join(self.checkpoint_dir, name)

        if not os.path.isdir(checkpoint_path):
            if allow_missing:
                print0(f"Checkpoint {checkpoint_path} does not exist, skipping load")
                return
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")

        print0(f"Loading checkpoint from {checkpoint_path}")
        state_dict = self._get_state_dict()
        dcp.load(state_dict, checkpoint_id=checkpoint_path)

        # Load model and optimizer state dicts
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )

        # Load train and validation dataloader states
        self.train_loader.load_state_dict(state_dict["train_loader"])
        self.val_loader.load_state_dict(state_dict["val_loader"])

        self.step = state_dict["step"]
        self.wandb_id = state_dict["wandb_id"]
        dist.barrier()


class JSONLogger:
    """
    Simple file-based metric logger.

    Only rank 0 writes to avoid conflicts.
    Appends metrics to list and rewrites entire file for crash durability.
    """

    def __init__(self, log_dir: str, run_name: str):
        """
        Initialize JSON logger.

        Args:
            log_dir: Directory for log files
            run_name: Name for this run (used in filename)
        """
        self.log_dir = log_dir
        self.run_name = run_name
        self.log_file = None
        self.metrics = []

        if dist.get_rank() == 0:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f"{run_name}_metrics.json")
            print0(f"JSON logging to {self.log_file}")

    def log(self, metrics: dict):
        """
        Log metrics to JSON file.

        Only rank 0 writes. Rewrites entire file each time to ensure
        durability (no partial writes if crash occurs).

        Args:
            metrics: Dictionary of metrics to log
        """
        if dist.get_rank() == 0:
            self.metrics.append(metrics)
            # Write the entire list to file each time (ensures we don't lose data)
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)

    def close(self):
        """Finalize the log file and print summary."""
        if dist.get_rank() == 0 and self.metrics:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print0(f"Saved {len(self.metrics)} metric entries to {self.log_file}")


def save_initialization(model: torch.nn.Module, filepath: str):
    """
    Save model initialization to pickle file for reproducible experiments.

    Only rank 0 saves to avoid race conditions.
    All ranks wait at barrier for synchronization.

    Args:
        model: Model whose state_dict to save
        filepath: Path for pickle file
    """
    if dist.get_rank() == 0:
        # Get model state dict
        state_dict = model.state_dict()

        # Save to pickle file
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)
        print0(f"Saved initialization to {filepath}")

    # Wait for rank 0 to finish writing
    dist.barrier()


def load_initialization(model: torch.nn.Module, filepath: str):
    """
    Load model initialization from pickle file.

    All ranks load the same file for consistency.

    Args:
        model: Model to load state_dict into
        filepath: Path to pickle file

    Raises:
        FileNotFoundError: If initialization file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Initialization file not found: {filepath}")

    print0(f"Loading initialization from {filepath}")

    # All ranks load the same file
    with open(filepath, 'rb') as f:
        state_dict = pickle.load(f)

    # Load state dict into model
    model.load_state_dict(state_dict)

    # Ensure all ranks are synchronized
    dist.barrier()
    print0("Successfully loaded initialization")
