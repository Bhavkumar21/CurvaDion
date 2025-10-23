"""
Configuration management for CurvaDion training.

This module handles:
- Hyperparameter dataclass definition
- CLI argument parsing with YAML config support
- Configuration overrides and validation
"""

import argparse
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Hyperparameters:
    """
    Complete training configuration.

    All hyperparameters have defaults that can be overridden via:
    1. YAML config file (--config)
    2. CLI arguments (highest priority)
    """
    # Data directory
    data_dir: str = "data/fineweb10B"

    # Training config
    batch_size: int = 8 * 64  # global batch size (across devices)
    device_batch_size: int = 64  # per-device batch size
    sequence_length: int = 1024  # tokens per sequence
    num_iterations: int = 5000
    warmup_ratio: float = 0.01
    warmdown_ratio: float = 0.2

    # Model config
    model_dim: int = 768
    n_layer: int = 12
    n_head: int = 6

    # Evaluation and logging
    val_loss_every: int = 125
    val_tokens: int = 10485760
    checkpoint_freq: int = 0
    checkpoint_dir: str = None
    wandb_project_name: str = "dion-test"

    # Optimizer
    optimizer: str = "dion"
    scalar_opt: str = "lion"

    # Main optimizer hyperparameters
    lr: float = 0.02
    mu: float = 0.95
    weight_decay: float = 0.01
    rank_fraction: float = 0.125

    # Optimizer specific hyperparameters
    qr_method: str = "rcqr"
    cqr_warmup: float = 0.05
    rcqr_oversample: float = 1.25
    replicate_mesh_grad_sync: bool = False
    mixed_precision: bool = False
    adjust_lr: str = "spectral_norm"  # for Muon only
    rmmc_threshold: float = 0.1  # for CurvaDion only
    sync_every: int = 50  # for diloco_dion only


# Global flag for master process printing
# Set during distributed initialization
MASTER_PROCESS = True


def print0(*args):
    """Print only on rank 0 to avoid duplicate output."""
    if MASTER_PROCESS:
        print(*args)


def parse_cli_args():
    """
    Parse command-line arguments with YAML config file support.

    CLI arguments override YAML config values, which override dataclass defaults.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML file whose keys match train.py flags "
        "(CLI values always override the YAML).",
    )

    # Data and checkpointing
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory that contains fineweb_train_*.bin and fineweb_val_*.bin",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to load and save checkpoints",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps, 0 to disable",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer", type=str, default=None, help="Choice of optimizer algorithm"
    )
    parser.add_argument(
        "--scalar_opt", type=str, help="Optimizer for scalar parameters", default=None
    )
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate")
    parser.add_argument(
        "--adjust_lr",
        type=str,
        default=None,
        help="Adjust learning rate method for Muon",
    )
    parser.add_argument(
        "--inv_rank_fraction",
        type=int,
        default=None,
        help="1/r rank fraction for Dion",
    )
    parser.add_argument(
        "--qr_method", type=str, default=None, choices=["qr", "cqr", "rcqr"]
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Use mixed precision for Dion"
    )
    parser.add_argument(
        "--sync_every", type=int, default=None, help="Sync every N steps for DiLocoDion"
    )
    parser.add_argument(
        "--rmmc_threshold", type=float, default=None, help="RMMC threshold for CurvaDion"
    )

    # Model
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)

    # Training hyperparameters
    parser.add_argument(
        "--num_iterations", type=int, default=None, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Global batch size"
    )
    parser.add_argument("--device_batch_size", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--warmdown_ratio", type=float, default=None)
    parser.add_argument(
        "--val_loss_every", type=int, default=None, help="Validate every N steps"
    )

    # WandB logging
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--wandb_project_name", type=str, default=None, help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_job_name",
        type=str,
        default=None,
        help="Append custom text to wandb job name",
    )

    # JSON logging
    parser.add_argument(
        "--json_log_dir",
        type=str,
        default=None,
        help="Directory to save JSON logs (default: no JSON logging)",
    )

    # Distributed training
    parser.add_argument(
        "--dp_size", type=int, default=None, help="Data Parallel size (no sharding)"
    )
    parser.add_argument(
        "--fs_size", type=int, default=None, help="Fully Sharded Data Parallel size"
    )
    parser.add_argument(
        "--tp_size", type=int, default=None, help="Tensor Parallel size"
    )
    parser.add_argument(
        "--replicate_mesh_grad_sync",
        action="store_true",
        help="Do data-parallel gradient sync inside Dion optimizer",
    )
    parser.add_argument(
        "--fast_fsdp",
        action="store_true",
        help="Optimizer FSDP for speed instead of memory efficiency",
    )

    # Initialization
    parser.add_argument(
        "--save_init",
        type=str,
        default=None,
        help="Path to save model initialization as pickle file"
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Path to load model initialization from pickle file"
    )

    # Debugging
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--curvadion_debug", action="store_true", help="Enable detailed RMMC debug printing for CurvaDion"
    )
    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile for model"
    )
    parser.add_argument(
        "--no_triton", action="store_true", help="Disable Triton kernels"
    )

    cli_args = parser.parse_args()

    # Load YAML config if provided
    if cli_args.config:
        cfg_path = Path(cli_args.config)
        with cfg_path.open("r") as f:
            yaml_cfg = yaml.safe_load(f)

        # Copy any key the user did NOT supply on the CLI
        for k, v in yaml_cfg.items():
            if getattr(cli_args, k, None) is None:
                setattr(cli_args, k, v)

        # Manually handle store_true flags from YAML
        for flag in (
            "mixed_precision",
            "replicate_mesh_grad_sync",
            "fast_fsdp",
            "no_wandb",
            "no_compile",
            "no_triton",
            "debug",
        ):
            if yaml_cfg.get(flag, False):
                setattr(cli_args, flag, True)

    return cli_args


def override_args_from_cli(
    hp: Hyperparameters, cli_args: argparse.Namespace
) -> Hyperparameters:
    """
    Override Hyperparameters dataclass with CLI arguments.

    Args:
        hp: Hyperparameters instance with defaults
        cli_args: Parsed CLI arguments

    Returns:
        Updated Hyperparameters instance
    """
    for key, value in vars(cli_args).items():
        if value is not None:
            if hasattr(hp, key):
                print0(f"Setting hyperparameter {key}={value}")
                setattr(hp, key, value)
    return hp
