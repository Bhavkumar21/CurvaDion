"""
CurvaDion training utilities.

This package provides:
- Configuration management (config.py)
- Training infrastructure (training_utils.py)
"""

from utils.config import (
    Hyperparameters,
    MASTER_PROCESS,
    print0,
    parse_cli_args,
    override_args_from_cli,
)

from utils.training_utils import (
    init_distributed,
    init_optimizer,
    get_lr_schedule_fn,
    CheckpointManager,
    JSONLogger,
    save_initialization,
    load_initialization,
)

__all__ = [
    # Configuration
    "Hyperparameters",
    "MASTER_PROCESS",
    "print0",
    "parse_cli_args",
    "override_args_from_cli",
    # Training infrastructure
    "init_distributed",
    "init_optimizer",
    "get_lr_schedule_fn",
    "CheckpointManager",
    "JSONLogger",
    "save_initialization",
    "load_initialization",
]
