"""
DiLocoDion: Scheduled Communication Optimizer for DDP

Performs full gradient synchronization every H steps (configurable), with local-only
updates in between. This serves as a fixed-schedule baseline to compare against
CurvaDion's adaptive RMMC-based synchronization.

Inspired by DiLoCo (Distributed Low-Communication) approach with periodic synchronization.
"""

import torch
import torch.distributed as dist
from dataclasses import dataclass
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Any, Dict, List, Optional, Tuple

from .dion_simple import dion_update as dion_simple_update
from .scalar_opts import adamw_update, lion_update


@dataclass
class DionMixedPrecisionConfig:
    """Mixed precision configuration for optimizer states."""
    momentum_dtype: Optional[torch.dtype] = None
    Q_dtype: Optional[torch.dtype] = None
    variance_dtype: Optional[torch.dtype] = None


class DiLocoDion(Optimizer):
    """
    DiLocoDion: Scheduled Communication Optimizer for DDP Training

    Reduces communication by only performing full gradient synchronization every
    H training iterations, with local-only updates in between.

    Schedule: Every sync_every steps → full sync
              Other steps → local updates only

    Args:
        params: Parameters for the optimizer
        process_group: ProcessGroup for DDP communication
        sync_every: Number of steps between full gradient synchronizations (default: 50)
        track_metrics: Whether to track sync rate and communication metrics (default: True)
        lr: Base learning rate
        mu: Momentum factor for Dion
        rank_fraction: r/d fraction for low-rank approximation in Dion
        weight_decay: Weight decay factor
        betas: Betas for AdamW/Lion scalar optimizers
        epsilon: Small constant for numerical stability
        mixed_precision_config: Mixed precision configuration for optimizer states

    Note: Designed specifically for DDP with replicate_mesh_grad_sync=True.
    """

    def __init__(
        self,
        params: ParamsT,
        process_group: Optional[ProcessGroup] = None,
        sync_every: int = 50,
        track_metrics: bool = True,
        debug: bool = False,
        lr: float = 0.02,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        rank_fraction: float = 0.125,
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
    ):
        # Validate sync_every
        if sync_every < 1:
            raise ValueError(f"sync_every must be >= 1, got {sync_every}")

        # Store config
        self.sync_every = sync_every
        self.track_metrics = track_metrics
        self.debug = debug
        self._process_group = process_group
        self._mixed_precision_config = mixed_precision_config or DionMixedPrecisionConfig()

        # Get world size and rank
        if process_group is not None:
            self._world_size = dist.get_world_size(process_group)
            self._rank = dist.get_rank(process_group)
        else:
            self._world_size = 1
            self._rank = 0

        # Step counter for scheduling
        self._step_count = 0

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            rank_fraction=rank_fraction,
            weight_decay=weight_decay,
            epsilon=epsilon,
            algorithm="dion",  # Default algorithm
            step=0,
        )
        super().__init__(params, defaults)

        # Metrics tracking
        if self.track_metrics:
            self.metrics = {
                "total_steps": 0,
                "sync_steps": 0,
                "bytes_transferred": 0,  # Total bytes transferred
                "bytes_per_step": [],    # Bytes transferred per step
            }

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with scheduled communication.

        Steps:
        1. Increment step counter
        2. Check if current step is a sync step (step_count % sync_every == 0)
        3. If yes: sync gradients and apply Dion update
           If no: apply local Dion update (no sync)
        4. Track bytes communicated
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Increment step counter
        self._step_count += 1

        # Track total steps
        step_bytes = 0
        if self.track_metrics:
            self.metrics["total_steps"] += 1

        # Determine if this is a sync step
        is_sync_step = (self._step_count % self.sync_every) == 0

        if self.debug and self._rank == 0:
            print(f"[Rank {self._rank}] Step {self._step_count}: is_sync_step={is_sync_step} (sync_every={self.sync_every})")

        # Track metrics
        if self.track_metrics and is_sync_step:
            self.metrics["sync_steps"] += 1

        # Apply optimizer update (with or without gradient sync)
        if is_sync_step:
            grad_bytes = self._apply_update_with_sync()
            step_bytes += grad_bytes
        else:
            self._apply_update_local()

        # Track bytes transferred this step
        if self.track_metrics:
            self.metrics["bytes_transferred"] += step_bytes
            self.metrics["bytes_per_step"].append(step_bytes)

        return loss

    def _initialize_state(self, param: Tensor, state: Dict[str, Any], group: dict):
        """Initialize optimizer state for a parameter."""
        algo = group["algorithm"]

        # Initialize based on algorithm
        if algo == "dion":
            if param.ndim != 2:
                raise ValueError(
                    f"Expected Dion parameters to be 2D matrix, but got {param.ndim}D. "
                    f"For scalar parameters, set 'algorithm' to 'lion' or 'adamw'."
                )

            # Initialize momentum
            mom_dtype = self._mixed_precision_config.momentum_dtype or param.dtype
            state["momentum"] = torch.zeros_like(param, dtype=mom_dtype)

            # Initialize Q matrix for dion_simple
            q_dtype = self._mixed_precision_config.Q_dtype or param.dtype
            rank = int(group["rank_fraction"] * min(param.shape))
            state["Q"] = torch.randn((param.size(1), rank), device=param.device, dtype=q_dtype)

        elif algo == "adamw":
            var_dtype = self._mixed_precision_config.variance_dtype or param.dtype
            state["momentum"] = torch.zeros_like(param)
            state["variance"] = torch.zeros_like(param, dtype=var_dtype)

        elif algo == "lion":
            state["momentum"] = torch.zeros_like(param)

    def _apply_update_with_sync(self) -> int:
        """
        Sync gradients and apply Dion update.
        Returns number of bytes communicated for gradient sync.
        """
        total_bytes = 0

        for group in self.param_groups:
            algo = group["algorithm"]
            group["step"] += 1
            step = group["step"]

            # Wrap hyperparameters in tensors
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])

            if algo == "dion":
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    state = self.state[param]
                    if not state:
                        self._initialize_state(param, state, group)

                    # All-reduce gradients
                    if self._world_size > 1:
                        # Track bytes: each element is 4 bytes (float32) or 2 bytes (bfloat16/float16)
                        bytes_per_element = param.grad.element_size()
                        num_elements = param.grad.numel()
                        total_bytes += bytes_per_element * num_elements

                        dist.all_reduce(param.grad, group=self._process_group)
                        param.grad.div_(self._world_size)

                    # Run dion_simple update with synchronized gradients
                    dion_simple_update(
                        X=param,
                        G=param.grad,
                        M=state["momentum"],
                        Q=state["Q"],
                        lr=lr,
                        mu=mu,
                        weight_decay=weight_decay,
                        epsilon=epsilon,
                    )

            elif algo == "adamw":
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    state = self.state[param]
                    if not state:
                        self._initialize_state(param, state, group)

                    # Sync gradients
                    if self._world_size > 1:
                        bytes_per_element = param.grad.element_size()
                        num_elements = param.grad.numel()
                        total_bytes += bytes_per_element * num_elements

                        dist.all_reduce(param.grad, group=self._process_group)
                        param.grad.div_(self._world_size)

                    adamw_update(
                        X=param,
                        G=param.grad,
                        M=state["momentum"],
                        V=state["variance"],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        step=step,
                        epsilon=epsilon.item(),
                    )

            elif algo == "lion":
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    state = self.state[param]
                    if not state:
                        self._initialize_state(param, state, group)

                    # Sync gradients
                    if self._world_size > 1:
                        bytes_per_element = param.grad.element_size()
                        num_elements = param.grad.numel()
                        total_bytes += bytes_per_element * num_elements

                        dist.all_reduce(param.grad, group=self._process_group)
                        param.grad.div_(self._world_size)

                    lion_update(
                        X=param,
                        G=param.grad,
                        M=state["momentum"],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                    )

        return total_bytes

    def _apply_update_local(self):
        """
        Apply Dion update locally with no gradient synchronization.
        Models will diverge slightly but reconverge at next full sync.
        """
        for group in self.param_groups:
            algo = group["algorithm"]
            group["step"] += 1
            step = group["step"]

            # Wrap hyperparameters in tensors
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])

            if algo == "dion":
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    state = self.state[param]
                    if not state:
                        self._initialize_state(param, state, group)

                    # Local update only, no communication
                    dion_simple_update(
                        X=param,
                        G=param.grad,
                        M=state["momentum"],
                        Q=state["Q"],
                        lr=lr,
                        mu=mu,
                        weight_decay=weight_decay,
                        epsilon=epsilon,
                    )

            elif algo == "adamw":
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    state = self.state[param]
                    if not state:
                        self._initialize_state(param, state, group)

                    adamw_update(
                        X=param,
                        G=param.grad,
                        M=state["momentum"],
                        V=state["variance"],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        step=step,
                        epsilon=epsilon.item(),
                    )

            elif algo == "lion":
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    state = self.state[param]
                    if not state:
                        self._initialize_state(param, state, group)

                    lion_update(
                        X=param,
                        G=param.grad,
                        M=state["momentum"],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                    )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get tracked metrics for logging (local to this GPU, no communication).

        Returns:
            Dictionary containing:
            - sync_rate: Fraction of steps that triggered full sync (should be ~1/sync_every)
            - total_steps: Total optimization steps
            - sync_steps: Number of steps with full sync
            - total_bytes_transferred: Total bytes communicated
            - recent_bytes: Bytes transferred in last step
            - avg_bytes_per_step: Average bytes per step
        """
        if not self.track_metrics:
            return {}

        total = self.metrics["total_steps"]
        synced = self.metrics["sync_steps"]

        return {
            "sync_rate": synced / max(total, 1),
            "total_steps": total,
            "sync_steps": synced,
            "total_bytes_transferred": self.metrics["bytes_transferred"],
            "recent_bytes": self.metrics["bytes_per_step"][-1] if self.metrics["bytes_per_step"] else 0,
            "avg_bytes_per_step": self.metrics["bytes_transferred"] / max(total, 1),
        }

    def reset_metrics(self):
        """Reset metrics tracking."""
        if self.track_metrics:
            self.metrics = {
                "total_steps": 0,
                "sync_steps": 0,
                "bytes_transferred": 0,
                "bytes_per_step": [],
            }
