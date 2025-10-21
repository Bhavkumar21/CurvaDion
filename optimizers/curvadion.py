"""
CurvaDion: Adaptive Communication Optimizer for DDP
Reduces synchronization overhead by only syncing when momentum changes significantly.

Based on Dion (https://arxiv.org/abs/2504.05295) with adaptive communication scheduling.
Simplified DDP-only implementation.
"""

import math
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


class CurvaDion(Optimizer):
    """
    CurvaDion: Adaptive Communication Optimizer for DDP Training

    Reduces communication by only performing full gradient synchronization when
    momentum changes significantly (measured by RMMC metric).

    RMMC_ℓ(t) = |‖M_ℓ(t)‖ - ‖M_ℓ(t-1)‖| / ‖M_ℓ(t-1)‖

    Communication per step:
    - Stable training (RMMC < threshold): 1 boolean flag all-reduce
    - Unstable training (RMMC > threshold): 1 flag + gradient all-reduces

    Args:
        params: Parameters for the optimizer
        process_group: ProcessGroup for DDP communication
        rmmc_threshold: Threshold for maximum RMMC to trigger full sync (default: 0.1)
        track_metrics: Whether to track sync rate and RMMC metrics (default: True)
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
        rmmc_threshold: float = 0.1,
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
        # Validate rmmc_threshold
        if rmmc_threshold <= 0 or rmmc_threshold > 1.0:
            raise ValueError(f"rmmc_threshold must be in (0, 1], got {rmmc_threshold}")

        # Store config
        self.rmmc_threshold = rmmc_threshold
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
                "max_rmmc_history": [],
                "bytes_transferred": 0,  # Total bytes transferred
                "bytes_per_step": [],    # Bytes transferred per step
            }

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with adaptive communication.

        Steps:
        1. Add gradients to momentum (local, no comm)
        2. Calculate RMMC by comparing new momentum to previous norms
        3. All-reduce boolean flag: does any GPU need sync? (4 bytes)
        4. If yes: sync gradients and apply dion update
           If no: apply local dion update (no sync)
        5. Store current momentum norms for next iteration
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Track total steps
        step_bytes = 0
        if self.track_metrics:
            self.metrics["total_steps"] += 1

        # Step 1: Calculate RMMC based on what momentum norm WILL BE after adding gradients
        local_needs_sync, max_rmmc = self._calculate_rmmc_and_check_threshold()

        if self.debug and self._rank == 0:
            print(f"[Rank {self._rank}] max_rmmc={max_rmmc:.6f}, threshold={self.rmmc_threshold:.6f}, local_needs_sync={local_needs_sync}")

        # Step 3: All-reduce boolean flag (4 bytes communication)
        if self._world_size > 1:
            # Create tensor for all-reduce (float32 = 4 bytes)
            local_flag = torch.tensor([1.0 if local_needs_sync else 0.0], device='cuda')
            dist.all_reduce(local_flag, op=dist.ReduceOp.MAX, group=self._process_group)
            global_needs_sync = local_flag.item() > 0.5
            step_bytes += 4  # 4 bytes for float32
        else:
            global_needs_sync = local_needs_sync

        if self.debug and self._rank == 0:
            print(f"[Rank {self._rank}] global_needs_sync={global_needs_sync}")

        # Track metrics
        if self.track_metrics:
            if global_needs_sync:
                self.metrics["sync_steps"] += 1
            self.metrics["max_rmmc_history"].append(max_rmmc)

        # Step 4: Apply optimizer update (with or without gradient sync)
        if global_needs_sync:
            grad_bytes = self._apply_update_with_sync()
            step_bytes += grad_bytes
        else:
            self._apply_update_local()

        # Step 5: Store current momentum norms for next iteration
        self._update_momentum_norms()

        # Track bytes transferred this step
        if self.track_metrics:
            self.metrics["bytes_transferred"] += step_bytes
            self.metrics["bytes_per_step"].append(step_bytes)

        return loss

    def _calculate_rmmc_and_check_threshold(self) -> Tuple[bool, float]:
        """
        Calculate RMMC for each layer based on what momentum norm WILL BE after adding gradients.
        We don't actually modify momentum here - that happens in the update functions.

        RMMC_ℓ(t) = |‖M_ℓ(t-1) + G_ℓ(t)‖ - ‖M_ℓ(t-1)‖| / ‖M_ℓ(t-1)‖

        Returns:
            (needs_sync, max_rmmc): Whether this GPU needs sync and the max RMMC value
        """
        max_rmmc = 0.0
        max_rmmc_layer = None  # Track which layer has max RMMC
        layer_rmmcs = []  # For debugging
        MIN_MOMENTUM_FOR_RMMC = 0.1  # Only calculate RMMC if momentum is reasonably large

        for group in self.param_groups:
            algo = group.get("algorithm", "dion")
            for param in group["params"]:
                if param.grad is None:
                    continue

                # Get or initialize state
                state = self.state[param]
                if "momentum" not in state:
                    self._initialize_state(param, state, group)

                # Get current momentum and gradient
                momentum = state["momentum"]
                gradient = param.grad.to(momentum.dtype)

                # Calculate what the NEW momentum norm will be after adding gradient
                # new_momentum = momentum + gradient
                new_momentum_norm = (momentum + gradient).norm().item()

                # Get previous norm (stored from last iteration)
                prev_norm = state.get("prev_momentum_norm", 0.0)

                # Calculate RMMC: |‖M(t)‖ - ‖M(t-1)‖| / ‖M(t-1)‖
                # Use a minimum threshold to avoid explosion when momentum is very small
                if prev_norm > MIN_MOMENTUM_FOR_RMMC:
                    rmmc = abs(new_momentum_norm - prev_norm) / prev_norm
                    if rmmc > max_rmmc:
                        max_rmmc = rmmc
                        max_rmmc_layer = (param.shape, algo, rmmc, prev_norm, new_momentum_norm)
                    if self.debug:
                        layer_rmmcs.append((param.shape, algo, rmmc, prev_norm, new_momentum_norm))
                elif self.debug and prev_norm > 1e-8:
                    # Still track for debugging, but mark as "ignored"
                    rmmc = abs(new_momentum_norm - prev_norm) / prev_norm
                    layer_rmmcs.append((param.shape, algo + " (IGNORED)", rmmc, prev_norm, new_momentum_norm))

        if self.debug and self._rank == 0:
            if layer_rmmcs:
                print(f"[Rank {self._rank}] Sample Layer RMMCs (first 5):")
                for shape, algo, rmmc, prev_norm, new_norm in layer_rmmcs[:5]:
                    print(f"  Shape {shape}, algo={algo}: RMMC={rmmc:.6f}, prev_norm={prev_norm:.4f}, new_norm={new_norm:.4f}")

            if max_rmmc_layer:
                shape, algo, rmmc, prev_norm, new_norm = max_rmmc_layer
                print(f"[Rank {self._rank}] MAX RMMC Layer (used for sync decision):")
                print(f"  Shape {shape}, algo={algo}: RMMC={rmmc:.6f}, prev_norm={prev_norm:.4f}, new_norm={new_norm:.4f}")
            else:
                print(f"[Rank {self._rank}] No layers with prev_norm > {MIN_MOMENTUM_FOR_RMMC} yet (early training)")

        needs_sync = max_rmmc > self.rmmc_threshold
        return needs_sync, max_rmmc

    def _update_momentum_norms(self):
        """Store current momentum norms for next iteration's RMMC calculation."""
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if "momentum" in state:
                    current_norm = state["momentum"].norm().item()
                    state["prev_momentum_norm"] = current_norm

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

            # Initialize prev norm to current (zero)
            state["prev_momentum_norm"] = 0.0

        elif algo == "adamw":
            var_dtype = self._mixed_precision_config.variance_dtype or param.dtype
            state["momentum"] = torch.zeros_like(param)
            state["variance"] = torch.zeros_like(param, dtype=var_dtype)
            state["prev_momentum_norm"] = 0.0

        elif algo == "lion":
            state["momentum"] = torch.zeros_like(param)
            state["prev_momentum_norm"] = 0.0

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
            - sync_rate: Fraction of steps that triggered full sync
            - total_steps: Total optimization steps
            - sync_steps: Number of steps with full sync
            - avg_max_rmmc: Average maximum RMMC value
            - recent_max_rmmc: Most recent RMMC value
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
            "avg_max_rmmc": sum(self.metrics["max_rmmc_history"]) / max(len(self.metrics["max_rmmc_history"]), 1),
            "recent_max_rmmc": self.metrics["max_rmmc_history"][-1] if self.metrics["max_rmmc_history"] else 0.0,
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
                "max_rmmc_history": [],
                "bytes_transferred": 0,
                "bytes_per_step": [],
            }
