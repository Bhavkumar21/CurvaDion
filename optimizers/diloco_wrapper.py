"""
This wrapper performs fixed-interval parameter synchronization without outer momentum:
- Inner optimization: H local steps with inner optimizer (e.g., Dion, AdamW)
- Outer synchronization: Simple parameter averaging across workers (no momentum)

"""

import torch
import torch.distributed as dist
from typing import Optional, Union, List
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.optim.optimizer import Optimizer


class DiLoCoWrapper(Optimizer):
    """
    Periodic synchronization wrapper for distributed training (Simplified DiLoCo).

    Args:
        inner_optimizer: The optimizer to use for inner steps (e.g., Dion)
        model: The model being trained
        replicate_mesh: DeviceMesh or ProcessGroup for all-reduce across workers
        num_inner_steps: Number of inner optimization steps (H) before synchronization
    """

    def __init__(
        self,
        inner_optimizer,
        model: torch.nn.Module,
        replicate_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        num_inner_steps: int = 50,
    ):
        self.inner_optimizer = inner_optimizer
        self.model = model
        self.replicate_mesh = replicate_mesh
        self.num_inner_steps = num_inner_steps

        # Initialize parent Optimizer class with inner optimizer's param_groups
        defaults = {}
        super().__init__(inner_optimizer.param_groups, defaults)

        # Get world size and rank
        if isinstance(replicate_mesh, DeviceMesh):
            self._world_size = replicate_mesh.size()
            self._rank = replicate_mesh.get_local_rank()
        elif isinstance(replicate_mesh, ProcessGroup):
            self._world_size = dist.get_world_size(replicate_mesh)
            self._rank = dist.get_rank(replicate_mesh)
        elif replicate_mesh is None:
            self._world_size = 1
            self._rank = 0
        else:
            raise TypeError(f"Invalid replicate mesh type: {type(replicate_mesh)}.")

        # Track synchronization and inner steps
        self.sync_step = 0
        self.inner_step_counter = 0

        # Track if we just did a sync step
        self.just_did_sync_step = False

        # Diagnostic: Print initialization info
        if self._rank == 0:
            print(f"[DiLoCoWrapper] Initialized with world_size={self._world_size}, H={num_inner_steps}")
            print(f"[DiLoCoWrapper] Replicate mesh type: {type(replicate_mesh)}")
            print(f"[DiLoCoWrapper] Will sync parameters every {num_inner_steps} steps")

        self._initial_param_norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._initial_param_norms[name] = param.data.norm().item()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one optimization step.

        This handles the inner/sync loop logic:
        - Inner steps (0 to H-1): Call inner optimizer (local only)
        - Sync step (at H): Average parameters across workers
        """
        self.just_did_sync_step = False

        if self.inner_step_counter == 0 and self.sync_step > 0:
            # Just after a sync - all ranks should have identical parameters
            param_checksum = self._compute_param_checksum()
            if self._rank == 0:
                print(f"[DiLoCoWrapper] Post-sync param checksum (all ranks should match): {param_checksum:.6f}")

        # Inner optimizer step (local-only, no communication)
        loss = self.inner_optimizer.step(closure)
        self.inner_step_counter += 1

        if self.inner_step_counter == self.num_inner_steps - 1:
            param_checksum = self._compute_param_checksum()
            if self._rank == 0:
                print(f"[DiLoCoWrapper] Pre-sync param checksum on rank 0: {param_checksum:.6f}")
            if self._rank == 1:
                print(f"[DiLoCoWrapper] Pre-sync param checksum on rank 1: {param_checksum:.6f}")

        # End of inner cycle: perform synchronization
        if self.inner_step_counter >= self.num_inner_steps:
            self._sync_step()
            self.inner_step_counter = 0
            self.sync_step += 1
            self.just_did_sync_step = True

        return loss

    def _compute_param_checksum(self) -> float:
        checksum = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                checksum += param.data.sum().item()
        return checksum

    @torch.no_grad()
    def _sync_step(self):
        """
        Perform synchronization step: simply average parameters across all workers.

        This is a straightforward parameter averaging with no momentum or pseudo-gradients.
        Each worker has independently optimized for H steps, and we now synchronize by
        averaging the model parameters across all workers.
        """
        if self._rank == 0:
            total_param_norm = 0.0
            for param in self.model.parameters():
                if param.requires_grad:
                    total_param_norm += param.data.norm().item() ** 2
            total_param_norm = total_param_norm ** 0.5
            print(f"[DiLoCoWrapper] Step {self.sync_step}: Syncing parameters (H={self.num_inner_steps})")
            print(f"[DiLoCoWrapper]   Pre-sync param norm: {total_param_norm:.6f}")

        pre_sync_norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                pre_sync_norms[name] = param.data.norm().item()

        # Synchronize parameters across all workers
        param_count = 0
        for param in self.model.parameters():
            if not param.requires_grad:
                continue

            # All-reduce current params to get average across workers
            if self._world_size > 1:
                if isinstance(self.replicate_mesh, DeviceMesh):
                    dist.all_reduce(
                        param.data,
                        op=dist.ReduceOp.AVG,
                        group=self.replicate_mesh.get_group()
                    )
                elif isinstance(self.replicate_mesh, ProcessGroup):
                    dist.all_reduce(
                        param.data,
                        op=dist.ReduceOp.AVG,
                        group=self.replicate_mesh
                    )
                param_count += 1
            else:
                # Single worker - no sync needed but this is a warning sign!
                if self._rank == 0 and self.sync_step == 0:
                    print(f"[DiLoCoWrapper] WARNING: world_size=1, no synchronization will occur!")

        if self._rank == 0:
            total_param_norm_post = 0.0
            for param in self.model.parameters():
                if param.requires_grad:
                    total_param_norm_post += param.data.norm().item() ** 2
            total_param_norm_post = total_param_norm_post ** 0.5
            print(f"[DiLoCoWrapper]   Post-sync param norm: {total_param_norm_post:.6f}")
            print(f"[DiLoCoWrapper]   Synchronized {param_count} parameter tensors")

    def zero_grad(self, set_to_none: bool = False):
        """Forward zero_grad to inner optimizer."""
        self.inner_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """
        Return state dict including both inner optimizer and wrapper state.
        """
        return {
            'inner_optimizer': self.inner_optimizer.state_dict(),
            'sync_step': self.sync_step,
            'inner_step_counter': self.inner_step_counter,
        }

    def load_state_dict(self, state_dict):
        """
        Load state dict.
        """
        self.inner_optimizer.load_state_dict(state_dict['inner_optimizer'])
        self.sync_step = state_dict['sync_step']
        self.inner_step_counter = state_dict['inner_step_counter']

    def get_sync_step(self) -> int:
        """Get current sync step number."""
        return self.sync_step

    def get_inner_step_in_cycle(self) -> int:
        """Get current position within inner loop (0 to H-1)."""
        return self.inner_step_counter

    def is_sync_step(self) -> bool:
        """Check if the last step() call resulted in a sync step."""
        return self.just_did_sync_step

    def get_metrics(self):
        """Get synchronization metrics for logging."""
        return {
            "diloco/sync_step": self.sync_step,
            "diloco/inner_step_counter": self.inner_step_counter,
            "diloco/just_synced": 1 if self.just_did_sync_step else 0,
            "diloco/world_size": self._world_size,
        }
