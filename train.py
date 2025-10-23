"""
Main training script for CurvaDion.
"""

import math
import os
import time
import torch
import torch.distributed as dist
import wandb

from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from models.gpt_model import GPT, GPTConfig
from models.gpt_utils import DistributedDataLoader
from optimizers import CurvaDion

from utils import (
    Hyperparameters,
    print0,
    parse_cli_args,
    override_args_from_cli,
    init_distributed,
    init_optimizer,
    get_lr_schedule_fn,
    CheckpointManager,
    JSONLogger,
    save_initialization,
    load_initialization,
)


def main():
    """Main training function."""
    torch._dynamo.config.cache_size_limit = 100

    # --- Parse command line arguments and set hyperparams ---
    cli_args = parse_cli_args()
    hp = Hyperparameters()
    hp = override_args_from_cli(hp, cli_args)

    if cli_args.inv_rank_fraction:
        hp.rank_fraction = 1.0 / cli_args.inv_rank_fraction

    # Only check for checkpoint_dir if we're actually going to train (not just saving init)
    if hp.checkpoint_freq > 0 and not cli_args.save_init:
        if not hp.checkpoint_dir:
            raise ValueError("Must specify --checkpoint_dir to save checkpoints")

    # --- Distributed training initialization ---
    device_mesh = init_distributed(
        dp_size=cli_args.dp_size,
        fs_size=cli_args.fs_size,
        tp_size=cli_args.tp_size,
    )
    print0("=" * 80)

    # --- DataLoader Setup ---
    if device_mesh is not None:
        # Combine replicated and sharded data parallel meshes for data loading
        data_parallel_mesh = device_mesh["dp", "fs"]._flatten()
        data_parallel_size = data_parallel_mesh.size()
        data_parallel_rank = data_parallel_mesh.get_local_rank()
    else:
        # We are using DDP with one global process group
        data_parallel_mesh = None
        data_parallel_size = dist.get_world_size()
        data_parallel_rank = dist.get_rank()

    if cli_args.debug:
        # in debug mode, make batch size very small
        hp.batch_size = 2 * data_parallel_size
        hp.device_batch_size = 1

    # Calculate validation steps
    tokens_in_global_batch = (
        hp.device_batch_size * hp.sequence_length * data_parallel_size
    )
    assert hp.val_tokens % tokens_in_global_batch == 0, "Invalid val_tokens"
    val_steps = hp.val_tokens // tokens_in_global_batch

    if cli_args.debug:
        # train for just a few steps
        hp.num_iterations = 20
        val_steps = min(val_steps, 2)

    # Calculate gradient accumulation steps
    sequences_in_global_batch = hp.device_batch_size * data_parallel_size
    assert hp.batch_size % sequences_in_global_batch == 0, "Invalid batch_size"
    grad_accum_steps = hp.batch_size // sequences_in_global_batch
    assert grad_accum_steps >= 1, "Invalid grad_accum_steps"

    print0(f"Global batch size: {hp.batch_size} sequences")
    print0(f"Per-device batch size: {hp.device_batch_size} sequences")
    print0(f"Sequence length: {hp.sequence_length} tokens")
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0("=" * 80)

    train_glob = os.path.join(hp.data_dir, "fineweb_train_*.bin")
    val_glob = os.path.join(hp.data_dir, "fineweb_val_*.bin")

    print0(f"Training data: {train_glob}")
    print0(f"Validation data: {val_glob}")

    # Each data parallel rank gets different data
    # TP ranks must all use identical data
    train_loader = DistributedDataLoader(
        train_glob,
        hp.device_batch_size,
        hp.sequence_length,
        data_parallel_rank,
        data_parallel_size,
    )
    val_loader = DistributedDataLoader(
        val_glob,
        hp.device_batch_size,
        hp.sequence_length,
        data_parallel_rank,
        data_parallel_size,
    )

    print0(f"Training DataLoader: {len(train_loader.files)} files")
    print0(f"Validation DataLoader: {len(val_loader.files)} files")
    print0("=" * 80)

    # --- Model Initialization ---
    print0(f"Model dimension: {hp.model_dim}")
    print0(f"Number of layers: {hp.n_layer}")
    print0(f"Number of heads: {hp.n_head}")

    num_vocab = 50304  # nearest multiple of 128 for efficiency
    gpt_config = GPTConfig(
        sequence_len=hp.sequence_length,
        vocab_size=num_vocab,
        n_layer=hp.n_layer,
        n_head=hp.n_head,
        n_embd=hp.model_dim,
    )
    with torch.device("meta"):
        model = GPT(gpt_config)

    # Shard the model if using a device mesh
    # If replicate_mesh_grad_sync is True, FSDP will not handle data-parallel gradient sync
    # If replicate_mesh_grad_sync is False, we use Pytorch HSDP to do data-parallel gradient sync
    if device_mesh is not None:
        from models.gpt_model import parallelize_gpt_model
        parallelize_gpt_model(
            model,
            device_mesh=device_mesh,
            dp_name=(None if hp.replicate_mesh_grad_sync else "dp"),
            fs_name="fs",
            tp_name="tp",
            fsdp_reshard_after_forward=(not cli_args.fast_fsdp),
        )
        raw_model = model

    # Move model to GPU and initialize weights
    model.to_empty(device="cuda")
    model.init_weights()

    # --- Handle initialization load BEFORE wrapping/compilation ---
    # IMPORTANT: Load initialization before DDP wrapping to ensure all ranks start with identical weights
    # If we load after DDP init, DDP will broadcast rank 0's random weights first, defeating the purpose
    if cli_args.init_from:
        load_initialization(model if device_mesh is not None else model, cli_args.init_from)
        # Broadcast parameters from rank 0 to ensure perfect synchronization across all ranks
        if device_mesh is None:
            # For DDP, broadcast all parameters from rank 0
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        print0("Initialization loaded and synchronized across all ranks")

    # --- Handle initialization save (if requested) ---
    # Save initialization before compilation and DDP wrapping to get clean weights
    if cli_args.save_init:
        save_initialization(model if device_mesh is not None else model, cli_args.save_init)
        print0(f"Initialization saved. Exiting as this was just for saving init.")
        dist.destroy_process_group()
        return

    # Compile model after initialization is loaded
    if not cli_args.no_compile:
        model.compile()

    # If no device mesh, we are using DDP
    if device_mesh is None:
        # CRITICAL: For DiLoCo with replicate_mesh_grad_sync=True, DO NOT use DDP!
        # DiLoCo handles ALL synchronization through the optimizer.
        # DDP interferes with this by synchronizing gradients, preventing divergence.
        use_ddp = not (hp.optimizer == "diloco_dion" and hp.replicate_mesh_grad_sync)

        if use_ddp:
            # Use LOCAL_RANK here (per-node GPU index)
            local_rank = int(os.environ["LOCAL_RANK"])
            # Wrap in DDP with broadcast_buffers=False to prevent unintended synchronization
            # This is critical for CurvaDion to control synchronization schedule
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,  # Prevent automatic buffer sync during forward
            )
            raw_model = model.module  # the underlying model
            print0("Using DDP for gradient synchronization")
        else:
            # No DDP - optimizer handles all synchronization
            raw_model = model
            print0(f"NOT using DDP - {hp.optimizer} optimizer handles all synchronization")

    # Ensure parameters are contiguous
    for i, p in enumerate(model.parameters()):
        if not p.is_contiguous():
            raise ValueError(f"Parameter {i} is not contiguous")

    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Total parameters: {num_params}")
    print0(f"Using torch.compile: {not cli_args.no_compile}")

    # Print model architecture
    print0(model)
    print0("=" * 80)

    # --- Optimizer Setup ---
    print0(f"Optimizer: {hp.optimizer}")
    print0(f"Scalar optimizer: {hp.scalar_opt}")
    print0(f"Base learning rate: {hp.lr}")

    # For optimizer initialization, we need the process group
    # If using DDP, get it from DDP model; otherwise get the default process group
    if isinstance(model, DDP):
        ddp_model_for_opt = model
    elif device_mesh is None:
        # No device mesh and not using DDP - create a reference that has process_group attribute
        # This is for diloco_dion which needs process_group but doesn't use DDP
        class _ProcessGroupHolder:
            def __init__(self):
                self.process_group = dist.group.WORLD
        ddp_model_for_opt = _ProcessGroupHolder()
    else:
        ddp_model_for_opt = None

    optimizer = init_optimizer(
        model=raw_model,
        device_mesh=device_mesh,
        ddp_model=ddp_model_for_opt,
        hp=hp,
        cli_args=cli_args,
    )

    # Learning rate scheduler
    get_lr = get_lr_schedule_fn(hp)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    print0("=" * 80)

    # --- Logging initialization ---
    # Create a name to identify this run
    run_name = f"({hp.optimizer}+{hp.scalar_opt})"
    if "dion" in hp.optimizer:
        run_name += f"_rank={hp.rank_fraction}"
    if cli_args.dp_size is not None:
        run_name += f"_dp={cli_args.dp_size}_fs={cli_args.fs_size}_tp={cli_args.tp_size}_gradsync={cli_args.replicate_mesh_grad_sync}"
    if cli_args.wandb_job_name:
        run_name += f"_{cli_args.wandb_job_name}"

    # --- Set up checkpointing ---
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=hp.checkpoint_dir,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_id=None,
    )

    print0(f"Run name: {run_name}")
    print0(f"Debug mode: {cli_args.debug}")
    print0(f"Checkpoint directory: {hp.checkpoint_dir}")
    print0(
        f"Checkpoint frequency: {hp.checkpoint_freq if hp.checkpoint_freq > 0 else 'disabled'}"
    )

    # Load the latest checkpoint if it exists
    if hp.checkpoint_dir:
        checkpoint_manager.load(allow_missing=True)
        if checkpoint_manager.step is not None:
            print0(f"Resuming from step {checkpoint_manager.step}")
        else:
            print0("No previous checkpoint found, training model from scratch")
    else:
        # No checkpoint path provided
        print0("Training model from scratch")

    print0("=" * 80)

    # --- WandB initialization ---
    if not cli_args.no_wandb and not cli_args.debug:
        assert hp.wandb_project_name, "wandb project name is required"
        import utils.config
        if utils.config.MASTER_PROCESS:
            # Check if we already have a wandb ID from the checkpoint
            wandb_id = checkpoint_manager.wandb_id
            resume = "must" if wandb_id else "never"
            wandb.login(
                key=os.environ.get("WANDB_API_KEY"),
                host=os.environ.get("WANDB_HOST"),
                timeout=0,
            )
            wandb.init(
                project=hp.wandb_project_name,
                name=run_name,
                config=hp.__dict__,
                id=wandb_id,
                resume=resume,
            )
            # If we got a new ID, update the checkpoint manager
            checkpoint_manager.wandb_id = wandb.run.id

        # Broadcast wandb_id to all processes
        # Do this to ensure consistency of distributed checkpoint
        obj_list = [checkpoint_manager.wandb_id]
        dist.broadcast_object_list(obj_list, src=0)
        checkpoint_manager.wandb_id = obj_list[0]

    # --- JSON Logger initialization ---
    json_logger = None
    if cli_args.json_log_dir:
        json_logger = JSONLogger(cli_args.json_log_dir, run_name)

    # --- Training Loop ---
    x, y = train_loader.next_batch()
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.time()

    # Use autocast for mixed precision
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    import utils.config
    start_step = 0 if checkpoint_manager.step is None else checkpoint_manager.step + 1
    pbar = tqdm(total=hp.num_iterations, desc="Training", disable=not utils.config.MASTER_PROCESS)
    pbar.update(start_step)

    for step in range(start_step, hp.num_iterations + 1):
        # Skip the first few steps for timing to avoid torch.compile overhead
        if step == 10:
            training_time_ms = 0
            torch.cuda.synchronize()
            t0 = time.time()
        timed_steps = (step - 10) if step > 10 else float("nan")

        # --- Validation ---
        last_step = step == hp.num_iterations
        if last_step or (hp.val_loss_every > 0 and step % hp.val_loss_every == 0):
            # Measure elapsed time for training
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)

            # Run validation
            model.eval()
            val_loader.reset()
            val_loss = torch.tensor(0.0, device=x.device)
            for _ in range(val_steps):
                with torch.no_grad():
                    x_val, y_val = val_loader.next_batch()
                    with autocast_ctx:
                        loss = model(x_val, y_val)
                    val_loss += loss

            # Average validation loss across devices
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss = val_loss.item() / val_steps
            val_perplexity = math.exp(val_loss)
            log_message = (
                f"step:{step}/{hp.num_iterations} val_loss:{val_loss:.4f} val_ppl:{val_perplexity:.2f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps):.2f}ms"
            )
            print0(log_message)
            val_metrics = {
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "step": step,
                "time/training_time_ms": training_time_ms,  # Log total elapsed training time in ms
            }
            if utils.config.MASTER_PROCESS and not cli_args.no_wandb and not cli_args.debug:
                wandb.log(val_metrics)
            if json_logger:
                json_logger.log(val_metrics)
            pbar.set_postfix(val_loss=f"{val_loss:.4f}")

            # Restart training time for the next iteration
            torch.cuda.synchronize()
            t0 = time.time()

        if last_step:
            break

        model.train()
        for i in range(1, grad_accum_steps + 1):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()  # for logging
            loss = loss / grad_accum_steps
            x, y = train_loader.next_batch()

            # Handle backward pass based on model type
            if isinstance(model, DDP):
                # Turn off DDP grad sync if replicate_mesh_grad_sync is True
                ddp_no_sync = i < grad_accum_steps or hp.replicate_mesh_grad_sync
                if ddp_no_sync:
                    with model.no_sync():
                        loss.backward()
                else:
                    loss.backward()
            elif isinstance(model, FSDPModule):
                # Gradient accumulation for DP on top of FSDP
                model.set_is_last_backward(i == grad_accum_steps)
                if cli_args.fast_fsdp:
                    # Only reshard and reduce-scatter gradients upon the last backward pass
                    # Keep the entire unsharded model in memory during gradient accumulation
                    model.set_reshard_after_backward(i == grad_accum_steps)
                    model.set_requires_gradient_sync(i == grad_accum_steps)
                else:
                    # FSDP always synchronizes sharded gradients via reduce-scatter
                    model.set_requires_gradient_sync(True)
                loss.backward()
            else:
                # No DDP/FSDP - just do backward (optimizer handles sync)
                loss.backward()

        # Gradient norm
        grad_norm = torch.nn.utils.get_total_norm(
            [p.grad for p in model.parameters() if p.grad is not None]
        )

        # Optimizer step
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)

        # Approximate updated training time just before logging
        approx_time = training_time_ms + 1000 * (time.time() - t0)

        # Prepare logging dictionary
        train_perplexity = math.exp(train_loss.item())
        log_dict = {
            "train/loss": train_loss.item(),
            "train/perplexity": train_perplexity,
            "train/grad_norm": grad_norm.item(),
            "step": step,
            "time/training_time_ms": approx_time,  # Log approximate elapsed training time in ms
        }

        # Add CurvaDion metrics if using CurvaDion optimizer
        if hp.optimizer == "curvadion" and isinstance(optimizer, CurvaDion):
            metrics = optimizer.get_metrics()
            log_dict.update({
                "curvadion/sync_rate": metrics.get("sync_rate", 0.0),
                "curvadion/max_rmmc": metrics.get("recent_max_rmmc", 0.0),
                "curvadion/avg_max_rmmc": metrics.get("avg_max_rmmc", 0.0),
                "curvadion/recent_bytes": metrics.get("recent_bytes", 0),
                "curvadion/total_bytes_MB": metrics.get("total_bytes_transferred", 0) / (1024 * 1024),
                "curvadion/avg_bytes_per_step": metrics.get("avg_bytes_per_step", 0),
            })

        # Add DiLoCo metrics if using DiLoCoWrapper optimizer
        if hp.optimizer == "diloco_dion" and hasattr(optimizer, "get_metrics"):
            diloco_metrics = optimizer.get_metrics()
            log_dict.update(diloco_metrics)

        if utils.config.MASTER_PROCESS and not cli_args.no_wandb and not cli_args.debug:
            wandb.log(log_dict)
        if json_logger:
            json_logger.log(log_dict)
        if utils.config.MASTER_PROCESS and cli_args.debug:
            print0(
                f"Step {step}: train_loss={train_loss.item():.4f}, train_ppl={train_perplexity:.2f}, grad_norm={grad_norm.item():.4f}"
            )
        pbar.update(1)
        pbar.set_postfix(train_loss=f"{train_loss.item():.4f}", train_ppl=f"{train_perplexity:.2f}")

        if hp.checkpoint_freq > 0 and step % hp.checkpoint_freq == 0 and step > 0:
            # See if optimizer defines synchronize_for_checkpoint()
            if hasattr(optimizer, "synchronize_for_checkpoint"):
                # Dion with replicate_mesh_grad_sync will have decoupled optimizer states
                # Calling this is necessary to synchronize state across the replicate mesh
                # Otherwise, checkpoint results will not be consistent
                optimizer.synchronize_for_checkpoint()

            # Save a distributed checkpoint
            checkpoint_manager.save(step=step)

        torch.cuda.synchronize()
        t0 = time.time()  # reset timer after optimizer step

    pbar.close()
    print0(
        f"Peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )

    # Finalize JSON logger
    if json_logger:
        json_logger.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
