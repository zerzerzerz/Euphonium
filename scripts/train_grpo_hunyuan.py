# ============================================================================
# Standard Library Imports
# ============================================================================
import json
import math
import os
import time
from collections import deque

# ============================================================================
# Third-party Imports
# ============================================================================
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# ============================================================================
# Local Imports
# ============================================================================
# Dataset module
from euphonium.dataset import PromptEmbeddingDataset, prompt_embedding_collate_function

# Reward model module
from euphonium.reward_models import create_reward_model

# GRPO module
from euphonium.grpo.grpo_config import parse_args, create_reward_config_from_args
from euphonium.grpo.grpo_training import train_one_step

# Utility module
from euphonium.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)
from euphonium.utils.communications import sp_parallel_dataloader_wrapper
from euphonium.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
from euphonium.utils.load import load_transformer, load_vae
from euphonium.utils.logging_ import main_print
from euphonium.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
)

# Check diffusers version - ensure compatibility
check_min_version("0.31.0")


def main(args):
    # Enable TF32 for faster matrix multiplication (provides better performance on Ampere GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True

    # === Distributed Training Initialization ===
    local_rank = int(os.environ["LOCAL_RANK"])    # GPU ID within the current node
    rank = int(os.environ["RANK"])                # Global process ID
    world_size = int(os.environ["WORLD_SIZE"])    # Total number of processes (all GPUs)
    
    # Initialize NCCL communication backend for multi-GPU coordination
    dist.init_process_group("nccl")
    
    # Set the GPU used by the current process
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    
    # Initialize sequence parallel state (used for processing long sequences)
    initialize_sequence_parallel_state(args.sp_size)

    # === Random Seed Setup ===
    if args.seed is not None:
        # Set random seed to ensure reproducibility
        # Note: The same sequence parallel group should use the same seed, but noise generation requires different seeds
        set_seed(args.seed)

    # === Output Directory Creation ===
    # Only create output directory in the main process (rank 0) to avoid race conditions
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        args_save_path = os.path.join(args.output_dir, 'args.json')
        with open(args_save_path, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        main_print(f"Saved training args to {args_save_path}")

    # === Reward Model Initialization ===
    # Using the new extensible reward model architecture
    reward_model = None
    
    # Create reward model configuration from command line arguments (directly returns a dict)
    reward_config = create_reward_config_from_args(args)
    reward_config['device'] = f'cuda:{device}'
    
    # Create TRL reward model
    reward_model = create_reward_model('trl', reward_config)

    # === Main Model Loading ===
    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    main_print(f"--> loading model from {args.model_type}")
    
    # === Extract init_step from checkpoint path ===
    init_steps = 0
    if args.dit_model_name_or_path:
        # Try to extract the checkpoint number from the path
        # Example format: /path/to/checkpoint-50-0 or checkpoint-50
        import re
        checkpoint_pattern = r'checkpoint-(\d+)'
        match = re.search(checkpoint_pattern, args.dit_model_name_or_path)
        if match:
            init_steps = int(match.group(1))
            main_print(f"✓ Extracted init_steps = {init_steps} from checkpoint path")
        else:
            main_print(f"✓ Could NOT extract step count from checkpoint path, using default init_steps = 0")
    else:
        main_print(f"✓ No checkpoint path specified, using default init_steps = 0")
    
    # Load diffusion Transformer model
    transformer = load_transformer(
        args.model_type,                      # Model type (e.g., "hunyuan_hf")
        args.dit_model_name_or_path,          # DiT checkpoint path
        args.pretrained_model_name_or_path,   # Pretrained weight path
        # Master weight type: fp32 provides higher precision, bf16 saves memory
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )

    # Print model size information
    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    
    # === FSDP (Fully Sharded Data Parallel) Initialization ===
    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    
    # Get FSDP configuration parameters
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,                    # Model to be sharded
        args.fsdp_sharding_startegy,   # Sharding strategy (e.g., "full", "grad_op")
        False,                         # Whether to use activation checkpointing
        args.use_cpu_offload,          # Whether to offload parameters to CPU
        args.master_weight_type,       # Master weight data type
    )
    
    # Apply FSDP wrapper to enable distributed training for large models
    transformer = FSDP(transformer, **fsdp_kwargs,)
    
    main_print(f"--> model loaded")

    # === KL Divergence Regularization: Load Reference Model ===
    ref_transformer = None
    if args.use_kl_loss:
        main_print(f"--> loading reference model for KL divergence regularization")
        main_print(f"    KL coefficient: {args.kl_coef}, KL type: {args.kl_type}")
        
        # Load the untrained pretrained model as reference
        ref_transformer = load_transformer(
            args.model_type,
            None,  # Do not use checkpoint, load original pretrained weights
            args.pretrained_model_name_or_path,
            torch.bfloat16,  # Reference model uses bf16 to save video memory
        )
        
        # Freeze all parameters of the reference model
        for param in ref_transformer.parameters():
            param.requires_grad = False
        
        # Apply FSDP wrapper to the reference model as well (for distributed inference)
        ref_fsdp_kwargs, _ = get_dit_fsdp_kwargs(
            ref_transformer,
            args.fsdp_sharding_startegy,
            False,
            args.use_cpu_offload,
            "bf16",  # Reference model uses bf16
        )
        ref_transformer = FSDP(ref_transformer, **ref_fsdp_kwargs,)
        ref_transformer.eval()  # Set to evaluation mode
        
        main_print(f"--> reference model loaded and frozen")

    # === Gradient Checkpointing Setup ===
    if args.gradient_checkpointing:
        # Apply gradient checkpointing to save video memory (trade computation time for memory space)
        apply_fsdp_checkpointing(
            transformer, 
            no_split_modules,              # Modules that should NOT be checkpointed
            args.selective_checkpointing   # Selective checkpointing ratio
        )

    # === Model Training Mode Setup ===
    transformer.train()  # Set to training mode (enable dropout, batch norm, etc.)

    # === Optimizer Initialization ===
    # Get all parameters that need to be trained
    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    # Use AdamW optimizer (Adam + weight decay)
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,      # Learning rate
        betas=(0.9, 0.999),         # Adam momentum parameters
        weight_decay=args.weight_decay,  # L2 regularization coefficient
        eps=1e-8,                   # Numerical stability parameter
    )

    main_print(f"optimizer: {optimizer}")

    # === Learning Rate Scheduler Setup ===
    # Note: If resuming from a checkpoint, last_epoch will be set in load_checkpoint
    lr_scheduler = get_scheduler(
        args.lr_scheduler,              # Scheduler type (e.g., "cosine", "linear")
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,    # Warmup steps
        num_training_steps=1000000,     # Total training steps
        num_cycles=args.lr_num_cycles,  # Number of cycles (for cosine with restarts)
        power=args.lr_power,            # Power for polynomial scheduler
        last_epoch=-1,                  # Start from scratch or recover from checkpoint
    )

    # === Resume from Checkpoint: Load Full Training State ===
    if init_steps > 0 and args.dit_model_name_or_path:
        main_print(f"🔄 Resuming from checkpoint, loading full training state...")
        try:
            transformer, optimizer, lr_scheduler, loaded_step, loaded_epoch = load_checkpoint(
                transformer=transformer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                checkpoint_dir=args.dit_model_name_or_path,
                rank=rank
            )
            # Update init_steps to the actual loaded step count
            init_steps = loaded_step
            main_print(f"✓ Checkpoint loaded successfully, resuming training from step {init_steps + 1}")
        except Exception as e:
            main_print(f"⚠ Failed to load checkpoint: {e}")
            main_print(f"  Will use loaded model weights, but optimizer and scheduler will start from scratch")

    # === Dataset and DataLoader Setup ===
    
    # Create latent space dataset (pre-processed text embeddings, etc.)
    train_dataset = PromptEmbeddingDataset(args.data_json_path, args.t, args.cfg)
    
    # Distributed sampler: ensures each GPU processes a different subset of data
    sampler = DistributedSampler(
            train_dataset, 
            rank=rank,                    # Current process rank
            num_replicas=world_size,      # Total number of processes
            shuffle=True,                 # Shuffle data each epoch
            seed=args.sampler_seed        # Random seed for sampling
        )
    
    # DataLoader configuration
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,              # Distributed sampler
        collate_fn=prompt_embedding_collate_function,  # Data collation function
        pin_memory=True,              # Pin memory to speed up GPU transfers
        batch_size=args.train_batch_size,    # Batch size per GPU
        num_workers=args.dataloader_num_workers,  # Number of data loading processes
        drop_last=True,               # Drop the last incomplete batch
    )

    # === Training Steps Calculation ===
    # Calculate update steps per epoch (considering gradient accumulation and sequence parallelism)
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)                # DataLoader length
        / args.gradient_accumulation_steps   # Gradient accumulation steps
        * args.sp_size                       # Sequence parallel size
        / args.train_sp_batch_size          # Sequence parallel batch size
    )
    # Calculate required epochs based on maximum training steps
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # === VAE Model Loading ===
    # Load VAE for converting between latent vectors and video pixels
    vae, autocast_type, fps = load_vae(args.model_type, args.vae_model_path)

    # === Experiment Tracking Initialization ===
    writer = None
    if rank <= 0:
        # Initialize TensorBoard logger only in the main process to avoid duplicate logs
        output_base = args.output_dir if args.output_dir else "outputs"
        log_dir = os.path.join(output_base, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    # === Training Information Print ===
    # Calculate effective total batch size (considering all parallelization strategies)
    total_batch_size = (
        world_size                          # Total GPUs
        * args.gradient_accumulation_steps  # Gradient accumulation multiplier
        / args.sp_size                      # Sequence parallel group size
        * args.train_sp_batch_size         # Sequence parallel batch size
    )
    
    main_print("=" * 80)
    main_print("***** GRPO Training Configuration *****")
    main_print("=" * 80)
    
    # === Basic Training Info ===
    main_print(f"📊 Dataset Info:")
    main_print(f"  Dataset sample count: {len(train_dataset)}")
    main_print(f"  DataLoader size: {len(train_dataloader)}")
    main_print(f"  Data file path: {args.data_json_path}")
    
    # === Model Configuration Info ===
    main_print(f"🤖 Model Configuration:")
    main_print(f"  Model type: {args.model_type}")
    main_print(f"  Pretrained model path: {args.pretrained_model_name_or_path}")
    main_print(f"  VAE model path: {args.vae_model_path}")
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    main_print(f"  Total parameters: {total_params / 1e9:.2f}B")
    main_print(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")
    main_print(f"  Parameter dtype: {transformer.parameters().__next__().dtype}")
    main_print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    
    # === Distributed Training Configuration ===
    main_print(f"🌐 Distributed Configuration:")
    main_print(f"  World size (Total GPUs): {world_size}")
    main_print(f"  Current rank: {rank}")
    main_print(f"  Local rank: {local_rank}")
    main_print(f"  FSDP sharding strategy: {args.fsdp_sharding_startegy}")
    main_print(f"  Sequence parallel size: {args.sp_size}")
    main_print(f"  CPU offload: {args.use_cpu_offload}")
    
    # === Batch Configuration ===
    main_print(f"Box Batch Configuration:")
    main_print(f"  Per-device batch size: {args.train_batch_size}")
    main_print(f"  Sequence parallel batch size: {args.train_sp_batch_size}")
    main_print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    main_print(f"  Effective total batch size: {total_batch_size}")
    
    # === Training Hyperparameters ===
    main_print(f"Hyperparameters:")
    main_print(f"  Learning rate: {args.learning_rate}")
    main_print(f"  Weight decay: {args.weight_decay}")
    main_print(f"  Max grad norm: {args.max_grad_norm}")
    main_print(f"  LR scheduler: {args.lr_scheduler}")
    main_print(f"  Warmup steps: {args.lr_warmup_steps}")
    main_print(f"  Max training steps: {args.max_train_steps}")
    main_print(f"  Checkpointing steps: {args.checkpointing_steps}")
    
    # === GRPO Specific Parameters ===
    main_print(f"GRPO Configuration:")
    main_print(f"  Use group: {args.use_group}")
    main_print(f"  Num generations per prompt: {args.num_generations}")
    main_print(f"  Best-of-N count: {args.bestofn}")
    main_print(f"  Use same noise: {args.use_same_noise}")
    main_print(f"  Timestep training fraction: {args.timestep_fraction}")
    main_print(f"  Noise param eta: {args.eta}")
    main_print(f"  Time shift: {args.shift}")
    main_print(f"  Use KL loss: {args.use_kl_loss}")
    if args.use_kl_loss:
        main_print(f"  KL coef: {args.kl_coef}")
        main_print(f"  KL type: {args.kl_type}")
    
    # === Video Parameters ===
    main_print(f"Video Configuration:")
    main_print(f"  Video dimensions: {args.w}x{args.h}")
    main_print(f"  Video frames: {args.t}")
    main_print(f"  Sampling steps: {args.sampling_steps}")
    main_print(f"  Output FPS: {args.fps}")
    
    # === Reward Model Configuration ===
    main_print(f"Reward Model:")
    if reward_model is not None:
        main_print(f"  Reward model type: trl")
        main_print(f"  TRL model path: {getattr(args, 'trl_model_path', 'N/A')}")
    else:
        main_print(f"  ❌ Reward model NOT configured")
    
    # === System Resources ===
    if torch.cuda.is_available():
        main_print(f"System Resource (GPU):")
        main_print(f"  GPU device: {torch.cuda.get_device_name(device)}")
        main_print(f"  GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB")
        main_print(f"  Current memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB")
    
    main_print(f"🚀 Starting Training:")
    main_print(f"  Resuming from step {init_steps}")
    output_base = args.output_dir if args.output_dir else "outputs"
    main_print(f"  Output directory: {output_base}")
    main_print("=" * 80)

    # === Training Progress Bar Setup ===
    # Note: total set to max_train_steps to ensure progress bar shows correct total steps
    # initial set to init_steps to show completed steps (e.g., "150/501")
    progress_bar = tqdm(
        range(init_steps+1, args.max_train_steps+1),
        initial=init_steps,
        total=args.max_train_steps,  # Show correct total steps
        desc="Steps",
        disable=local_rank > 0,  # Enable progress bar only on the main GPU
    )

    # === Sequence Parallel DataLoader Wrapper ===
    # Wrap DataLoader to support sequence parallel training
    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,             # Sequence parallel group size
        args.train_sp_batch_size, # Sequence parallel batch size
    )

    # === EMA handler (unused, kept for compatibility) ===
    ema_handler = None

    # === Performance Monitoring ===
    step_times = deque(maxlen=100)  # Store time stats for last 100 steps

    # === Resume from Checkpoint: Skip Trained Data ===
    if init_steps > 0:
        main_print(f"🔄 Resuming from checkpoint, preparing to skip {init_steps} steps of data...")
        skipped_count = 0
        skip_start_time = time.time()
        
        # Skip already trained data
        for _ in range(init_steps):
            try:
                _ = next(loader)
                skipped_count += 1
                if skipped_count % 100 == 0:
                    main_print(f"  Skipped {skipped_count}/{init_steps} steps of data...")
            except StopIteration:
                # If data runs out, restart iterator
                main_print(f"  DataLoader iterator finished, restarting...")
                loader = sp_parallel_dataloader_wrapper(
                    train_dataloader,
                    device,
                    args.train_batch_size,
                    args.sp_size,
                    args.train_sp_batch_size,
                )
                _ = next(loader)
                skipped_count += 1
        
        skip_time = time.time() - skip_start_time
        main_print(f"✓ Skipped {skipped_count} steps of data in {skip_time:.2f}s")
        main_print(f"  Continuing training from step {init_steps + 1}")
    
    # === Main Training Loop ===
    main_print(f"=== Starting Training Loop ===")
    for epoch in range(1):  # Currently training for 1 epoch (can be adjusted)
        main_print(f"Starting epoch {epoch+1}")
        # Distributed Training: set epoch to ensure different shuffling each round
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)  # Critical: ensure different data distribution each epoch
            main_print(f"DistributedSampler epoch set to {epoch}")

        # Train by steps instead of epochs (more precise control)
        for step in range(init_steps+1, args.max_train_steps+1):
            start_time = time.time()
            
            # === Periodically Save Checkpoints ===
            if step % args.checkpointing_steps == 0:
                # Save model state, optimizer, lr_scheduler for resuming
                save_checkpoint(
                    transformer=transformer, 
                    rank=rank, 
                    output_dir=args.output_dir, 
                    step=step, 
                    epoch=epoch,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler
                )
                
                dist.barrier()  # Sync all GPUs to ensure saving is complete
            
            # === Execute One GRPO Training Step ===
            loss, grad_norm, train_rewards, extra_info = train_one_step(
                args,
                device, 
                transformer,      # Model to be trained
                vae,             # VAE decoder
                reward_model,    # Unified reward model interface
                optimizer,       # Optimizer
                lr_scheduler,    # LR scheduler
                loader,          # DataLoader
                args.max_grad_norm,  # Gradient clipping threshold
                step,            # Current step
                writer,          # TensorBoard writer
                ref_transformer, # Reference model (for KL regularization)
            )
    
            # === Performance Monitoring and Logging ===
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s", 
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            
            # Main process records detailed logs to TensorBoard
            if rank <= 0 and writer is not None:
                # === Basic Training Metrics ===
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], step)
                writer.add_scalar("train/step_time", step_time, step)
                writer.add_scalar("train/avg_step_time", avg_step_time, step)
                writer.add_scalar("train/grad_norm", grad_norm, step)
                
                # === KL Loss ===
                if args.use_kl_loss and "kl_loss" in extra_info:
                    writer.add_scalar("train/kl_loss", extra_info["kl_loss"], step)
                
                # === GPU Resource Usage ===
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
                    memory_cached = torch.cuda.memory_cached(device) / 1024**3       # GB
                    
                    writer.add_scalar("system/gpu_memory_allocated_gb", memory_allocated, step)
                    writer.add_scalar("system/gpu_memory_reserved_gb", memory_reserved, step)
                    writer.add_scalar("system/gpu_memory_cached_gb", memory_cached, step)
                    writer.add_scalar("system/gpu_utilization", memory_allocated / memory_reserved * 100 if memory_reserved > 0 else 0, step)
                
                # === Training Efficiency Metrics ===
                samples_per_second = args.train_batch_size * world_size / step_time if step_time > 0 else 0
                writer.add_scalar("perf/samples_per_second", samples_per_second, step)
                writer.add_scalar("perf/steps_per_minute", 60.0 / step_time if step_time > 0 else 0, step)
                
                # === Reward Analysis ===
                for reward_name, reward_values in train_rewards.items():
                    # Basic statistics
                    reward_mean = reward_values.mean().item()
                    reward_std = reward_values.std().item()
                    reward_max = reward_values.max().item()
                    reward_min = reward_values.min().item()
                    
                    writer.add_scalar(f"rewards/{reward_name}_mean", reward_mean, step)
                    writer.add_scalar(f"rewards/{reward_name}_std", reward_std, step)
                    writer.add_scalar(f"rewards/{reward_name}_max", reward_max, step)
                    writer.add_scalar(f"rewards/{reward_name}_min", reward_min, step)
                    writer.add_scalar(f"rewards/{reward_name}_range", reward_max - reward_min, step)
                    
                    # Distribution Analysis
                    if len(reward_values) > 1:
                        # Calculate quartiles
                        sorted_rewards = torch.sort(reward_values)[0]
                        q25_idx = len(sorted_rewards) // 4
                        q75_idx = 3 * len(sorted_rewards) // 4
                        median_idx = len(sorted_rewards) // 2
                        
                        writer.add_scalar(f"rewards/{reward_name}_q25", sorted_rewards[q25_idx].item(), step)
                        writer.add_scalar(f"rewards/{reward_name}_median", sorted_rewards[median_idx].item(), step)
                        writer.add_scalar(f"rewards/{reward_name}_q75", sorted_rewards[q75_idx].item(), step)
                        writer.add_scalar(f"rewards/{reward_name}_iqr", sorted_rewards[q75_idx].item() - sorted_rewards[q25_idx].item(), step)
                    
                    # Log reward histogram
                    if step % 10 == 0:  # Record histogram every 10 steps
                        writer.add_histogram(f"rewards_dist/{reward_name}_dist_{step:03d}", reward_values, step)
 
                # === Scheduler State ===
                if hasattr(lr_scheduler, 'get_last_lr'):
                    for i, lr in enumerate(lr_scheduler.get_last_lr()):
                        writer.add_scalar(f"lr/group_{i}", lr, step)
                
                writer.flush()  # Ensure logs are written timely

    # === Training Summary ===
    main_print("=" * 80)
    main_print("🎉 GRPO Training Finished!")
    main_print("=" * 80)
    
    # === Resource Cleanup ===
    # Close TensorBoard writer
    if rank <= 0 and writer is not None:
        writer.close()
    
    if get_sequence_parallel_state():
        # Destroy sequence parallel communication group
        destroy_sequence_parallel_group()
    
    main_print("All resources cleaned up, training script finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)