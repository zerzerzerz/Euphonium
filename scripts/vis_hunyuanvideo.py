#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HunyuanVideo Visualization/Inference Script

This script generates videos from a dataset for visualization.
It supports:
- Loading pre-trained or fine-tuned HunyuanVideo models
- Distributed inference with FSDP
- Multiple videos per prompt with different seeds
"""

# ============================================================================
# Standard Library Imports
# ============================================================================
import argparse
import json
import os
import re
from typing import Optional

# ============================================================================
# Third-party Imports
# ============================================================================
import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from diffusers.utils import check_min_version, export_to_video
from diffusers.video_processor import VideoProcessor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

# ============================================================================
# Local Imports from Euphonium
# ============================================================================
from euphonium.dataset import PromptEmbeddingDataset, prompt_embedding_collate_function
from euphonium.grpo.grpo_sampling import flux_step, sd3_time_shift
from euphonium.utils.communications import sp_parallel_dataloader_wrapper
from euphonium.utils.fsdp_util import get_dit_fsdp_kwargs
from euphonium.utils.load import load_transformer, load_vae
from euphonium.utils.logging_ import main_print
from euphonium.utils.parallel_states import (
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    initialize_sequence_parallel_state,
)

# Check diffusers version for compatibility
check_min_version("0.31.0")


def run_inference_step(
    args,
    z: torch.Tensor,
    progress_bar,
    sigma_schedule: torch.Tensor,
    transformer,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
):
    """
    Runs the main denoising loop for inference.
    
    Args:
        args: Configuration arguments
        z: Initial noisy latents
        progress_bar: tqdm progress bar
        sigma_schedule: Noise schedule
        transformer: The diffusion transformer model (FSDP wrapped)
        encoder_hidden_states: Text embeddings
        encoder_attention_mask: Attention mask for text
        
    Returns:
        latents: Final denoised latents scaled for VAE decoding
    """
    pred_original = None
    
    for i in progress_bar:
        B = encoder_hidden_states.shape[0]
        sigma = sigma_schedule[i]
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([B], timestep_value, device=z.device, dtype=torch.long)
        
        transformer.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            model_pred = transformer(
                hidden_states=z,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                guidance=torch.tensor([6018.0], device=z.device, dtype=torch.bfloat16),
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        # Call flux_step (grpo=False for inference, returns 2 values)
        result = flux_step(
            model_pred, z.to(torch.float32), args.eta, 
            sigmas=sigma_schedule, index=i,
            prev_sample=None,
            grpo=False,
            sde_solver=False,
        )
        z, pred_original = result[0], result[1]
        z = z.to(torch.bfloat16)
        
        progress_bar.set_postfix({"sigma": f"{sigma:.3f}"})

    # Scale the final latents for VAE decoding
    latents = pred_original.to(torch.float32) / 0.476986
    return latents


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Cleans a string to be used as a valid filename."""
    sanitized = text[:max_length]
    sanitized = re.sub(r'[\\/*?:"<>|]', "", sanitized)
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.strip('_')
    return sanitized


def generate_and_save_video(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states: torch.Tensor, 
    encoder_attention_mask: torch.Tensor,
    captions,
    batch_index: int = 0,
):
    """
    Generates videos based on text embeddings and saves them locally.
    
    Args:
        args: Configuration arguments
        device: CUDA device
        transformer: Diffusion transformer model
        vae: VAE decoder
        encoder_hidden_states: Text embeddings
        encoder_attention_mask: Attention mask
        captions: List of text captions
        batch_index: Current batch index
    """
    # Setup sampling parameters
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, sample_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    # Prepare latent dimensions
    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    # Process one video at a time
    batch_indices = torch.chunk(torch.arange(B), B)

    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_encoder_attention_mask = encoder_attention_mask[batch_idx]
        caption = captions[index]
        
        # Generate multiple videos with different seeds
        num_videos_per_prompt = getattr(args, 'num_videos_per_prompt', 1)
        
        for video_idx in range(num_videos_per_prompt):
            # Generate initial latent noise with unique seed
            generator = torch.Generator(device=device)
            current_seed = args.seed + index * 1000 + video_idx
            generator.manual_seed(current_seed)
            input_latents = torch.randn(
                (len(batch_idx), IN_CHANNELS, latent_t, latent_h, latent_w),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )

            # Run denoising process
            if num_videos_per_prompt > 1:
                desc = f"Sampling '{caption[:30]}...' ({video_idx+1}/{num_videos_per_prompt})"
            else:
                desc = f"Sampling for '{caption[:30]}...'"
            progress_bar = tqdm(range(0, sample_steps), desc=desc, disable=(int(os.environ.get("LOCAL_RANK", 0)) != 0))
            
            with torch.no_grad():
                final_latents = run_inference_step(
                    args,
                    input_latents.clone(),
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    batch_encoder_hidden_states,
                    batch_encoder_attention_mask,
                )
            
            # Decode latents to video
            if num_videos_per_prompt > 1:
                main_print(f"Decoding latents for caption: {caption} (seed: {current_seed}, {video_idx+1}/{num_videos_per_prompt})")
            else:
                main_print(f"Decoding latents for caption: {caption}")
                
            vae.enable_tiling()
            video_processor = VideoProcessor(vae_scale_factor=SPATIAL_DOWNSAMPLE)
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                video = vae.decode(final_latents, return_dict=False)[0]
                video_frames = video_processor.postprocess_video(video)

            # Create filename and save video
            rank = dist.get_rank()
            clean_caption = sanitize_filename(caption)
            if num_videos_per_prompt > 1:
                output_path = os.path.join(args.video_dir, f"rank{rank}_batch{batch_index}_{clean_caption}_seed{current_seed}.mp4")
            else:
                output_path = os.path.join(args.video_dir, f"rank{rank}_batch{batch_index}_{clean_caption}.mp4")
        
            main_print(f"Exporting video to: {output_path}")
            export_to_video(video_frames[0], output_path, fps=args.fps)


def main(args):
    """
    Main entry point for the visualization script.
    """
    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Distributed setup
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    
    if args.sp_size > 1:
        initialize_sequence_parallel_state(args.sp_size)

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Set video_dir to output_dir/videos if not specified
    if args.video_dir is None:
        args.video_dir = os.path.join(args.output_dir, "videos")
    
    # Create output directories
    if rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            main_print(f"Output directory created at: {args.output_dir}")
        if args.video_dir is not None:
            os.makedirs(args.video_dir, exist_ok=True)
            main_print(f"Video directory created at: {args.video_dir}")
        
        # Save args
        args_path = os.path.join(args.output_dir, "inference_args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # Load transformer model
    main_print(f"--> Loading transformer model from {args.pretrained_model_name_or_path}")
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.bfloat16,
    )

    # Load VAE model
    main_print("--> Loading VAE model...")
    vae, autocast_type, fps = load_vae(args.model_type, args.vae_model_path)
    
    # Initialize FSDP
    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    fsdp_kwargs, _ = get_dit_fsdp_kwargs(
        transformer, args.fsdp_sharding_startegy, False, args.use_cpu_offload, "bf16"
    )
    transformer = FSDP(transformer, **fsdp_kwargs)
    main_print(f"--> FSDP model loaded on rank {rank}")

    # Set models to evaluation mode
    transformer.eval()
    vae.eval()

    # Setup dataset and dataloader
    main_print("--> Setting up dataset and dataloader...")
    dataset = PromptEmbeddingDataset(args.data_json_path, args.t, args.cfg)
    sampler = DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=False, seed=args.seed
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=prompt_embedding_collate_function,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Calculate expected number of batches
    num_batches_per_rank = len(dataloader)
    
    # Wrapper for sequence parallelism
    loader = sp_parallel_dataloader_wrapper(
        dataloader, device, args.batch_size, args.sp_size, args.sp_batch_size
    )

    # Main visualization loop
    num_videos_per_prompt = getattr(args, 'num_videos_per_prompt', 1)
    main_print("***** Starting Video Generation *****")
    main_print(f"  Num examples = {len(dataset)}")
    main_print(f"  Batch size per device = {args.batch_size}")
    main_print(f"  Num batches per device = {num_batches_per_rank}")
    main_print(f"  Num videos per prompt = {num_videos_per_prompt}")
    main_print(f"  Total videos will be generated = {len(dataset) * num_videos_per_prompt} (across all {world_size} ranks)")
    
    # Calculate number of batches to process
    if args.num_visualization_samples is not None:
        samples_per_rank = (args.num_visualization_samples + world_size - 1) // world_size
        num_batches_to_process = (samples_per_rank + args.batch_size - 1) // args.batch_size
        num_batches_to_process = min(num_batches_to_process, num_batches_per_rank)
        main_print(f"  Limited to {num_batches_to_process} batches per rank (targeting ~{args.num_visualization_samples} total samples)")
    else:
        num_batches_to_process = num_batches_per_rank
        main_print(f"  Processing all {num_batches_to_process} batches per rank")
    
    progress_bar = tqdm(
        range(num_batches_to_process),
        desc="Processing Batches",
        disable=(int(os.environ.get("LOCAL_RANK", 0)) != 0),
    )

    for i, (encoder_hidden_states, encoder_attention_mask, captions) in enumerate(loader):
        if i >= num_batches_to_process:
            break
            
        main_print(f"\nProcessing batch {i+1}/{num_batches_to_process}...")
        
        generate_and_save_video(
            args,
            device,
            transformer,
            vae,
            encoder_hidden_states,
            encoder_attention_mask,
            captions,
            batch_index=i,
        )
        
        # Barrier to ensure all processes finish the batch
        dist.barrier()
        progress_bar.update(1)

    main_print("***** Visualization Complete *****")

    # Cleanup
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos from a dataset for visualization.")
    
    # === Essential Arguments ===
    parser.add_argument("--data_json_path", type=str, required=True, 
                        help="Path to the dataset JSON file.")
    parser.add_argument("--output_dir", type=str, default="./visualization_output", 
                        help="Directory to save the generated videos.")
    parser.add_argument("--video_dir", type=str, default=None, 
                        help="Directory to save video files (defaults to output_dir/videos).")
    parser.add_argument("--num_visualization_samples", type=int, default=None, 
                        help="Limit the number of videos to generate.")

    # === Model Paths ===
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained HunyuanVideo model.")
    parser.add_argument("--dit_model_name_or_path", type=str, default=None,
                        help="Path to fine-tuned DiT weights (optional).")
    parser.add_argument("--vae_model_path", type=str, required=True,
                        help="Path to VAE model.")
    parser.add_argument("--model_type", type=str, default="hunyuan_hf", 
                        help="Model type: hunyuan_hf or hunyuan.")

    # === Video & Sampling Parameters ===
    parser.add_argument("--h", type=int, default=256, help="Video height.")
    parser.add_argument("--w", type=int, default=256, help="Video width.")
    parser.add_argument("--t", type=int, default=16, help="Video length in frames.")
    parser.add_argument("--fps", type=int, default=8, help="FPS for the saved video.")
    parser.add_argument("--sampling_steps", type=int, default=50, 
                        help="Number of diffusion sampling steps.")
    parser.add_argument("--eta", type=float, default=0.0, help="Eta for the SDE solver.")
    parser.add_argument("--shift", type=float, default=1.0, 
                        help="Time shift value for sigma schedule.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed for reproducible generation.")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, 
                        help="Number of videos to generate per prompt with different seeds.")
    parser.add_argument("--cfg", type=float, default=0.0, 
                        help="Classifier-Free Guidance scale (used in dataset).")

    # === Dataloader and Batching ===
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size per device.")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    # === Distributed & System Configuration ===
    parser.add_argument("--sp_size", type=int, default=1, 
                        help="Sequence Parallelism size.")
    parser.add_argument("--sp_batch_size", type=int, default=1, 
                        help="Batch size for sequence parallel processing.")
    parser.add_argument("--fsdp_sharding_startegy", default="full", 
                        help="FSDP sharding strategy.")
    parser.add_argument("--use_cpu_offload", action="store_true", 
                        help="Use CPU offload with FSDP.")
    
    args = parser.parse_args()
    main(args)
