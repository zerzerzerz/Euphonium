"""SPSA (Simultaneous Perturbation Stochastic Approximation) Gradient Estimation."""

from __future__ import annotations

import os
from typing import Dict, Tuple, Optional

import torch
from diffusers.utils import export_to_video

from ...utils.logging_ import main_print


def compute_spsa_reward_gradient(
    pred_original_sample: torch.Tensor,
    vae,
    video_processor,
    reward_model,
    caption: str,
    args,
    step: int,
    batch_index: int,
    sample_index: int,
    sigma: float = 0.01,
    num_samples: int = 1,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    transformer_config = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Estimate the gradient of the reward with respect to the latent in RGB space using SPSA (Simultaneous Perturbation Stochastic Approximation).
    
    Core Idea: Use high-dimensional stochastic finite-difference approximation to estimate the gradient.
    
    Mathematical Principle:
    Assume the entire process of decoding and scoring is a black-box function f(x) = VLM(VAE(x)).
    Based on Taylor expansion, the gradient can be approximated as:
    
    ∇_x f(x) ≈ E_{u ~ N(0, I)} [ (f(x + σu) - f(x - σu)) / (2σ) · u ]
    
    Args:
        pred_original_sample: (B, C, T, H, W) Predicted clean latent sample.
        vae: VAE model, used for decoding latents to video.
        video_processor: Video processor, used for post-processing videos.
        reward_model: Reward model, used to calculate video scores.
        caption: Text description corresponding to the video.
        args: Training arguments.
        step: Current training step.
        batch_index: Current batch index.
        sample_index: Current sampling step index.
        sigma: Perturbation coefficient, controlling the magnitude of random noise.
        num_samples: Number of random sample sets used to estimate the gradient (more = more accurate but slower).
        encoder_hidden_states: Prompt embeddings, used for the latent reward model.
        encoder_attention_mask: Attention mask.
        transformer_config: Transformer configuration, used to obtain `pooled_projection_dim`.
        
    Returns:
        gradient: (B, C, T, H, W) Estimated gradient of the reward with respect to the latent.
        stats: A dictionary containing statistical information.
    """
    device = pred_original_sample.device
    dtype = pred_original_sample.dtype
    
    stats = {
        "r_plus_mean": 0.0,
        "r_minus_mean": 0.0,
        "gradient_norm": 0.0,
    }
    
    accumulated_gradient = torch.zeros_like(pred_original_sample, dtype=torch.float32)
    
    rank = int(os.environ.get("RANK", 0))
    
    # Create temporary directory to store SPSA videos
    spsa_video_dir = os.path.join(args.output_dir, "spsa_videos_temp")
    os.makedirs(spsa_video_dir, exist_ok=True)
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Generate random perturbation vector u ~ N(0, I)
            u = torch.randn_like(pred_original_sample, dtype=torch.float32)
            
            # Compute x + σu and x - σu
            x_plus = pred_original_sample.float() + sigma * u
            x_minus = pred_original_sample.float() - sigma * u
            
            # Decode x + σu to get video and save
            latents_plus = x_plus.to(dtype) / 0.476986
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video_plus = vae.decode(latents_plus, return_dict=False)[0]
                videos_plus = video_processor.postprocess_video(video_plus)
            
            # Save plus video
            video_plus_filename = f"spsa_step{step:06d}_rank{rank}_batch{batch_index}_sample{sample_index}_n{sample_idx}_plus.mp4"
            video_plus_path = os.path.join(spsa_video_dir, video_plus_filename)
            export_to_video(videos_plus[0], video_plus_path, fps=args.fps)
            
            # Decode x - σu to get video and save
            latents_minus = x_minus.to(dtype) / 0.476986
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video_minus = vae.decode(latents_minus, return_dict=False)[0]
                videos_minus = video_processor.postprocess_video(video_minus)
            
            # Save minus video
            video_minus_filename = f"spsa_step{step:06d}_rank{rank}_batch{batch_index}_sample{sample_index}_n{sample_idx}_minus.mp4"
            video_minus_path = os.path.join(spsa_video_dir, video_minus_filename)
            export_to_video(videos_minus[0], video_minus_path, fps=args.fps)
            
            # Construct latent_data for reward computation (if latent reward is needed)
            latent_data_plus = None
            latent_data_minus = None
            if encoder_hidden_states is not None and transformer_config is not None:
                pooled_projection_dim = transformer_config.pooled_projection_dim
                pooled_projections = encoder_hidden_states[:, 0, :pooled_projection_dim].to(torch.bfloat16)
                prompt_embeds = encoder_hidden_states[:, 1:].to(torch.bfloat16)
                
                latent_data_plus = {
                    'noisy_latents': x_plus.to(torch.bfloat16),
                    'timestep': torch.zeros(x_plus.shape[0], device=device, dtype=torch.long),
                    'prompt_embeds': prompt_embeds,
                    'prompt_attention_mask': encoder_attention_mask,
                    'pooled_prompt_embeds': pooled_projections,
                }
                latent_data_minus = {
                    'noisy_latents': x_minus.to(torch.bfloat16),
                    'timestep': torch.zeros(x_minus.shape[0], device=device, dtype=torch.long),
                    'prompt_embeds': prompt_embeds,
                    'prompt_attention_mask': encoder_attention_mask,
                    'pooled_prompt_embeds': pooled_projections,
                }
            
            # Compute scores using the reward model
            try:
                reward_output_plus = reward_model.compute_rewards(
                    [os.path.abspath(video_plus_path)],
                    [caption],
                    latent_data=latent_data_plus,
                )
                # Get primary score (score)
                r_plus = reward_output_plus.rewards.get('score', list(reward_output_plus.rewards.values())[0])
                if r_plus.dim() == 0:
                    r_plus = r_plus.unsqueeze(0)
                
                reward_output_minus = reward_model.compute_rewards(
                    [os.path.abspath(video_minus_path)],
                    [caption],
                    latent_data=latent_data_minus,
                )
                r_minus = reward_output_minus.rewards.get('score', list(reward_output_minus.rewards.values())[0])
                if r_minus.dim() == 0:
                    r_minus = r_minus.unsqueeze(0)
                    
            except Exception as e:
                main_print(f"[SPSA] Reward computation failed: {e}")
                r_plus = torch.zeros(1, device=device)
                r_minus = torch.zeros(1, device=device)
            
            # Clean up temporary video files
            try:
                # if os.path.exists(video_plus_path):
                #     os.remove(video_plus_path)
                # if os.path.exists(video_minus_path):
                #     os.remove(video_minus_path)
                pass
            except Exception:
                pass
            
            # Record statistics
            stats["r_plus_mean"] += r_plus.mean().item() / num_samples
            stats["r_minus_mean"] += r_minus.mean().item() / num_samples
            
            # Calculate gradient approximation: g ≈ (r+ - r-) / (2σ) · u
            # Note: r_plus and r_minus are (B,), need to expand dimensions to match u
            r_diff = (r_plus - r_minus).float().to(device)  # (B,)
            # Expand r_diff to the same dimension as u
            for _ in range(u.dim() - 1):
                r_diff = r_diff.unsqueeze(-1)
            
            gradient_sample = (r_diff / (2 * sigma)) * u
            accumulated_gradient = accumulated_gradient + gradient_sample
    
    # Average gradient estimates from multiple samples
    estimated_gradient = accumulated_gradient / num_samples
    
    stats["gradient_norm"] = estimated_gradient.norm().item()
    
    return estimated_gradient.to(dtype), stats
