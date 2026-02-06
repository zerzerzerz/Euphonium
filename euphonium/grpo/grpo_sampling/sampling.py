"""Main sampling functions for GRPO."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from tqdm.auto import tqdm

from ...reward_models import RewardOutput
from ...utils.logging_ import main_print

from .flux_step import flux_step, sd3_time_shift
from .spsa_gradient import compute_spsa_reward_gradient
from .utils import should_apply_process_reward, should_apply_spsa_reward


def run_sample_step(
    args,
    z: torch.Tensor,
    progress_bar,
    sigma_schedule: torch.Tensor,
    transformer,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    grpo_sample: bool,
    reward_model=None,
    vae=None,
    caption: str = "",
    step: int = 0,
    batch_index: int = 0,
    sde_solver: bool = True,
):
    """
    Given an initial noise, complete the full denoising process.
    
    Note: The format of `encoder_hidden_states` is:
    - The first token ([:, 0, :pooled_projection_dim]) contains `pooled_projections`.
    - The remaining tokens ([:, 1:]) are the actual `encoder_hidden_states`.
    
    For latent reward calculation, these two parts need to be extracted from `encoder_hidden_states`.
    `pooled_projection_dim` is obtained from `transformer.config.pooled_projection_dim`.
    
    SPSA Reward Guidance:
    When `spsa_reward_enabled` is set to True, the SPSA method is used to estimate the gradient of the reward with respect to the latent in the RGB space during each sampling step:
    1. Generate random noise u ~ N(0, I).
    2. Compute r+ = reward(decode(pred_original + σu)) and r- = reward(decode(pred_original - σu)).
    3. Gradient estimation: g ≈ (r+ - r-) / (2σ) · u.
    4. Use the estimated gradient for guidance.

    Returns:
        z: shape = [b, *], the final latent after step-by-step denoising.
        latents: shape = [b, *], the final clean latent (pred_original), with some scaling applied, used for exporting video.
        all_latents: shape = [b, num_steps+1, *], records the latent at each step, +1 because it includes the initial latent.
        all_log_probs: shape = [b, num_steps], records the log_prob at each step.
        all_pred_original: shape = [b, num_steps, *], records the pred_original at each step.
        all_process_rewards: shape = [b, num_steps] or None, records the process reward score at each step (if enabled).
    """
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        all_pred_original = []
        all_process_rewards = []
        all_spsa_stats = []

        sampling_stats = {
            "step_times": [],
            "model_pred_stats": [],
            "log_prob_stats": [],
            "process_reward_stats": [],
            "spsa_reward_stats": [],
            "rgg_times": [],  # Records RGG computation time for each step.
        }
        
        # Check if process reward guidance during sampling is enabled
        use_process_reward = (
            reward_model is not None 
            and getattr(args, 'process_latent_reward_enabled', False)
            and getattr(args, 'process_latent_reward_sampling_enabled', False)
            and hasattr(reward_model, 'is_process_reward_enabled')
            and reward_model.is_process_reward_enabled()
        )
        
        # Pre-fetch process reward relevant parameters
        process_reward_guidance_scale = getattr(args, 'process_latent_reward_guidance_scale', 0.1)
        normalize_reward_gradient = getattr(args, 'process_latent_reward_normalize_gradient', True)
        reward_gradient_kl_beta = getattr(args, 'process_latent_reward_kl_beta', 0.1)
        
        if use_process_reward:
            if process_reward_guidance_scale >= 0:
                main_print(f"[Process Reward] Enabled, guidance_scale={process_reward_guidance_scale}, normalize={normalize_reward_gradient}")
            else:
                actual_scale = (args.eta ** 2) / (2 * reward_gradient_kl_beta)
                main_print(f"[Process Reward] Enabled, using KL divergence mode: eta^2/(2*beta)={actual_scale:.4f} (eta={args.eta}, beta={reward_gradient_kl_beta}), normalize={normalize_reward_gradient}")
        
        # Check if SPSA reward gradient guidance is enabled
        use_spsa_reward = (
            getattr(args, 'spsa_reward_enabled', False)
            and vae is not None
            and reward_model is not None
        )
        
        # Pre-fetch SPSA relevant parameters
        spsa_sigma = getattr(args, 'spsa_reward_sigma', 0.01)
        spsa_guidance_scale = getattr(args, 'spsa_reward_guidance_scale', 0.1)
        spsa_normalize_gradient = getattr(args, 'spsa_reward_normalize_gradient', True)
        spsa_num_samples = getattr(args, 'spsa_reward_num_samples', 1)
        spsa_gradient_kl_beta = getattr(args, 'spsa_reward_kl_beta', 0.1)
        
        if use_spsa_reward:
            if spsa_guidance_scale >= 0:
                main_print(f"[SPSA Reward] Enabled, sigma={spsa_sigma}, guidance_scale={spsa_guidance_scale}, "
                          f"normalize={spsa_normalize_gradient}, num_samples={spsa_num_samples}")
            else:
                actual_scale = (args.eta ** 2) / (2 * spsa_gradient_kl_beta)
                main_print(f"[SPSA Reward] Enabled, using KL divergence mode: eta^2/(2*beta)={actual_scale:.4f} "
                          f"(eta={args.eta}, beta={spsa_gradient_kl_beta}), sigma={spsa_sigma}, "
                          f"normalize={spsa_normalize_gradient}, num_samples={spsa_num_samples}")
            video_processor_for_spsa = VideoProcessor(vae_scale_factor=8)
        
        total_steps = len(list(range(progress_bar.n, progress_bar.total)))

        for i in progress_bar:
            step_start_time = time.time()

            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full(
                [encoder_hidden_states.shape[0]],
                timestep_value,
                device=z.device,
                dtype=torch.long,
            )

            transformer.eval()

            with torch.autocast("cuda", torch.bfloat16):
                model_pred = transformer(
                    hidden_states=z,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    guidance=torch.tensor(
                        [6018.0],
                        device=z.device,
                        dtype=torch.bfloat16,
                    ),
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

            # Process Reward Guidance
            process_reward_score = None
            reward_gradient = None
            apply_reward_guidance = use_process_reward and should_apply_process_reward(args, i, total_steps)
            
            rgg_time_this_step = 0.0  # RGG computation time for this step.
            
            if apply_reward_guidance:
                try:
                    _pooled_projection_dim = transformer.config.pooled_projection_dim
                    pooled_projections_for_reward = encoder_hidden_states[:, 0, :_pooled_projection_dim]
                    prompt_embeds_for_reward = encoder_hidden_states[:, 1:]
                    
                    rgg_start_time = time.time()  # Start RGG timing
                    score, gradient = reward_model.compute_process_reward(
                        noisy_latents=z.to(torch.bfloat16),
                        timestep=timesteps,
                        prompt_embeds=prompt_embeds_for_reward,
                        prompt_attention_mask=encoder_attention_mask,
                        pooled_prompt_embeds=pooled_projections_for_reward,
                        return_gradient=True,
                    )
                    rgg_time_this_step = time.time() - rgg_start_time  # End RGG timing
                    
                    process_reward_score = score.mean().item()
                    reward_gradient = gradient
                    all_process_rewards.append(score.detach())
                    
                except Exception as e:
                    main_print(f"[Process Reward] Step {i} computation failed: {e}")
                    # In case of failure, record zero values to maintain consistent array lengths.
                    batch_size = z.shape[0]
                    all_process_rewards.append(torch.zeros(batch_size, device=z.device, dtype=torch.float32))
            elif use_process_reward:
                # RGG is enabled but not applied at this step: record zeros to maintain consistent array lengths.
                batch_size = z.shape[0]
                all_process_rewards.append(torch.zeros(batch_size, device=z.device, dtype=torch.float32))
            
            sampling_stats["rgg_times"].append(rgg_time_this_step)

            # SPSA Reward Gradient Guidance
            spsa_reward_gradient = None
            spsa_stats = None
            apply_spsa_guidance = use_spsa_reward and should_apply_spsa_reward(args, i, total_steps)
            
            if apply_spsa_guidance:
                try:
                    _sigma = sigma_schedule[i]
                    temp_pred_original = z.to(torch.float32) - _sigma * model_pred.to(torch.float32)
                    
                    spsa_reward_gradient, spsa_stats = compute_spsa_reward_gradient(
                        pred_original_sample=temp_pred_original,
                        vae=vae,
                        video_processor=video_processor_for_spsa,
                        reward_model=reward_model,
                        caption=caption,
                        args=args,
                        step=step,
                        batch_index=batch_index,
                        sample_index=i,
                        sigma=spsa_sigma,
                        num_samples=spsa_num_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        transformer_config=transformer.config,
                    )
                    
                    all_spsa_stats.append(spsa_stats)
                    sampling_stats["spsa_reward_stats"].append({
                        "step": i,
                        "r_plus_mean": spsa_stats["r_plus_mean"],
                        "r_minus_mean": spsa_stats["r_minus_mean"],
                        "gradient_norm": spsa_stats["gradient_norm"],
                    })
                    
                except Exception as e:
                    main_print(f"[SPSA Reward] Step {i} computation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    spsa_reward_gradient = None
                    spsa_stats = None

            # Call flux_step
            flux_step_res = flux_step(
                model_pred,
                z.to(torch.float32),
                args.eta,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
                grpo=True,
                sde_solver=sde_solver,
                use_reward_gradient=apply_reward_guidance,
                reward_gradient=reward_gradient,
                reward_gradient_scale=process_reward_guidance_scale if use_process_reward else 0.0,
                normalize_reward_gradient=normalize_reward_gradient,
                reward_gradient_kl_beta=reward_gradient_kl_beta,
                use_spsa_reward_gradient=apply_spsa_guidance,
                spsa_reward_gradient=spsa_reward_gradient,
                spsa_gradient_scale=spsa_guidance_scale if use_spsa_reward else 0.0,
                normalize_spsa_gradient=spsa_normalize_gradient,
                spsa_gradient_kl_beta=spsa_gradient_kl_beta if use_spsa_reward else 0.1,
                use_reward_guided_mean_for_logprob=getattr(args, 'use_reward_guided_mean_for_logprob', True),
                use_delta_t_for_gradient_scaling=getattr(args, 'use_delta_t_for_gradient_scaling', False),
            )
            assert len(flux_step_res) == 3, "flux_step must return 3 values for GRPO sampling"
            z, pred_original, log_prob = flux_step_res

            z.to(torch.bfloat16)

            step_time = time.time() - step_start_time
            sampling_stats["step_times"].append(step_time)
            sampling_stats["model_pred_stats"].append({
                "mean": model_pred.mean().item(),
                "std": model_pred.std().item(),
                "max": model_pred.max().item(),
                "min": model_pred.min().item(),
            })
            sampling_stats["log_prob_stats"].append({
                "mean": log_prob.mean().item(),
                "std": log_prob.std().item(),
            })
            
            if process_reward_score is not None:
                sampling_stats["process_reward_stats"].append({
                    "step": i,
                    "score": process_reward_score,
                })

            postfix_dict = {
                "sigma": f"{sigma:.3f}",
                "step_time": f"{step_time:.3f}s",
                "log_prob": f"{log_prob.mean().item():.3f}",
                "pred_norm": f"{model_pred.norm().item():.3f}",
            }
            if process_reward_score is not None:
                postfix_dict["proc_rew"] = f"{process_reward_score:.4f}"
            if spsa_stats is not None:
                postfix_dict["spsa_r+"] = f"{spsa_stats['r_plus_mean']:.4f}"
                postfix_dict["spsa_r-"] = f"{spsa_stats['r_minus_mean']:.4f}"
            progress_bar.set_postfix(postfix_dict)

            all_latents.append(z)
            all_log_probs.append(log_prob)
            all_pred_original.append(pred_original)

        latents = pred_original.to(torch.float32) / 0.476986
        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)
        all_pred_original = torch.stack(all_pred_original, dim=1)
        
        # Only aggregate process rewards if enabled (all elements are now valid tensors, so no need for None checks).
        if use_process_reward and all_process_rewards:
            # Each step is appended to this list, then stacked on the steps dimension, hence final shape = [batch_size, num_steps].
            all_process_rewards_tensor = torch.stack(all_process_rewards, dim=1)
        else:
            all_process_rewards_tensor = None
        
        # Output timing statistical summary for the sampling phase.
        if sampling_stats["step_times"] and use_process_reward:
            avg_step_time = sum(sampling_stats["step_times"]) / len(sampling_stats["step_times"])
            avg_rgg_time = sum(sampling_stats["rgg_times"]) / len(sampling_stats["rgg_times"]) if sampling_stats["rgg_times"] else 0.0
            rgg_ratio = (avg_rgg_time / avg_step_time * 100) if avg_step_time > 0 else 0.0
            main_print(f"[Timing] Sampling: avg_step_time={avg_step_time:.4f}s, avg_rgg_time={avg_rgg_time:.4f}s, rgg_ratio={rgg_ratio:.1f}%")

        return z, latents, all_latents, all_log_probs, all_pred_original, all_process_rewards_tensor


def sample_reference_model(
    args,
    step,
    device,
    transformer,
    vae,
    encoder_hidden_states,
    encoder_attention_mask,
    reward_model,
    caption,
):
    """Sample from the reference model and compute rewards."""
    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps

    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert len(sigma_schedule) == sample_steps + 1, "sigma_schedule must have length sample_steps + 1"

    B = encoder_hidden_states.shape[0]

    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 4
    IN_CHANNELS = 16

    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    batch_progress = tqdm(
        enumerate(batch_indices),
        total=len(batch_indices),
        desc="Generating multiple videos for each prompt",
        disable=dist.get_rank() > 0,
    )

    video_time_total = 0.0
    reward_time_total = 0.0
    video_time_count = 0
    reward_time_count = 0

    all_latents = []
    all_log_probs = []
    all_pred_original = []
    all_rewards = {}
    all_process_rewards_list = []
    videos = None

    use_same_noise = args.use_same_noise

    if use_same_noise:
        input_latents = torch.randn(
            (1, IN_CHANNELS, latent_t, latent_h, latent_w),
            device=device,
            dtype=torch.bfloat16,
        )

    for index, batch_idx in batch_progress:
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_encoder_attention_mask = encoder_attention_mask[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]

        if not use_same_noise:
            input_latents = torch.randn(
                (len(batch_idx), IN_CHANNELS, latent_t, latent_h, latent_w),
                device=device,
                dtype=torch.bfloat16,
            )

        grpo_sample = True
        progress_bar = tqdm(range(0, sample_steps), desc=f"Sampling Batch {index+1}", disable=True)

        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs, batch_pred_original, batch_process_rewards = run_sample_step(
                args, input_latents.clone(), progress_bar, sigma_schedule,
                transformer, batch_encoder_hidden_states, batch_encoder_attention_mask,
                grpo_sample, reward_model, vae, batch_caption[0], step, index,
            )

        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_pred_original.append(batch_pred_original)
        if batch_process_rewards is not None:
            all_process_rewards_list.append(batch_process_rewards)

        video_generation_start = time.time()

        vae.enable_tiling()
        video_processor = VideoProcessor(vae_scale_factor=8)

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = vae.decode(latents, return_dict=False)[0]
                videos = video_processor.postprocess_video(video)

        rank = int(os.environ["RANK"])

        video_dir = os.path.join(args.output_dir, "videos")
        video_meta_dir = os.path.join(args.output_dir, "video_metadata")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(video_meta_dir, exist_ok=True)

        video_filename = f"step{step:06d}_rank{rank}_batch{index}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        export_to_video(videos[0], video_path, fps=args.fps)

        video_meta = {
            "caption": batch_caption[0],
            "video_path": video_path,
        }
        video_meta_path = os.path.join(video_meta_dir, f"step{step:06d}_rank{rank}_batch{index}.json")
        with open(video_meta_path, "w", encoding="utf-8") as f:
            json.dump(video_meta, f, ensure_ascii=False, indent=4)

        video_generation_time = time.time() - video_generation_start
        video_time_total += video_generation_time
        video_time_count += 1

        absolute_path = os.path.abspath(video_path)

        reward_compute_start = time.time()
        
        pooled_projection_dim = transformer.config.pooled_projection_dim

        try:
            with torch.no_grad():
                num_timesteps = batch_pred_original.shape[1]
                eval_timestep_idx = num_timesteps - 1
                eval_timestep = 0
                
                pooled_projections = batch_encoder_hidden_states[:, 0, :pooled_projection_dim].to(torch.bfloat16)
                prompt_embeds = batch_encoder_hidden_states[:, 1:].to(torch.bfloat16)
                
                latent_data = {
                    'noisy_latents': batch_pred_original[:, eval_timestep_idx].to(torch.bfloat16),
                    'timestep': torch.full((batch_pred_original.shape[0],), eval_timestep, device=batch_pred_original.device, dtype=torch.long),
                    'prompt_embeds': prompt_embeds,
                    'prompt_attention_mask': batch_encoder_attention_mask,
                    'pooled_prompt_embeds': pooled_projections,
                }
                
                reward_output: RewardOutput = reward_model.compute_rewards(
                    [absolute_path], [batch_caption[0]], latent_data=latent_data,
                )

            reward_compute_time = time.time() - reward_compute_start
            reward_time_total += reward_compute_time
            reward_time_count += 1

            if dist.get_rank() == 0:
                log_entry = {
                    "video_path": absolute_path,
                    "prompt": batch_caption[0],
                    "rewards": {
                        reward_name: float(reward_values.item() if reward_values.dim() == 0 else reward_values.mean().item())
                        for reward_name, reward_values in reward_output.rewards.items()
                    },
                    "timestamp": time.time(),
                }

                video_scores_log_path = os.path.join(args.output_dir, "video_scores_log.jsonl")
                with open(video_scores_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            for reward_name, reward_values in reward_output.rewards.items():
                if reward_name not in all_rewards:
                    all_rewards[reward_name] = []
                all_rewards[reward_name].append(
                    reward_values.unsqueeze(0) if reward_values.dim() == 0 else reward_values
                )

        except Exception as e:
            reward_compute_time = time.time() - reward_compute_start
            reward_time_total += reward_compute_time
            reward_time_count += 1
            main_print(f"❌ Batch {index+1} reward calculation failed: {type(e).__name__}: {str(e)}")
            
            default_reward = torch.tensor(-1.0).to(device)

            if getattr(args, "reward_type", None) == "trl":
                if "score" not in all_rewards:
                    all_rewards["score"] = []
                all_rewards["score"].append(default_reward.unsqueeze(0))
            else:
                for reward_name in ["VQ", "MQ"]:
                    if reward_name not in all_rewards:
                        all_rewards[reward_name] = []
                    all_rewards[reward_name].append(default_reward.unsqueeze(0))

        avg_video_time = video_time_total / video_time_count if video_time_count else 0.0
        avg_reward_time = reward_time_total / reward_time_count if reward_time_count else 0.0
        batch_progress.set_postfix({
            "avg_video_s": f"{avg_video_time:.2f}",
            "avg_reward_s": f"{avg_reward_time:.2f}",
        })

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_pred_original = torch.cat(all_pred_original, dim=0)
    
    all_process_rewards = None
    if all_process_rewards_list and len(all_process_rewards_list) == len(batch_indices):
        all_process_rewards = torch.cat(all_process_rewards_list, dim=0)

    for reward_name in all_rewards:
        all_rewards[reward_name] = torch.cat(all_rewards[reward_name], dim=0)

    if videos is None:
        raise RuntimeError("No videos were generated during sampling")

    return videos, z, all_rewards, all_latents, all_log_probs, sigma_schedule, all_pred_original, all_process_rewards

