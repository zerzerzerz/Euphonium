"""Main training step function for GRPO."""

import json
import os
import time
import numpy as np

import torch
import torch.distributed as dist
from torch.nn import functional as F

from ...utils.logging_ import main_print
from ..grpo_sampling import sample_reference_model, should_apply_process_reward
from .grpo_step import grpo_one_step, get_ref_model_pred
from .utils import gather_tensor
from .kl_divergence import compute_kl_for_grpo_step
from tqdm.auto import tqdm


def train_one_step(
    args,
    device,
    transformer,
    vae,
    reward_model,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
    step,
    writer=None,
    ref_transformer=None,
):
    """
    Execute a complete GRPO training step.
    
    Includes:
    1. Data sampling and video generation.
    2. Reward computation.
    3. Advantage calculation.
    4. Best-of-N selection.
    5. Gradient calculation and parameter updates.
    6. KL divergence regularization (optional).
    
    Args:
        args: Training arguments.
        device: Training device.
        transformer: Transformer model.
        vae: VAE model.
        reward_model: Reward model.
        optimizer: Optimizer.
        lr_scheduler: Learning rate scheduler.
        loader: Data loader.
        max_grad_norm: Maximum gradient norm.
        step: Current training step.
        writer: TensorBoard writer (optional).
        ref_transformer: Reference model for KL divergence calculation (optional).
        
    Returns:
        total_loss: Total loss.
        grad_norm: Gradient norm.
        gathered_rewards: Collected rewards.
    """
    total_loss = 0.0
    total_kl_loss = 0.0  # Track KL loss
    optimizer.zero_grad()
    grad_norm = torch.tensor(0.0, device=device)
    
    # Training time statistics
    training_start_time = time.time()
    training_rgg_time_total = 0.0  # Total RGG time

    # Fetch data
    encoder_hidden_states, encoder_attention_mask, caption = next(loader)

    # Group expansion
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        encoder_attention_mask = repeat_tensor(encoder_attention_mask)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    # Sample and compute rewards
    videos, latents, all_rewards, all_latents, all_log_probs, sigma_schedule, all_pred_original, all_process_rewards = sample_reference_model(
        args, step, device, transformer, vae,
        encoder_hidden_states, encoder_attention_mask,
        reward_model, caption,
    )

    # Prepare training data
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps = torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[:, :-1][:, :-1],
        "next_latents": all_latents[:, 1:][:, :-1],
        "log_probs": all_log_probs[:, :-1],
        "pred_original": all_pred_original[:, :-1],
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
    }
    
    if all_process_rewards is not None:
        samples["process_rewards"] = all_process_rewards[:, :-1].to(torch.float32)

    for reward_name, reward_values in all_rewards.items():
        samples[f"{reward_name}_rewards"] = reward_values.to(torch.float32)

    # Gather rewards
    gathered_rewards = {}
    for reward_name in all_rewards.keys():
        gathered_rewards[reward_name] = gather_tensor(samples[f"{reward_name}_rewards"])

    if dist.get_rank() == 0:
        for reward_name, reward_values in gathered_rewards.items():
            reward_log_path = os.path.join(args.output_dir, f"{reward_name}_reward.txt")
            with open(reward_log_path, "a") as f:
                f.write(f"{reward_values.mean().item()}\n")

    reward_names = list(all_rewards.keys())
    if not reward_names:
        raise ValueError("No reward dimensions found, please check reward model configuration")

    reward_weights = {
        reward_name: getattr(args, f"{reward_name.lower()}_coef", 1.0)
        for reward_name in reward_names
    }

    # Calculate Outcome Reward Advantages (ORA)
    first_reward_key = f"{reward_names[0]}_rewards"
    n = len(samples[first_reward_key]) // args.num_generations

    group_statistics = []

    for reward_name in reward_names:
        reward_key = f"{reward_name}_rewards"
        advantage_key = f"{reward_name}_advantages"
        advantages = torch.zeros_like(samples[reward_key])

        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations

            group_rewards = samples[reward_key][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std

            if reward_name == reward_names[0]:
                group_stat = {
                    "prompt_index": i,
                    "sample_indices": list(range(start_idx, end_idx)),
                }
            else:
                group_stat = group_statistics[i]

            group_stat[f"{reward_name}_rewards"] = group_rewards.cpu().tolist()
            group_stat[f"{reward_name}_mean"] = group_mean.item()
            group_stat[f"{reward_name}_std"] = group_std.item()
            group_stat[f"{reward_name}_advantages"] = advantages[start_idx:end_idx].cpu().tolist()

            if reward_name == reward_names[0]:
                group_statistics.append(group_stat)

        samples[advantage_key] = advantages

    # Calculate Process Reward Advantages (PRA) if enabled
    process_reward_advantage_mode = getattr(args, 'process_reward_advantage_mode', 'none').lower()
    process_reward_advantage_coef = getattr(args, 'process_reward_advantage_coef', 1.0)
    outcome_reward_advantage_coef = getattr(args, 'outcome_reward_advantage_coef', 1.0)
    
    if process_reward_advantage_mode not in ['none', 'only', 'both', '0', '1', '2']:
        raise ValueError(f"Unsupported process_reward_advantage_mode: {process_reward_advantage_mode}")
    
    mode_map = {'0': 'none', '1': 'only', '2': 'both'}
    if process_reward_advantage_mode in mode_map:
        process_reward_advantage_mode = mode_map[process_reward_advantage_mode]
    
    if process_reward_advantage_mode != 'none' and "process_rewards" in samples:
        num_steps = samples["process_rewards"].shape[1]
        process_advantages = torch.zeros_like(samples["process_rewards"])
        
        for timestep_idx in range(num_steps):
            step_rewards = samples["process_rewards"][:, timestep_idx]
            
            for group_idx in range(n):
                start_idx = group_idx * args.num_generations
                end_idx = (group_idx + 1) * args.num_generations
                
                group_step_rewards = step_rewards[start_idx:end_idx]
                group_mean = group_step_rewards.mean()
                group_std = group_step_rewards.std() + 1e-8
                process_advantages[start_idx:end_idx, timestep_idx] = (group_step_rewards - group_mean) / group_std
        
        samples["process_advantages"] = process_advantages
        
        if dist.get_rank() == 0:
            main_print(f"[Process Reward Advantage] mode={process_reward_advantage_mode}, PRA_coef={process_reward_advantage_coef}, ORA_coef={outcome_reward_advantage_coef}")

    # Best-of-N selection
    total_scores = torch.zeros(len(samples[first_reward_key]), device=device)
    for reward_name in reward_names:
        advantage_key = f"{reward_name}_advantages"
        total_scores += reward_weights[reward_name] * samples[advantage_key]

    sorted_indices = torch.argsort(total_scores)
    top_indices = sorted_indices[-args.bestofn // 2 :]
    bottom_indices = sorted_indices[: args.bestofn // 2]
    selected_indices = torch.cat([top_indices, bottom_indices])
    shuffled_order = torch.randperm(len(selected_indices), device=selected_indices.device)
    selected_indices = selected_indices[shuffled_order]

    if args.num_generations != args.bestofn:
        for key in samples:
            samples[key] = samples[key][selected_indices]
        batch_size = len(selected_indices)

    # Shuffle timesteps and determine train_timesteps
    total_timesteps = len(samples["timesteps"][0])
    timestep_selection_mode = getattr(args, 'timestep_selection_mode', 'fraction')
    
    if timestep_selection_mode == 'range':
        # Range mode: use start/end/interval to select specific timestep indices
        start = getattr(args, 'timestep_range_start', 0)
        end = getattr(args, 'timestep_range_end', -1)
        end = total_timesteps if end < 0 else min(end + 1, total_timesteps)
        interval = getattr(args, 'timestep_range_interval', 1)
        
        # Generate selected timestep indices
        base_indices = torch.arange(start, end, interval, device=device)
        train_timesteps = len(base_indices)
        
        main_print(f"[Timestep Selection] mode=range, start={start}, end={end}, interval={interval}, train_timesteps={train_timesteps}")
        
        # Shuffle timesteps
        perms = torch.stack([
            base_indices[torch.randperm(len(base_indices), device=device)]
            for _ in range(batch_size)
        ])
    else:
        # Fraction mode (default): Original logic
        train_timesteps = int(total_timesteps * args.timestep_fraction)
        
        perms = torch.stack([
            torch.randperm(total_timesteps, device=device)
            for _ in range(batch_size)
        ])

        for key in ["timesteps", "latents", "next_latents", "log_probs"]:
            samples[key] = samples[key][
                torch.arange(batch_size, device=device)[:, None], perms,
            ]

    samples_batched = {k: v.unsqueeze(1) for k, v in samples.items()}
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]

    training_logs = []
    
    # Check if KL divergence regularization is enabled
    use_kl_loss = getattr(args, 'use_kl_loss', False) and ref_transformer is not None
    kl_coef = getattr(args, 'kl_coef', 0.01)
    kl_type = getattr(args, 'kl_type', 'forward')
    
    if use_kl_loss:
        main_print(f"[KL Loss] Enabled, coef={kl_coef}, type={kl_type}")
    
    # Check if process reward guidance during training is enabled
    use_training_process_reward = (
        reward_model is not None 
        and getattr(args, 'process_latent_reward_enabled', False)
        and getattr(args, 'process_latent_reward_training_enabled', False)
        and hasattr(reward_model, 'is_process_reward_enabled')
        and reward_model.is_process_reward_enabled()
    )
    process_reward_guidance_scale = getattr(args, 'process_latent_reward_guidance_scale', 0.1)
    normalize_reward_gradient = getattr(args, 'process_latent_reward_normalize_gradient', True)
    reward_gradient_kl_beta = getattr(args, 'process_latent_reward_kl_beta', 0.1)
    use_reward_guided_mean_for_logprob_training = getattr(args, 'use_reward_guided_mean_for_logprob_training', True)
    use_delta_t_for_gradient_scaling = getattr(args, 'use_delta_t_for_gradient_scaling', False)
    
    pooled_projection_dim = transformer.config.pooled_projection_dim if hasattr(transformer, 'config') else None

    total_samples = len(samples_batched_list)
    num_groups = total_samples // args.bestofn
    
    # GRPO mode: process each sample individually
    sample_pbar = tqdm(
        enumerate(samples_batched_list),
        total=total_samples,
        desc="Training Samples",
        disable=dist.get_rank() > 0,
    )
    for i, sample in sample_pbar:
        timestep_pbar = tqdm(
            range(train_timesteps),
            desc=f"Sample {i+1}/{total_samples} Timesteps",
            leave=False,
            disable=dist.get_rank() > 0,
        )
        for timestep_index in timestep_pbar:
            clip_range = 1e-4
            adv_clip_max = 5.0
            
            # Calculate reward gradient (if enabled)
            reward_gradient = None
            
            apply_reward_guidance = (
                use_training_process_reward 
                and should_apply_process_reward(args, int(perms[i][timestep_index].item()), args.sampling_steps)
            )
            
            if apply_reward_guidance and pooled_projection_dim is not None:
                try:
                    # Extract pooled_projections and actual prompt_embeds from encoder_hidden_states
                    pooled_projections_for_reward = sample["encoder_hidden_states"][:, 0, :pooled_projection_dim]
                    prompt_embeds_for_reward = sample["encoder_hidden_states"][:, 1:]
                    
                    # Compute process reward and gradient for the current step
                    rgg_start_time = time.time()  # Start RGG timing
                    _, reward_gradient = reward_model.compute_process_reward(
                        noisy_latents=sample["latents"][:, timestep_index].to(torch.bfloat16),
                        timestep=sample["timesteps"][:, timestep_index],
                        prompt_embeds=prompt_embeds_for_reward,
                        prompt_attention_mask=sample["encoder_attention_mask"],
                        pooled_prompt_embeds=pooled_projections_for_reward,
                        return_gradient=True,
                    )
                    training_rgg_time_total += time.time() - rgg_start_time  # End RGG timing
                except Exception as e:
                    main_print(f"[Training Process Reward] Computation failed: {e}")
                    reward_gradient = None
                    apply_reward_guidance = False

            # If KL loss is enabled, model prediction is needed
            if use_kl_loss:
                _, _, new_log_probs, policy_model_pred = grpo_one_step(
                    args,
                    sample["latents"][:, timestep_index],
                    sample["next_latents"][:, timestep_index],
                    sample["encoder_hidden_states"],
                    sample["encoder_attention_mask"],
                    transformer,
                    sample["timesteps"][:, timestep_index],
                    perms[i][timestep_index],
                    sigma_schedule,
                    # Process Reward Guidance parameters
                    use_reward_gradient=apply_reward_guidance,
                    reward_gradient=reward_gradient,
                    reward_gradient_scale=process_reward_guidance_scale,
                    normalize_reward_gradient=normalize_reward_gradient,
                    reward_gradient_kl_beta=reward_gradient_kl_beta,
                    # Log Prob calculation control parameters
                    use_reward_guided_mean_for_logprob=use_reward_guided_mean_for_logprob_training,
                    use_delta_t_for_gradient_scaling=use_delta_t_for_gradient_scaling,
                    # KL divergence parameters
                    return_model_pred=True,
                )
                
                # Get reference model prediction
                ref_model_pred = get_ref_model_pred(
                    sample["latents"][:, timestep_index],
                    sample["encoder_hidden_states"],
                    sample["encoder_attention_mask"],
                    ref_transformer,
                    sample["timesteps"][:, timestep_index],
                )
            else:
                _, _, new_log_probs = grpo_one_step(
                    args,
                    sample["latents"][:, timestep_index],
                    sample["next_latents"][:, timestep_index],
                    sample["encoder_hidden_states"],
                    sample["encoder_attention_mask"],
                    transformer,
                    sample["timesteps"][:, timestep_index],
                    perms[i][timestep_index],
                    sigma_schedule,
                    # Process Reward Guidance parameters
                    use_reward_gradient=apply_reward_guidance,
                    reward_gradient=reward_gradient,
                    reward_gradient_scale=process_reward_guidance_scale,
                    normalize_reward_gradient=normalize_reward_gradient,
                    reward_gradient_kl_beta=reward_gradient_kl_beta,
                    # Log Prob calculation control parameters
                    use_reward_guided_mean_for_logprob=use_reward_guided_mean_for_logprob_training,
                    use_delta_t_for_gradient_scaling=use_delta_t_for_gradient_scaling,
                )
                policy_model_pred = None
                ref_model_pred = None

            ratio = torch.exp(new_log_probs - sample["log_probs"][:, timestep_index])

            per_reward_losses = {}
            clamped_advantages = {}
            weighted_losses = []
            
            # Determine how to compute advantage based on process_reward_advantage_mode
            # mode: "none" - use ORA only; "only" - use PRA only; "both" - weighted combination of ORA and PRA
            has_process_adv = "process_advantages" in sample
            
            # ============ ORA (Outcome Reward Advantage) Calculation ============
            if process_reward_advantage_mode in ['none', 'both']:
                for reward_name in reward_names:
                    advantage_key = f"{reward_name}_advantages"

                    advantages = torch.clamp(
                        sample[advantage_key],
                        -adv_clip_max,
                        adv_clip_max,
                    )
                    clamped_advantages[f"ora_{reward_name}"] = advantages

                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - clip_range,
                        1.0 + clip_range,
                    )

                    per_reward_losses[f"ora_{reward_name}"] = (
                        torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        / (args.gradient_accumulation_steps * train_timesteps)
                    )

                    # Determine weight based on mode
                    if process_reward_advantage_mode == 'none':
                        # Use ORA only, with original reward_weights
                        weighted_losses.append(
                            reward_weights[reward_name] * per_reward_losses[f"ora_{reward_name}"]
                        )
                    else:
                        # "both" mode, use outcome_reward_advantage_coef
                        weighted_losses.append(
                            outcome_reward_advantage_coef * reward_weights[reward_name] * per_reward_losses[f"ora_{reward_name}"]
                        )
            
            # ============ PRA (Process Reward Advantage) Calculation ============
            if process_reward_advantage_mode in ['only', 'both'] and has_process_adv:
                # Use process reward advantage (per-step advantage)
                # Note: perms[i][timestep_index] is the shuffled timestep index
                original_timestep_idx = int(perms[i][timestep_index].item())
                process_adv = sample["process_advantages"][:, original_timestep_idx]
                
                clamped_process_adv = torch.clamp(
                    process_adv,
                    -adv_clip_max,
                    adv_clip_max,
                )
                clamped_advantages["pra"] = clamped_process_adv
                
                unclipped_loss = -clamped_process_adv * ratio
                clipped_loss = -clamped_process_adv * torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )
                
                per_reward_losses["pra"] = (
                    torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    / (args.gradient_accumulation_steps * train_timesteps)
                )
                
                weighted_losses.append(process_reward_advantage_coef * per_reward_losses["pra"])
            
            # If mode='only' but no process_advantages, fall back to ORA
            if process_reward_advantage_mode == 'only' and not has_process_adv:
                main_print(f"[Warning] mode='only' but no process_advantages, falling back to ORA")
                for reward_name in reward_names:
                    advantage_key = f"{reward_name}_advantages"
                    advantages = torch.clamp(sample[advantage_key], -adv_clip_max, adv_clip_max)
                    clamped_advantages[f"ora_{reward_name}"] = advantages
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    per_reward_losses[f"ora_{reward_name}"] = (
                        torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        / (args.gradient_accumulation_steps * train_timesteps)
                    )
                    weighted_losses.append(reward_weights[reward_name] * per_reward_losses[f"ora_{reward_name}"])
            
            # ============ KL Divergence Regularization ============
            kl_loss = torch.tensor(0.0, device=device)
            kl_info = {}
            if use_kl_loss and policy_model_pred is not None and ref_model_pred is not None:
                # Get the sigma index corresponding to the current timestep
                current_sigma_idx = int(perms[i][timestep_index].item())
                
                # Use the new SDE-aware KL calculation function
                kl_weight_version = getattr(args, 'kl_weight_version', 'v2')
                kl_loss, kl_info = compute_kl_for_grpo_step(
                    policy_model_pred=policy_model_pred.to(torch.float32),
                    ref_model_pred=ref_model_pred.to(torch.float32),
                    sigma_schedule=sigma_schedule,
                    sigma_index=current_sigma_idx,
                    kl_type=kl_type,
                    eta=args.eta if hasattr(args, 'eta') else 1.0,
                    kl_weight_version=kl_weight_version,
                )
                # Scale KL loss
                kl_loss = kl_loss * kl_coef / (args.gradient_accumulation_steps * train_timesteps)
                per_reward_losses["kl"] = kl_loss
                weighted_losses.append(kl_loss)
                
            final_loss = torch.stack(weighted_losses).sum()

            final_loss.backward()

            avg_loss = final_loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
            
            # Track KL loss
            if use_kl_loss:
                avg_kl_loss = kl_loss.detach().clone()
                dist.all_reduce(avg_kl_loss, op=dist.ReduceOp.AVG)
                total_kl_loss += avg_kl_loss.item()

            step_log = {
                "sample_index": i,
                "timestep_index": timestep_index,
                "timestep_value": sample["timesteps"][:, timestep_index].cpu().tolist(),
                "total_loss": final_loss.detach().cpu().item(),
                "ratio_mean": ratio.detach().mean().cpu().item(),
                "ratio_std": ratio.detach().std().cpu().item() if ratio.numel() > 1 else 0.0,
                "advantage_mode": process_reward_advantage_mode,
            }
            
            # Record KL loss
            if use_kl_loss:
                step_log["kl_loss"] = kl_loss.detach().cpu().item()

            # Record different information based on mode
            # ORA related logs
            if process_reward_advantage_mode in ['none', 'both']:
                for reward_name in reward_names:
                    ora_key = f"ora_{reward_name}"
                    if ora_key in per_reward_losses:
                        step_log[f"{reward_name}_ora_loss"] = per_reward_losses[ora_key].detach().cpu().item()
                        step_log[f"{reward_name}_reward"] = sample[f"{reward_name}_rewards"].detach().mean().cpu().item()
                        step_log[f"{reward_name}_ora_advantage"] = clamped_advantages[ora_key].detach().mean().cpu().item()
            
            # PRA related logs
            if process_reward_advantage_mode in ['only', 'both'] and "pra" in per_reward_losses:
                step_log["pra_loss"] = per_reward_losses["pra"].detach().cpu().item()
                step_log["pra_advantage"] = clamped_advantages["pra"].detach().mean().cpu().item()
                if "process_rewards" in sample:
                    original_timestep_idx = int(perms[i][timestep_index].item())
                    step_log["process_reward"] = sample["process_rewards"][:, original_timestep_idx].detach().mean().cpu().item()

            training_logs.append(step_log)
        
        # optimizer step for GRPO
        if (i + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        dist.barrier()

    if dist.get_rank() == 0:
        training_summary = {
            "step": step,
            "timestamp": time.time(),
            "group_statistics": group_statistics,
            "training_logs": training_logs,
        }
        
        # Add KL loss to training summary
        if use_kl_loss:
            training_summary["total_kl_loss"] = total_kl_loss

        log_file = os.path.join(args.output_dir, "training_details_log.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_summary, ensure_ascii=False) + "\n")

    # Construct returned extra information
    extra_info = {}
    if use_kl_loss:
        extra_info["kl_loss"] = total_kl_loss
    
    # Training time statistics
    training_total_time = time.time() - training_start_time
    rgg_ratio = (training_rgg_time_total / training_total_time * 100) if training_total_time > 0 else 0.0
    main_print(f"[Timing] Training: total_time={training_total_time:.4f}s, total_rgg_time={training_rgg_time_total:.4f}s, rgg_ratio={rgg_ratio:.1f}%")
    
    extra_info["timing/training_total_time"] = training_total_time
    extra_info["timing/training_rgg_time"] = training_rgg_time_total
    extra_info["timing/rgg_ratio"] = rgg_ratio

    return total_loss, grad_norm.item(), gathered_rewards, extra_info
