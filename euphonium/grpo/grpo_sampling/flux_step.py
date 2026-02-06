"""Flux step function for flow matching sampling."""

from __future__ import annotations

import math
from typing import Optional

import torch


def sd3_time_shift(shift, t):
    """Time shift function for SD3."""
    return (shift * t) / (1 + (shift - 1) * t)


def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: Optional[torch.Tensor],
    grpo: bool,
    sde_solver: bool,
    # Process Reward Guidance parameters (gradients based on Latent Reward Model)
    use_reward_gradient: bool = True,  # Whether to use reward gradient for guidance
    reward_gradient: Optional[torch.Tensor] = None,  # Gradient of reward with respect to latent
    reward_gradient_scale: float = 0.1,  # Scaling coefficient for reward gradient; <0 uses eta^2/(2*kl_beta)
    normalize_reward_gradient: bool = True,  # Whether to normalize the reward gradient
    reward_gradient_kl_beta: float = 0.1,  # KL divergence coefficient, used when scale < 0
    # SPSA Reward Gradient Guidance parameters (zero-order gradient estimation based on RGB space reward)
    use_spsa_reward_gradient: bool = False,  # Whether to use SPSA-estimated reward gradient
    spsa_reward_gradient: Optional[torch.Tensor] = None,  # SPSA-estimated reward gradient
    spsa_gradient_scale: float = 0.1,  # Scaling coefficient for SPSA gradient; <0 uses eta^2/(2*kl_beta)
    normalize_spsa_gradient: bool = True,  # Whether to normalize the SPSA gradient
    spsa_gradient_kl_beta: float = 0.1,  # KL divergence coefficient, used when scale < 0
    # Log Prob calculation control parameters
    use_reward_guided_mean_for_logprob: bool = True,  # Whether to use the mean incorporating reward gradient in log prob calculation
    use_delta_t_for_gradient_scaling: bool = False,  # Whether to multiply reward/SPSA gradient by delta t
):
    """
    Execute one flow matching sampling step.
    
    Args:
        latents: zt, the latents with a certain level of noise.
        model_output: The velocity field.
        sigmas: Noise intensity, actually linked to t.
        eta: SDE noise coefficient.
        index: Current step index.
        prev_sample: Samples produced from the data collection phase during training.
        grpo: Whether to use GRPO sampling.
        sde_solver: Whether to use an SDE solver.
        use_reward_gradient: Whether to use reward gradient for guidance (default: True).
        reward_gradient: Gradient of the reward with respect to the latent, same shape as latents.
        reward_gradient_scale: Scaling coefficient for the reward gradient; if >= 0, it's used directly; if < 0, eta^2/(2*kl_beta) is used.
        normalize_reward_gradient: Whether to normalize the reward gradient (default: True).
        reward_gradient_kl_beta: KL divergence coefficient beta; when scale < 0, actual scale = eta^2/(2*kl_beta).
        use_spsa_reward_gradient: Whether to use the SPSA-estimated reward gradient from RGB space.
        spsa_reward_gradient: Reward gradient estimated via SPSA, same shape as latents.
        spsa_gradient_scale: Scaling coefficient for the SPSA gradient.
        normalize_spsa_gradient: Whether to normalize the SPSA gradient.
        use_reward_guided_mean_for_logprob: Whether to use the mean incorporating reward gradient in log prob calculation.
            - True: Use the mean guided by the reward gradient.
            - False: Use the mean NOT guided by the reward gradient.
        use_delta_t_for_gradient_scaling: Whether to multiply reward/SPSA gradient by delta t.
    
    Returns: 
        prev_sample: z_{t-1}, the latents at the next timestep.
        pred_original_sample: Prediction of x0.
        (log_prob: If grpo=True, also returns log_prob)
    """
    sigma = sigmas[index] # Here sigma is actually t
    dsigma = sigmas[index + 1] - sigma # dsigma is negative here

    # 0 = clean
    # 1 = noise
    # latents = x t+1
    # model_output, x1 - x0, velocity is the vector from data to noise
    prev_sample_mean = latents + dsigma * model_output # model_output is the velocity field, from sample to noise; dsigma is negative, so this moves backwards, i.e., denoising from noise to sample.
    pred_original_sample = latents - sigma * model_output # current time = sigma, move back directly to original.

    delta_t = sigma - sigmas[index + 1] # in the sigma array, values decrease as index increases
    std_dev_t = eta * math.sqrt(delta_t) # delta_t > 0

    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    # Save the mean without reward gradient (used for log prob calculation)
    prev_sample_mean_without_reward = prev_sample_mean.clone()

    # Immediately apply reward gradient guidance after calculating prev_sample_mean
    # Helper function: Apply Reward Gradient Guidance (gradients from Latent Reward Model)
    def apply_reward_gradient_guidance(sample: torch.Tensor) -> torch.Tensor:
        if use_reward_gradient and reward_gradient is not None:
            grad_norm = reward_gradient.norm()
            if grad_norm > 0: # Prevent division by zero
                # Calculate actual scale
                if reward_gradient_scale >= 0:
                    actual_scale = reward_gradient_scale
                else:
                    # When scale < 0, use eta^2/(2*kl_beta) as scale
                    actual_scale = (eta ** 2) / (2 * reward_gradient_kl_beta)
                
                # Multiply by delta_t if delta_t scaling is enabled
                if use_delta_t_for_gradient_scaling:
                    actual_scale = actual_scale * delta_t
                
                # Normalize according to normalize_reward_gradient
                if normalize_reward_gradient:
                    normalized_gradient = reward_gradient / (grad_norm + 1e-8)
                    sample = sample + actual_scale * normalized_gradient.to(sample.dtype)
                else:
                    sample = sample + actual_scale * reward_gradient.to(sample.dtype)
        return sample

    # Helper function: Apply SPSA Reward Gradient Guidance (zero-order gradient estimation from RGB space reward)
    def apply_spsa_gradient_guidance(sample: torch.Tensor) -> torch.Tensor:
        if use_spsa_reward_gradient and spsa_reward_gradient is not None:
            grad_norm = spsa_reward_gradient.norm()
            if grad_norm > 0:  # Prevent division by zero
                # Calculate actual scale (same logic as LRM)
                if spsa_gradient_scale >= 0:
                    actual_scale = spsa_gradient_scale
                else:
                    # When scale < 0, use eta^2/(2*kl_beta) as scale
                    actual_scale = (eta ** 2) / (2 * spsa_gradient_kl_beta)
                
                # Multiply by delta_t if delta_t scaling is enabled
                if use_delta_t_for_gradient_scaling:
                    actual_scale = actual_scale * delta_t
                
                if normalize_spsa_gradient:
                    normalized_gradient = spsa_reward_gradient / (grad_norm + 1e-8)
                    sample = sample + actual_scale * normalized_gradient.to(sample.dtype)
                else:
                    sample = sample + actual_scale * spsa_reward_gradient.to(sample.dtype)
        return sample

    # Apply reward gradient guidance to prev_sample_mean
    # 1. First apply Latent Reward Model gradients (if enabled)
    prev_sample_mean = apply_reward_gradient_guidance(prev_sample_mean)
    # 2. Then apply SPSA-estimated RGB space reward gradients (if enabled)
    prev_sample_mean = apply_spsa_gradient_guidance(prev_sample_mean)

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if grpo:
        assert prev_sample is not None, "prev_sample must be provided for GRPO sampling"
        
        # Select which mean to use for log prob calculation based on use_reward_guided_mean_for_logprob
        if use_reward_guided_mean_for_logprob:
            # Use the mean incorporating reward gradient
            mean_for_logprob = prev_sample_mean
        else:
            # Use the mean WITHOUT reward gradient
            mean_for_logprob = prev_sample_mean_without_reward
        
        # Calculate logp
        log_prob = (
            -(
                (prev_sample.detach().to(torch.float32) - mean_for_logprob.to(torch.float32))
                ** 2
            )
            / (2 * (std_dev_t**2))
            - math.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample, pred_original_sample, log_prob

    # Non-GRPO mode
    return prev_sample_mean, pred_original_sample
