"""
KL Divergence computation for diffusion model GRPO training.

This module provides KL divergence computation between policy and reference models,
properly accounting for the SDE process in diffusion models.

Theoretical Background:
For the SDE process in diffusion models, we need to calculate the KL divergence between two policies.
In continuous time, the total KL divergence is an integral over time:
    KL_total = ∫_0^T KL(p_θ(x_t) || p_ref(x_t)) dt

After discretization, the KL contribution of each step needs to be multiplied by the step size (delta_t):
    KL_step = (μ_policy - μ_ref)² / (2σ²) · Δt

Where:
- σ is the noise level (sigma) at the current timestep.
- Δt is the timestep size (delta_t = sigma[i] - sigma[i+1]).

Physical Interpretation:
1. sigma²: When the noise is high, the model's prediction uncertainty is large, so the KL penalty should be small.
2. delta_t: A larger step size means this step accounts for a larger portion of the entire trajectory, so the KL contribution should be larger.
"""

from typing import Optional, Tuple
import torch


def compute_kl_divergence(
    policy_pred: torch.Tensor,
    ref_pred: torch.Tensor,
    sigma: float,
    delta_t: float,
    kl_type: str = "forward",
    eta: float = 1.0,
) -> torch.Tensor:
    """
    Compute the KL divergence between the prediction of the policy model and the reference model.
    
    For the SDE process in diffusion models, the KL divergence formula is:
    KL_step = ||μ_policy - μ_ref||² / (2 * eta² * sigma²) * delta_t
    
    Where `eta` is the noise coefficient of the SDE, which affects the variance of the process.
    
    Args:
        policy_pred: Velocity field prediction from the policy model (training model) [B, ...].
        ref_pred: Velocity field prediction from the reference model (pretrained model) [B, ...].
        sigma: Noise level corresponding to the current timestep (from sigma_schedule).
        delta_t: Current step size (sigma[i] - sigma[i+1]).
        kl_type: Type of KL divergence.
            - "forward": KL(policy || ref) - penalizes policy deviation from ref.
            - "reverse": KL(ref || policy) - penalizes ref deviation from policy.
            - "symmetric": (KL(policy||ref) + KL(ref||policy))/2 - Jensen-Shannon Divergence.
        eta: SDE noise coefficient, consistent with the `eta` parameter in `flux_step`.
    
    Returns:
        kl_loss: KL divergence loss (scalar).
    """
    # Calculate the square of the prediction difference
    diff_squared = (policy_pred - ref_pred).pow(2)
    
    # Prevent numerical issues
    sigma_sq = max(sigma ** 2, 1e-8)
    eta_sq = max(eta ** 2, 1e-8)
    delta_t = max(delta_t, 1e-8)
    
    # KL divergence formula: diff² / (2 * eta² * σ²) * Δt
    # - sigma²: Larger noise leads to smaller KL penalty (since predictions are inherently uncertain)
    # - eta²: Effect of SDE noise coefficient
    # - delta_t: Step size, used for integral approximation
    normalization = 2 * eta_sq * sigma_sq
    
    if kl_type == "forward":
        kl_loss = diff_squared.mean() * delta_t / normalization
    elif kl_type == "reverse":
        kl_loss = diff_squared.mean() * delta_t / normalization
    elif kl_type == "symmetric":
        kl_loss = diff_squared.mean() * delta_t / normalization
    else:
        raise ValueError(f"Unsupported KL type: {kl_type}. Supported: forward, reverse, symmetric")
    
    return kl_loss


def compute_kl_for_grpo_step(
    policy_model_pred: torch.Tensor,
    ref_model_pred: torch.Tensor,
    sigma_schedule: torch.Tensor,
    sigma_index: int,
    kl_type: str = "forward",
    eta: float = 1.0,
    kl_weight_version: str = "v2",
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the KL divergence for a single timestep in GRPO training.
    
    This is a convenience function that automatically extracts `sigma` and `delta_t` from `sigma_schedule`.
    
    Args:
        policy_model_pred: Prediction from the policy model.
        ref_model_pred: Prediction from the reference model.
        sigma_schedule: Sigma schedule table containing sigma values for all timesteps.
        sigma_index: Index of the current timestep in the sigma_schedule.
        kl_type: Type of KL divergence.
        eta: SDE noise coefficient.
        kl_weight_version: KL weight version ('v1' or 'v2').
        
    Returns:
        kl_loss: KL divergence loss.
        info: A dictionary containing debugging information.
    """
    # Get the current and next sigma from sigma_schedule
    sigma = float(sigma_schedule[sigma_index])
    
    # Calculate step size: delta_t = sigma[i] - sigma[i+1]
    if sigma_index + 1 < len(sigma_schedule):
        next_sigma = float(sigma_schedule[sigma_index + 1])
        delta_t = sigma - next_sigma  # sigma is decreasing, so delta_t > 0
    else:
        # Last step, use the current sigma as a step size estimate
        delta_t = sigma
    
    # Compute KL divergence
    kl_loss = compute_kl_divergence(
        policy_model_pred.to(torch.float32),
        ref_model_pred.to(torch.float32),
        sigma=sigma,
        delta_t=delta_t,
        kl_type=kl_type,
        eta=eta,
    )
    
    # Get the corresponding weight coefficient version
    if kl_weight_version == "v1":
        weight = compute_trajectory_kl_weight(sigma_schedule, sigma_index, eta=eta)
    elif kl_weight_version == "v2":
        weight = compute_trajectory_kl_weight_v2(sigma_schedule, sigma_index, eta=eta)
    else:
        raise ValueError(f"Unsupported kl_weight_version: {kl_weight_version}. Use 'v1' or 'v2'")
    
    # Return debugging information
    info = {
        "sigma": sigma,
        "delta_t": delta_t,
        "sigma_index": sigma_index,
        "kl_raw": kl_loss.item(),
        "weight_version": kl_weight_version,
        "weight": weight,
    }
    
    return kl_loss, info


def compute_trajectory_kl_weight(
    sigma_schedule: torch.Tensor,
    sigma_index: int,
    eta: float = 1.0,
) -> float:
    """
    Compute the KL weight factor for a given timestep.
    
    Weight = delta_t / (eta² * sigma²)
    
    This can be used to understand the relative KL contribution weights of different timesteps in advance.
    
    Args:
        sigma_schedule: Sigma schedule table.
        sigma_index: Current timestep index.
        eta: SDE noise coefficient.
        
    Returns:
        weight: KL weight factor.
    """
    sigma = float(sigma_schedule[sigma_index])
    
    if sigma_index + 1 < len(sigma_schedule):
        next_sigma = float(sigma_schedule[sigma_index + 1])
        delta_t = sigma - next_sigma
    else:
        delta_t = sigma
    
    sigma_sq = max(sigma ** 2, 1e-8)
    eta_sq = max(eta ** 2, 1e-8)
    delta_t = max(delta_t, 1e-8)
    
    weight = delta_t / (eta_sq * sigma_sq)
    
    return weight

def compute_trajectory_kl_weight_v2(
    sigma_schedule: torch.Tensor,
    sigma_index: int,
    eta: float = 1.0,
) -> float:
    """
    Compute the KL weight factor for a given timestep (Version 2).
    
    Uses a more complex formula:
    weight = (Δt / (2 * η²)) * (1 + η² * (1 - σ) / (2 * σ))²
    
    Args:
        sigma_schedule: Sigma schedule table.
        sigma_index: Current timestep index.
        eta: SDE noise coefficient.
        
    Returns:
        weight: KL weight factor.
    """
    sigma = float(sigma_schedule[sigma_index])
    
    if sigma_index + 1 < len(sigma_schedule):
        next_sigma = float(sigma_schedule[sigma_index + 1])
        delta_t = sigma - next_sigma
    else:
        delta_t = sigma
    
    sigma = max(sigma, 1e-8)
    eta_sq = max(eta ** 2, 1e-8)
    delta_t = max(delta_t, 1e-8)
    
    # weight = (Δt / (2 * η²)) * (1 + η² * (1 - σ) / (2 * σ))²
    term = 1.0 + eta_sq * (1.0 - sigma) / (2.0 * sigma)
    weight = (delta_t / (2.0 * eta_sq)) * (term ** 2)
    
    return weight