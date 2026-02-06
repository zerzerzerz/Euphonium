"""Utility functions for GRPO sampling."""

from __future__ import annotations


def should_apply_process_reward(args, step_idx: int, total_steps: int) -> bool:
    """
    Determine whether the Process Reward should be applied at the current step.
    
    Args:
        args: Parameter object containing process reward configuration.
        step_idx: Current step index (0-based).
        total_steps: Total number of steps.
        
    Returns:
        bool: Whether to apply the process reward.
    """
    if not getattr(args, 'process_latent_reward_enabled', False):
        return False
    
    start_step = getattr(args, 'process_latent_reward_start_step', 0)
    end_step = getattr(args, 'process_latent_reward_end_step', -1)
    interval = getattr(args, 'process_latent_reward_interval', 1)
    
    # Handle cases where end_step is -1 (meaning until the last step)
    if end_step < 0:
        end_step = total_steps
    
    # Check if within the valid range
    if step_idx < start_step or step_idx >= end_step:
        return False
    
    # Check if the interval condition is met
    if (step_idx - start_step) % interval != 0:
        return False
    
    return True


def should_apply_spsa_reward(args, step_idx: int, total_steps: int) -> bool:
    """
    Determine whether SPSA Reward Gradient Guidance should be applied at the current step.
    
    SPSA (Simultaneous Perturbation Stochastic Approximation) is a zero-order optimization method
    that estimates gradients by perturbing `pred_original_sample` and computing rewards in RGB space.
    
    Args:
        args: Parameter object containing SPSA reward configuration.
        step_idx: Current step index (0-based).
        total_steps: Total number of steps.
        
    Returns:
        bool: Whether to apply the SPSA reward gradient.
    """
    if not getattr(args, 'spsa_reward_enabled', False):
        return False
    
    start_step = getattr(args, 'spsa_reward_start_step', 0)
    end_step = getattr(args, 'spsa_reward_end_step', -1)
    interval = getattr(args, 'spsa_reward_interval', 1)
    
    # Handle cases where end_step is -1 (meaning until the last step)
    if end_step < 0:
        end_step = total_steps
    
    # Check if within the valid range
    if step_idx < start_step or step_idx >= end_step:
        return False
    
    # Check if the interval condition is met
    if (step_idx - start_step) % interval != 0:
        return False
    
    return True
