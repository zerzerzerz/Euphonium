"""GRPO one-step training function."""

from typing import Optional, Tuple, Union

import torch

from ..grpo_sampling import flux_step


def grpo_one_step(
    args,
    latents,
    pre_latents,
    encoder_hidden_states,
    encoder_attention_mask,
    transformer,
    timesteps,
    i,
    sigma_schedule,
    # Process Reward Guidance related parameters
    use_reward_gradient: bool = False,
    reward_gradient: Optional[torch.Tensor] = None,
    reward_gradient_scale: float = 0.1,
    normalize_reward_gradient: bool = True,
    reward_gradient_kl_beta: float = 0.1,
    # Log Prob calculation control parameters
    use_reward_guided_mean_for_logprob: bool = True,
    use_delta_t_for_gradient_scaling: bool = False,
    # KL divergence related parameters
    return_model_pred: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Execute one GRPO training step.
    
    Args:
        args: Training arguments.
        latents: Latents at the current timestep.
        pre_latents: Latents at the next timestep (used for log_prob calculation).
        encoder_hidden_states: Encoder hidden states.
        encoder_attention_mask: Encoder attention mask.
        transformer: Transformer model.
        timesteps: Timesteps.
        i: Index for sigma_schedule.
        sigma_schedule: Sigma schedule.
        use_reward_gradient: Whether to use reward gradient for guidance.
        reward_gradient: Gradient of the reward with respect to the latent.
        reward_gradient_scale: Scaling coefficient for the reward gradient; if >= 0, it is used directly; if < 0, eta^2/(2*kl_beta) is used.
        normalize_reward_gradient: Whether to normalize the reward gradient.
        reward_gradient_kl_beta: KL divergence coefficient beta.
        use_reward_guided_mean_for_logprob: Whether to use the mean incorporating reward gradient in log prob calculation.
        use_delta_t_for_gradient_scaling: Whether to multiply reward/SPSA gradient by delta t.
        return_model_pred: Whether to return the model prediction (used for KL divergence calculation).
        
    Returns:
        z: Updated latent.
        pred_original: Predicted original samples.
        log_prob: Log probability.
        model_pred (optional): Model prediction, if return_model_pred=True.
    """
    B = encoder_hidden_states.shape[0]

    with torch.autocast("cuda", torch.bfloat16):
        transformer.train()
        model_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            guidance=torch.tensor(
                [6018.0],
                device=latents.device,
                dtype=torch.bfloat16,
            ),
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

    z, pred_original, log_prob = flux_step(
        model_pred,
        latents.to(torch.float32),
        args.eta,
        sigma_schedule,
        i,
        prev_sample=pre_latents.to(torch.float32),
        grpo=True,
        sde_solver=True,
        # Process Reward Guidance parameters
        use_reward_gradient=use_reward_gradient,
        reward_gradient=reward_gradient,
        reward_gradient_scale=reward_gradient_scale,
        normalize_reward_gradient=normalize_reward_gradient,
        reward_gradient_kl_beta=reward_gradient_kl_beta,
        # Log Prob calculation control parameters
        use_reward_guided_mean_for_logprob=use_reward_guided_mean_for_logprob,
        use_delta_t_for_gradient_scaling=use_delta_t_for_gradient_scaling,
    )

    if return_model_pred:
        return z, pred_original, log_prob, model_pred
    return z, pred_original, log_prob


def get_ref_model_pred(
    latents,
    encoder_hidden_states,
    encoder_attention_mask,
    ref_transformer,
    timesteps,
):
    """
    Get prediction from reference model for KL divergence calculation.
    
    Args:
        latents: Latents at the current timestep.
        encoder_hidden_states: Encoder hidden states.
        encoder_attention_mask: Encoder attention mask.
        ref_transformer: Reference transformer model (frozen pretrained model).
        timesteps: Timesteps.
        
    Returns:
        ref_model_pred: Reference model prediction.
    """
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            ref_model_pred = ref_transformer(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                guidance=torch.tensor(
                    [6018.0],
                    device=latents.device,
                    dtype=torch.bfloat16,
                ),
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
    
    return ref_model_pred
