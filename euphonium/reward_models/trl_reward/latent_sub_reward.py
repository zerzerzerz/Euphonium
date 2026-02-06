"""
Latent Space Sub-reward Model

Latent reward computation integrated into TRLRewardModel.
"""

import os
import torch
from typing import Dict, List, Any, Optional

from .base_sub_reward import BaseSubReward


class LatentSubReward(BaseSubReward):
    """
    Latent Space Sub-reward Model.
    
    Computes reward scores directly in latent space without the need to decode videos.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        super().__init__(config, device)
        
        # Parse configuration
        self._parse_config()
        
        # Inferencer (deferred initialization)
        self.inferencer = None
    
    def _parse_config(self) -> None:
        """Parse configuration related to Latent Reward."""
        self._enabled = self._get_config_value(
            "latent_reward_in_trl_reward_enabled", False, bool
        )
        
        self.checkpoint_dir = self._get_config_value(
            "latent_reward_in_trl_reward_checkpoint_dir", "", str
        )
        
        self._coef = self._get_config_value(
            "latent_reward_in_trl_reward_coef", 1.0, float
        )
        
        self.dtype_str = self._get_config_value(
            "latent_reward_in_trl_reward_dtype", "bf16", str
        )
        
        # Parameters shared between Outcome and Process models
        self.use_sigmoid = self._get_config_value(
            "latent_reward_use_sigmoid", False, bool
        )
        
        # Path to Latent_PRM package
        self.latent_prm_path = self._get_config_value(
            "latent_prm_path", "", str
        )
    
    def _resolve_dtype(self) -> torch.dtype:
        """Resolve data type."""
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        return dtype_map.get(self.dtype_str.lower(), torch.bfloat16)

    def _import_module(self) -> None:
        """Add Latent_PRM package path to sys.path."""
        import sys
        # If the user specifies a path, use that path
        if self.latent_prm_path:
            latent_prm_path = os.path.abspath(self.latent_prm_path)
        else:
            # Default to relative path: from euphonium/reward_models/trl_reward/ back to third_party/Latent_PRM/ in project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            latent_prm_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "third_party", "Latent_PRM"))
        
        if latent_prm_path not in sys.path:
            sys.path.insert(0, latent_prm_path)
            print(f"[LatentSubReward] Adding Latent_PRM path to sys.path: {latent_prm_path}")
    
    def initialize(self) -> None:
        """Initialize Latent Reward Model."""
        if not self._enabled:
            return
        
        if not self.checkpoint_dir:
            raise ValueError(
                "latent_reward_in_trl_reward_enabled is enabled, but latent_reward_in_trl_reward_checkpoint_dir was not provided"
            )
        
        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(
                f"Latent Reward Model checkpoint directory not found: {self.checkpoint_dir}"
            )
        
        dtype = self._resolve_dtype()

        self._import_module()
        
        print(f"[LatentSubReward] Loading model: {self.checkpoint_dir}")
        
        try:
            from latent_prm.inferencer import RewardModelInferencer
            
            self.inferencer = RewardModelInferencer(
                checkpoint_dir=self.checkpoint_dir,
                device=self.device,
                dtype=dtype,
            )
            
            print(f"[LatentSubReward] Model loading complete")
            print(f"[LatentSubReward] Initialization successful")
            print(f"  - checkpoint_dir: {self.checkpoint_dir}")
            print(f"  - dtype: {self.dtype_str}")
            print(f"  - coef: {self._coef}")
            print(f"  - use_sigmoid: {self.use_sigmoid}")
            
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Latent Reward Model: {exc}"
            ) from exc
    
    def compute(self,
                noisy_latents: torch.Tensor,
                timestep: torch.Tensor,
                prompt_embeds: torch.Tensor,
                prompt_attention_mask: torch.Tensor,
                pooled_prompt_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Latent Reward score.
        
        Args:
            noisy_latents: (B, C, T, H, W) Noisy latents.
            timestep: (B,) Corresponding timesteps.
            prompt_embeds: (B, seq_len, hidden_dim) Prompt embeddings.
            prompt_attention_mask: (B, seq_len) Attention mask.
            pooled_prompt_embeds: (B, pooled_dim) Pooled prompt embeddings.
                If None, a zero vector will be used as a default value.
            
        Returns:
            torch.Tensor: (B,) Reward score for each sample.
        """
        if self.inferencer is None:
            raise RuntimeError("Latent Reward Model has not been initialized")
        
        batch_size = noisy_latents.shape[0]
        
        # Use zero vector if pooled_prompt_embeds is not provided
        if pooled_prompt_embeds is None:
            pooled_dim = self.inferencer.pooled_projection_dim
            pooled_prompt_embeds = torch.zeros(
                batch_size, pooled_dim,
                device=noisy_latents.device, dtype=noisy_latents.dtype
            )
            print(f"[LatentSubReward] WARNING: pooled_prompt_embeds not provided, using zero vector (dim={pooled_dim})")
        
        # Construct batch
        batch = {
            'noisy_latent': noisy_latents,
            'timestep': timestep,
            'prompt_embeds': prompt_embeds,
            'prompt_attention_mask': prompt_attention_mask,
            'pooled_prompt_embeds': pooled_prompt_embeds,
        }
        
        # Call inferencer using the use_sigmoid parameter from configuration
        scores, _ = self.inferencer.inference(
            batch,
            return_sigmoid=self.use_sigmoid,
            require_grad=False,
        )
        
        return scores

    def compute_with_gradient(self,
                              noisy_latents: torch.Tensor,
                              timestep: torch.Tensor,
                              prompt_embeds: torch.Tensor,
                              prompt_attention_mask: torch.Tensor,
                              pooled_prompt_embeds: Optional[torch.Tensor] = None,
                              return_sigmoid: Optional[bool] = None) -> tuple:
        """
        Compute Latent Reward score and return the gradient with respect to noisy_latents.
        
        Used for Process Reward Model to guide sampling during denoising using gradients.
        
        Args:
            noisy_latents: (B, C, T, H, W) Noisy latents.
            timestep: (B,) Corresponding timesteps.
            prompt_embeds: (B, seq_len, hidden_dim) Prompt embeddings.
            prompt_attention_mask: (B, seq_len) Attention mask.
            pooled_prompt_embeds: (B, pooled_dim) Pooled prompt embeddings.
                If None, a zero vector will be used as a default value.
            return_sigmoid: Whether to apply sigmoid to the output. If None, use `use_sigmoid` from configuration.
            
        Returns:
            tuple: (scores, gradient)
                - scores: (B,) Reward score for each sample.
                - gradient: (B, C, T, H, W) Gradient of the score with respect to noisy_latent.
        """
        if self.inferencer is None:
            raise RuntimeError("Latent Reward Model has not been initialized")
        
        batch_size = noisy_latents.shape[0]
        
        # Use zero vector if pooled_prompt_embeds is not provided
        if pooled_prompt_embeds is None:
            pooled_dim = self.inferencer.pooled_projection_dim
            pooled_prompt_embeds = torch.zeros(
                batch_size, pooled_dim,
                device=noisy_latents.device, dtype=noisy_latents.dtype
            )
        
        # Construct batch
        batch = {
            'noisy_latent': noisy_latents,
            'timestep': timestep,
            'prompt_embeds': prompt_embeds,
            'prompt_attention_mask': prompt_attention_mask,
            'pooled_prompt_embeds': pooled_prompt_embeds,
        }
        
        # Determine whether to use sigmoid: priority given to passed parameter, else use configuration
        use_sigmoid = return_sigmoid if return_sigmoid is not None else self.use_sigmoid
        
        # Call inferencer with require_grad=True to obtain gradients
        scores, gradient = self.inferencer.inference(
            batch,
            return_sigmoid=use_sigmoid,
            require_grad=True,
        )
        
        return scores, gradient
