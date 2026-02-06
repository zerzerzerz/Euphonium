"""
TRL Reward Model - Main Class

Integrates multiple sub-reward models such as TRL Core Reward, VideoAlign, and Latent Reward.
"""

import torch
from typing import Dict, List, Optional, Any

from ..base import BaseRewardModel, RewardOutput
from .trl_core_reward import TRLCoreReward
from .video_align_sub_reward import VideoAlignSubReward
from .latent_sub_reward import LatentSubReward


class TRLRewardModel(BaseRewardModel):
    """
    Adapter for TRL-trained reward models.
    
    Integrates reward models trained based on the TRL framework, supporting various model types such as InternVL and QwenVL.
    Also supports the integration of multiple sub-reward models including VideoAlign and Latent Reward.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize individual sub-reward models
        self.trl_core = TRLCoreReward(config, self.device)
        self.video_align = VideoAlignSubReward(config, self.device)
        self.latent_reward = LatentSubReward(config, self.device)
        
        # Initialize all sub-reward models
        self._initialize_all()
    
    def _initialize_all(self) -> None:
        """Initialize all sub-reward models."""
        # Initialize all sub-reward models (now all optional)
        self.trl_core.initialize()
        self.video_align.initialize()
        self.latent_reward.initialize()
        
        # Check that at least one reward model is enabled
        enabled_rewards = []
        if self.trl_core.enabled:
            enabled_rewards.append("TRL Core")
        if self.video_align.enabled:
            enabled_rewards.append("VideoAlign")
        if self.latent_reward.enabled:
            enabled_rewards.append("LatentReward")
        
        if not enabled_rewards:
            raise ValueError(
                "[TRLRewardModel] ERROR: At least one reward model must be enabled!\n"
                "Please check the configuration and ensure at least one of the following is enabled:\n"
                "  - TRL Core: set trl_core_enabled=True\n"
                "  - VideoAlign: set video_align_in_trl_reward_enabled=True\n"
                "  - LatentReward: set latent_reward_in_trl_reward_enabled=True"
            )
        
        # Print enabled status
        print(f"[TRLRewardModel] Sub-reward model status:")
        print(f"  - TRL Core: {'Enabled (coef=' + str(self.trl_core.coef) + ')' if self.trl_core.enabled else 'Disabled'}")
        print(f"  - VideoAlign: {'Enabled' if self.video_align.enabled else 'Disabled'}")
        print(f"  - LatentReward: {'Enabled' if self.latent_reward.enabled else 'Disabled'}")
        print(f"  Enabled reward models: {', '.join(enabled_rewards)}")
    
    @torch.no_grad()
    def compute_rewards(self, 
                       video_paths: List[str], 
                       captions: List[str],
                       latent_data: Optional[Dict[str, torch.Tensor]] = None) -> RewardOutput:
        """
        Compute rewards using models trained with TRL.
        
        Args:
            video_paths: A list of paths to video files.
            captions: A list of corresponding text descriptions (used for generating prompts).
            latent_data: Optional latent data dictionary for Latent Reward Model computation.
                - noisy_latents: (B, C, T, H, W) Noisy latents.
                - timestep: (B,) Corresponding timesteps.
                - prompt_embeds: (B, seq_len, hidden_dim) Prompt embeddings.
                - prompt_attention_mask: (B, seq_len) Attention mask.
                - pooled_prompt_embeds: (B, pooled_dim) Pooled prompt embeddings (optional).
            
        Returns:
            RewardOutput: Standardized output containing reward scores.
        """
        try:
            batch_size = len(video_paths)
            
            # Initialize combined_score
            combined_score = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            
            metadata: Dict[str, Any] = {
                'model_type': 'TRL',
            }
            
            # Initialize default values for scores in each dimension
            trl_scores = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            vq_tensor = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            mq_tensor = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            ta_tensor = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            latent_reward_tensor = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            
            # 1. Compute TRL core reward (if enabled)
            if self.trl_core.enabled:
                try:
                    trl_scores = self.trl_core.compute(video_paths, captions)
                    combined_score = combined_score + trl_scores * self.trl_core.coef
                    metadata['model_name'] = self.trl_core.model_type
                except Exception as e:
                    metadata['trl_core_error'] = str(e)
                    print(f"[TRLRewardModel] TRL Core reward computation failed: {e}")
            
            # 2. Compute VideoAlign reward
            if self.video_align.enabled:
                try:
                    weighted_va, va_scores = self.video_align.compute_weighted_score(video_paths, captions)
                    vq_tensor = va_scores['VQ']
                    mq_tensor = va_scores['MQ']
                    ta_tensor = va_scores['TA']
                    combined_score = combined_score + weighted_va
                    
                    metadata['components'] = metadata.get('components', {})
                    metadata['components']['video_align'] = {
                        'VQ': vq_tensor.detach().cpu().tolist(),
                        'MQ': mq_tensor.detach().cpu().tolist(),
                        'TA': ta_tensor.detach().cpu().tolist(),
                    }
                    metadata['components']['coefficients'] = metadata['components'].get('coefficients', {})
                    metadata['components']['coefficients']['VQ'] = self.video_align.vq_coef
                    metadata['components']['coefficients']['MQ'] = self.video_align.mq_coef
                    metadata['components']['coefficients']['TA'] = self.video_align.ta_coef
                    
                except Exception as e:
                    metadata['video_align_error'] = str(e)
                    print(f"[TRLRewardModel] VideoAlign reward computation failed: {e}")
            
            # 3. Compute Latent Reward
            if self.latent_reward.enabled:
                if latent_data is not None:
                    try:
                        # Validate that latent_data contains all required keys
                        required_keys = ['noisy_latents', 'timestep', 'prompt_embeds', 'prompt_attention_mask']
                        missing_keys = [k for k in required_keys if k not in latent_data]
                        if missing_keys:
                            raise ValueError(f"latent_data is missing required keys: {missing_keys}")
                        
                        latent_reward_tensor = self.latent_reward.compute(
                            noisy_latents=latent_data['noisy_latents'],
                            timestep=latent_data['timestep'],
                            prompt_embeds=latent_data['prompt_embeds'],
                            prompt_attention_mask=latent_data['prompt_attention_mask'],
                            pooled_prompt_embeds=latent_data.get('pooled_prompt_embeds', None),
                        )
                        combined_score = combined_score + self.latent_reward.coef * latent_reward_tensor
                        
                        metadata['components'] = metadata.get('components', {})
                        metadata['components']['latent_reward'] = {
                            'scores': latent_reward_tensor.detach().cpu().tolist(),
                        }
                        metadata['components']['coefficients'] = metadata['components'].get('coefficients', {})
                        metadata['components']['coefficients']['latent_reward'] = self.latent_reward.coef
                        
                    except Exception as e:
                        metadata['latent_reward_error'] = str(e)
                        print(f"[TRLRewardModel] Latent Reward computation failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("[TRLRewardModel] WARNING: Latent Reward is enabled but latent_data was not provided. Skipping computation.")
            
            # Update combined score and TRL score in metadata
            metadata['components'] = metadata.get('components', {})
            metadata['components']['trl_score'] = trl_scores.detach().cpu().tolist()
            metadata['components']['combined_score'] = combined_score.detach().cpu().tolist()
            
            # Build the return score dictionary
            score_dict = {
                "score": combined_score,
                "trl": trl_scores,
            }
            if self.video_align.enabled:
                score_dict["VQ"] = vq_tensor
                score_dict["MQ"] = mq_tensor
                score_dict["TA"] = ta_tensor
            if self.latent_reward.enabled:
                score_dict["latent_reward"] = latent_reward_tensor
            
            return RewardOutput(
                rewards=score_dict,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"TRLRewardModel compute_rewards error: {repr(e)}")
            import traceback
            traceback.print_exc()
            
            # Error handling: return default negative reward
            batch_size = len(video_paths)
            default_reward = torch.full((batch_size,), -1.0, device=self.device, dtype=torch.float32)
            
            return RewardOutput(
                rewards={'score': default_reward},
                metadata={'model_type': 'TRL', 'error': str(e)}
            )

    def compute_process_reward(self,
                               noisy_latents: torch.Tensor,
                               timestep: torch.Tensor,
                               prompt_embeds: torch.Tensor,
                               prompt_attention_mask: torch.Tensor,
                               pooled_prompt_embeds: Optional[torch.Tensor] = None,
                               return_gradient: bool = True) -> tuple:
        """
        Compute Process Latent Reward for guiding sampling during the denoising process.
        
        This method directly calls the Latent Reward Model to calculate the score and gradient of the current noisy_latent.
        The gradient can be used to adjust the sampling direction during denoising.
        
        Args:
            noisy_latents: (B, C, T, H, W) Current noisy latents.
            timestep: (B,) Corresponding timestep (or sigma value).
            prompt_embeds: (B, seq_len, hidden_dim) Prompt embeddings.
            prompt_attention_mask: (B, seq_len) Attention mask.
            pooled_prompt_embeds: (B, pooled_dim) Pooled prompt embeddings (optional).
            return_gradient: Whether to return the gradient.
            
        Returns:
            tuple: (scores, gradient)
                - scores: (B,) Reward score for each sample.
                - gradient: (B, C, T, H, W) Gradient of the score with respect to noisy_latent; None if return_gradient=False.
        """
        if not self.latent_reward.enabled:
            raise RuntimeError("Latent Reward Model is not enabled, cannot compute Process Reward")
        
        if return_gradient:
            scores, gradient = self.latent_reward.compute_with_gradient(
                noisy_latents=noisy_latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                pooled_prompt_embeds=pooled_prompt_embeds,
                # Use the use_sigmoid parameter from configuration
                return_sigmoid=self.latent_reward.use_sigmoid,
            )
            return scores, gradient
        else:
            scores = self.latent_reward.compute(
                noisy_latents=noisy_latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                pooled_prompt_embeds=pooled_prompt_embeds,
            )
            return scores, None
    
    def is_process_reward_enabled(self) -> bool:
        """Check if Process Reward is available."""
        return self.latent_reward.enabled
