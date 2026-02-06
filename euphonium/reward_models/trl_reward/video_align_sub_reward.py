"""
VideoAlign Sub-reward Model

VideoAlign reward computation integrated into TRLRewardModel.
"""

import os
import torch
from typing import Dict, List, Any, Tuple, Optional

from .base_sub_reward import BaseSubReward


class VideoAlignSubReward(BaseSubReward):
    """
    VideoAlign Sub-reward Model.
    
    Computes rewards across three dimensions: VQ (Visual Quality), MQ (Motion Quality), and TA (Text Alignment).
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        super().__init__(config, device)
        
        # Parse configuration
        self._parse_config()
        
        # Inferencer (deferred initialization)
        self.inferencer = None
        self.dtype = None
    
    def _parse_config(self) -> None:
        """Parse configuration related to VideoAlign."""
        self._enabled = self._get_config_value(
            "video_align_in_trl_reward_enabled", False, bool
        )
        
        self.use_norm = self._get_config_value(
            "video_align_in_trl_reward_use_norm", True, bool
        )
        
        self.vq_coef = self._get_config_value(
            "video_align_in_trl_reward_vq_coef", 0.5, float
        )
        
        self.mq_coef = self._get_config_value(
            "video_align_in_trl_reward_mq_coef", 0.5, float
        )
        
        self.ta_coef = self._get_config_value(
            "video_align_in_trl_reward_ta_coef", 0.0, float
        )
        
        self.pretrained_path = self._get_config_value(
            "video_align_in_trl_reward_pretrained_path", "", str
        )
    
    def _resolve_dtype(self) -> torch.dtype:
        """Resolve data type."""
        dtype_str = str(self.config.get('dtype', 'bf16')).lower()
        if dtype_str == 'fp16':
            return torch.float16
        if dtype_str == 'fp32':
            return torch.float32
        return torch.bfloat16
    
    def initialize(self) -> None:
        """Initialize VideoAlign inferencer."""
        if not self._enabled:
            return
            
        if not self.pretrained_path:
            raise ValueError(
                "video_align_in_trl_reward_enabled is enabled, but video_align_in_trl_reward_pretrained_path was not provided"
            )
            
        if not os.path.exists(self.pretrained_path):
            raise FileNotFoundError(
                f"VideoAlign pretrained model path not found: {self.pretrained_path}"
            )
        
        self.dtype = self._resolve_dtype()
        
        print(f"[VideoAlignSubReward] Loading model: {self.pretrained_path}")
        
        try:
            from .videoalign.inference import VideoVLMRewardInference
            
            self.inferencer = VideoVLMRewardInference(
                self.pretrained_path,
                device=self.device,
                dtype=self.dtype,
            )
            print(f"[VideoAlignSubReward] Model loading complete")
            print(f"[VideoAlignSubReward] Initialization successful")
            print(f"  - pretrained_path: {self.pretrained_path}")
            print(f"  - dtype: {self.dtype}")
            print(f"  - vq_coef: {self.vq_coef}, mq_coef: {self.mq_coef}, ta_coef: {self.ta_coef}")
            
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize VideoAlign reward inferencer, please check model path and dependencies"
            ) from exc
    
    def compute(self, 
                video_paths: List[str], 
                captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VideoAlign rewards.
        
        Args:
            video_paths: A list of paths to video files.
            captions: A list of corresponding text descriptions.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (VQ, MQ, TA) reward tensors.
        """
        if self.inferencer is None:
            raise RuntimeError("VideoAlign reward inferencer has not been initialized")
        
        raw_rewards = self.inferencer.reward(
            video_paths,
            captions,
            use_norm=self.use_norm,
        )
        
        vq_scores = []
        mq_scores = []
        ta_scores = []
        for reward in raw_rewards:
            vq_scores.append(float(reward.get('VQ', 0.0)))
            mq_scores.append(float(reward.get('MQ', 0.0)))
            ta_scores.append(float(reward.get('TA', 0.0)))
        
        vq_tensor = torch.tensor(vq_scores, device=self.device, dtype=torch.float32)
        mq_tensor = torch.tensor(mq_scores, device=self.device, dtype=torch.float32)
        ta_tensor = torch.tensor(ta_scores, device=self.device, dtype=torch.float32)
        
        return vq_tensor, mq_tensor, ta_tensor
    
    def compute_weighted_score(self, 
                               video_paths: List[str], 
                               captions: List[str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute a weighted composite score.
        
        Args:
            video_paths: A list of paths to video files.
            captions: A list of corresponding text descriptions.
            
        Returns:
            Tuple[torch.Tensor, Dict]: (Weighted score, dictionary of scores for each dimension).
        """
        vq, mq, ta = self.compute(video_paths, captions)
        
        weighted_score = (
            self.vq_coef * vq
            + self.mq_coef * mq
            + self.ta_coef * ta
        )
        
        scores_dict = {
            'VQ': vq,
            'MQ': mq,
            'TA': ta,
        }
        
        return weighted_score, scores_dict
