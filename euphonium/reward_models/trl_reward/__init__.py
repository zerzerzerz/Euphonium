"""
TRL Reward Model Package

Splits the TRL reward model into multiple sub-modules:
- TRLCoreReward: TRL core reward (based on InternVL/QwenVL, etc.)
- VideoAlignSubReward: VideoAlign sub-reward (VQ/MQ/TA)
- LatentSubReward: Latent space sub-reward

All sub-reward models are eventually integrated into TRLRewardModel.
"""

from .base_sub_reward import BaseSubReward
from .trl_core_reward import TRLCoreReward
from .video_align_sub_reward import VideoAlignSubReward
from .latent_sub_reward import LatentSubReward
from .trl_reward_model import TRLRewardModel

__all__ = [
    'BaseSubReward',
    'TRLCoreReward',
    'VideoAlignSubReward',
    'LatentSubReward',
    'TRLRewardModel',
]
