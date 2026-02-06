"""
Base classes and general data structures for reward models.
"""

import json
import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


def load_json(file_path: str):
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@dataclass
class RewardOutput:
    """
    Standardized format for reward model output.
    
    Supports multi-dimensional reward scoring for unified processing across different reward models.
    """
    # Main reward dimensions - corresponding to VQ/MQ or other custom dimensions in GRPO
    rewards: Dict[str, torch.Tensor]  # e.g., {"VQ": tensor([0.8]), "MQ": tensor([0.7])}
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = None  # Model-specific extra information


class BaseRewardModel(ABC):
    """
    Abstract base class for reward models.
    
    Defines a unified interface for reward computation. All specific reward models should
    inherit from this class and implement the `compute_rewards` method.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reward model.
        
        Args:
            config: A dictionary of model configuration parameters.
        """
        self.config = config
        self.device = config.get('device', 'cuda')
        
    @abstractmethod
    def compute_rewards(self, 
                       video_paths: List[str], 
                       captions: List[str]) -> RewardOutput:
        """
        Compute reward scores for video-text pairs.
        
        Args:
            video_paths: A list of paths to video files.
            captions: A list of corresponding text descriptions.
            
        Returns:
            RewardOutput: A standardized output containing reward scores for each dimension.
        """
        pass
    
    def __call__(self, video_paths: List[str], captions: List[str]) -> RewardOutput:
        """Convenience interface for calling the model."""
        return self.compute_rewards(video_paths, captions)


def create_reward_model(reward_type: str, config: Dict[str, Any]) -> BaseRewardModel:
    """
    Factory function: Creates the corresponding reward model based on the specified type.
    
    Args:
        reward_type: The type of reward model, currently supporting 'trl'.
        config: Model configuration parameters.
        
    Returns:
        BaseRewardModel: An instance of the corresponding reward model.
    """    
    # Deferred import to avoid circular dependencies
    from .trl_reward import TRLRewardModel
    
    if torch.distributed.is_initialized() and torch.distributed.get_rank() % 8 == 0:
        print("In function create_reward_model, config:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    return TRLRewardModel(config)

