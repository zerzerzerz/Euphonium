"""
Extensible Reward Model Architecture - Supports Reward Models trained via VideoAlign and TRL.

This module provides a unified reward model interface, supporting:
1. VideoAlign reward models (VQ + MQ + TA)
2. TRL-trained reward models (Single rating)
3. Latent Reward models (Latent space-based ratings)
4. Extensions for future reward models

Design Principles:
- Unified Interface: All reward models implement the same compute_rewards method.
- Configurability: Dynamic selection of reward model types via configuration parameters.
- Extensibility: New reward models only need to inherit from BaseRewardModel and implement compute_rewards.
- Backward Compatibility: Maintains full compatibility with existing GRPO training code.
"""

# Export base classes and data structures
from .base import (
    RewardOutput,
    BaseRewardModel,
    load_json,
)

# Export individual reward models
from .trl_reward import TRLRewardModel  # Import from trl_reward package

# Export factory function
from .base import create_reward_model

__all__ = [
    # Base classes and data structures
    'RewardOutput',
    'BaseRewardModel',
    'load_json',
    # Reward model classes
    'TRLRewardModel',
    # Factory function
    'create_reward_model',
]
