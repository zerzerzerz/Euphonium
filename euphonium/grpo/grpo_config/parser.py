"""Argument parser for GRPO configuration."""

from dataclasses import dataclass
from typing import Dict, Any

from .constants import TRL_PROMPT_VARIANTS, DEFAULT_TRL_PROMPT_VARIANT, DEFAULT_TRL_GENERATION_PROMPT_PREFIX
from .model_args import ModelArguments
from .data_args import DataArguments
from .diffusion_args import DiffusionArguments
from .video_args import VideoArguments
from .validation_args import ValidationArguments, CheckpointArguments
from .training_args import TrainingArguments, DistributedArguments
from .grpo_args import GRPOArguments, KLDivergenceArguments
from .reward_args import (
    RewardModelArguments,
    VideoAlignInTRLRewardArguments,
    LatentRewardInTRLRewardArguments,
)


def parse_args():
    """
    Parse command line arguments.
    
    Uses HuggingFace's HfArgumentParser to parse multiple dataclass configurations.
    
    Returns:
        Unified argument object containing all attributes from the parsed dataclasses.
    """
    from transformers import HfArgumentParser
    
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        DiffusionArguments,
        VideoArguments,
        ValidationArguments,
        CheckpointArguments,
        TrainingArguments,
        DistributedArguments,
        GRPOArguments,
        KLDivergenceArguments,
        RewardModelArguments,
        VideoAlignInTRLRewardArguments,
        LatentRewardInTRLRewardArguments,
    ))
    
    args_tuple = parser.parse_args_into_dataclasses()
    
    @dataclass
    class UnifiedArgs:
        """Unified argument object, merging attributes from all dataclasses."""
        def __init__(self, *dataclass_instances):
            for instance in dataclass_instances:
                for key, value in instance.__dict__.items():
                    setattr(self, key, value)
    
    unified_args = UnifiedArgs(*args_tuple)
    
    return unified_args


def process_reward_args(args) -> None:
    """
    Process reward model relevant arguments and set default values for prompt variants.
    
    Modifies the args object directly by adding/updating the following attributes:
    - trl_system_prompt
    - trl_user_prompt
    - trl_include_generation_prompt (used as append_generation_prompt)
    - trl_generation_prompt_prefix
    - trl_prompt_variant (used as prompt_variant)
    - model_name_or_path (mapped from trl_model_path)
    - dtype (mapped from trl_dtype)
    - device (mapped from reward_device)
    
    Args:
        args: Argument object containing all configurations.
    """
    prompt_variant = getattr(args, "trl_prompt_variant", DEFAULT_TRL_PROMPT_VARIANT).lower()
    if prompt_variant not in TRL_PROMPT_VARIANTS:
        raise ValueError(f"Unsupported TRL prompt configuration: {prompt_variant}. Supported: {list(TRL_PROMPT_VARIANTS.keys())}")
    
    variant_defaults = TRL_PROMPT_VARIANTS[prompt_variant]
    
    system_prompt = args.trl_system_prompt or variant_defaults["system_prompt"]
    user_prompt = args.trl_user_prompt or variant_defaults["user_prompt"]
    
    include_generation_prompt = getattr(args, "trl_include_generation_prompt", variant_defaults["include_generation_prompt"])
    
    if prompt_variant == "ta":
        include_generation_prompt = True
    
    generation_prompt_prefix = getattr(args, "trl_generation_prompt_prefix", DEFAULT_TRL_GENERATION_PROMPT_PREFIX)
    
    # Update values in args
    args.trl_system_prompt = system_prompt
    args.trl_user_prompt = user_prompt
    args.trl_include_generation_prompt = include_generation_prompt
    args.trl_generation_prompt_prefix = generation_prompt_prefix
    args.trl_prompt_variant = prompt_variant
    
    # Set compatibility fields (field names used internally by the reward model)
    args.system_prompt = system_prompt
    args.user_prompt = user_prompt
    args.prompt_variant = prompt_variant
    args.append_generation_prompt = include_generation_prompt
    args.generation_prompt_prefix = generation_prompt_prefix
    args.model_name_or_path = args.trl_model_path
    args.dtype = args.trl_dtype
    args.device = args.reward_device


def create_reward_config_from_args(args) -> Dict[str, Any]:
    """
    Create a reward model configuration dictionary from the unified args object.
    
    Args:
        args: Argument object containing all configurations.
        
    Returns:
        Dict[str, Any]: Configuration dictionary that can be passed directly to create_reward_model.
    """
    # Process arguments first to ensure all compatibility fields are set
    process_reward_args(args)
    
    # Return args in dictionary form
    return args.__dict__
