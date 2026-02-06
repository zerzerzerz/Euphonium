"""Validation and checkpoint configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationArguments:
    """Validation and logging configuration"""
    
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed for reproducibility"}
    )
    
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for models and checkpoints"}
    )


@dataclass
class CheckpointArguments:
    """Checkpoint saving and recovery configuration"""
    
    checkpointing_steps: int = field(
        default=500,
        metadata={"help": "Save a checkpoint every X steps"}
    )
