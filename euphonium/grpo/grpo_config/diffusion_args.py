"""Diffusion model configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiffusionArguments:
    """Diffusion model related configuration"""
    
    cfg: float = field(
        default=0.1,
        metadata={"help": "Classifier-free guidance scale"}
    )
    
    sampling_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of diffusion sampling steps"}
    )
    
    eta: Optional[float] = field(
        default=None,
        metadata={"help": "SDE noise intensity parameter"}
    )
    
    shift: float = field(
        default=1.0,
        metadata={"help": "Time shift parameter"}
    )
