"""Model-related configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """Model-related configuration."""
    
    model_type: str = field(
        default="hunyuan_hf",
        metadata={"help": "The type of model to train (supports hunyuan_hf, flux, etc.)"}
    )
    
    pretrained_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or HuggingFace model name."}
    )
    
    dit_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DiT model."}
    )
    
    vae_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to VAE model."}
    )
    
    cache_dir: str = field(
        default="./cache_dir",
        metadata={"help": "Directory for model cache."}
    )
    
    master_weight_type: str = field(
        default="fp32",
        metadata={"help": "Master weight type: fp32 or bf16."}
    )
