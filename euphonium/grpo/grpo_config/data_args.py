"""Data-related configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    """Dataset and data loader configuration"""
    
    data_json_path: str = field(
        metadata={"help": "Path to the training data JSON file, containing text prompts and other information"}
    )
    
    dataloader_num_workers: int = field(
        default=10,
        metadata={"help": "Number of data loading subprocesses. 0 means data is loaded in the main process."}
    )
    
    train_batch_size: int = field(
        default=16,
        metadata={"help": "Training batch size per device"}
    )
    
    sampler_seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed for the data sampler"}
    )
