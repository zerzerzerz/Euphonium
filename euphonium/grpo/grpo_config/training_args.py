"""Training-related configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    """Training hyperparameters configuration"""
    
    num_train_epochs: int = field(
        default=100,
        metadata={"help": "Total number of training epochs"}
    )
    
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum training steps; if specified, it overrides num_train_epochs"}
    )
    
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Initial learning rate"}
    )
    
    lr_warmup_steps: int = field(
        default=10,
        metadata={"help": "Number of learning rate warmup steps"}
    )
    
    max_grad_norm: float = field(
        default=2.0,
        metadata={"help": "Maximum gradient norm (for gradient clipping)"}
    )
    
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing to save GPU memory"}
    )
    
    selective_checkpointing: float = field(
        default=1.0,
        metadata={"help": "Selective checkpointing fraction"}
    )
    
    allow_tf32: bool = field(
        default=False,
        metadata={"help": "Whether to allow TF32 acceleration on Ampere GPUs"}
    )
    
    mixed_precision: Optional[str] = field(
        default=None,
        metadata={"help": "Mixed precision training: no, fp16, or bf16"}
    )
    
    lr_scheduler: str = field(
        default="constant_with_warmup",
        metadata={"help": "Type of learning rate scheduler"}
    )
    
    lr_num_cycles: int = field(
        default=1,
        metadata={"help": "Number of cycles for the learning rate scheduler"}
    )
    
    lr_power: float = field(
        default=1.0,
        metadata={"help": "Power for the polynomial scheduler"}
    )
    
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay coefficient"}
    )


@dataclass
class DistributedArguments:
    """Distributed training configuration"""
    
    use_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to offload parameters, gradients, and optimizer states to CPU"}
    )
    
    sp_size: int = field(
        default=1,
        metadata={"help": "Sequence parallel size"}
    )
    
    train_sp_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for sequence parallel training"}
    )
    
    fsdp_sharding_startegy: str = field(
        default="full",
        metadata={"help": "FSDP sharding strategy"}
    )
