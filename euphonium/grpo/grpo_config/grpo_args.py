"""GRPO algorithm configuration."""

from dataclasses import dataclass, field


@dataclass
class GRPOArguments:
    """GRPO algorithm specific configuration"""
    
    use_group: bool = field(
        default=False,
        metadata={"help": "Whether to use group training"}
    )
    
    num_generations: int = field(
        default=16,
        metadata={"help": "Number of samples generated per prompt"}
    )
    
    use_same_noise: bool = field(
        default=False,
        metadata={"help": "Whether to use the same initial noise for all samples"}
    )
    
    timestep_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of timesteps used during training"}
    )
    
    bestofn: int = field(
        default=8,
        metadata={"help": "Number of samples for Best-of-N selection"}
    )
    
    timestep_selection_mode: str = field(
        default="fraction",
        metadata={"help": "Timestep selection mode: 'fraction' (proportional) or 'range' (by range)"}
    )
    
    timestep_range_start: int = field(
        default=0,
        metadata={"help": "range mode: Start step index (0-indexed, inclusive)"}
    )
    
    timestep_range_end: int = field(
        default=-1,
        metadata={"help": "range mode: End step index (-1 for the last step, inclusive)"}
    )
    
    timestep_range_interval: int = field(
        default=1,
        metadata={"help": "range mode: Step interval (1=every step, 2=every other step, etc.)"}
    )



@dataclass
class KLDivergenceArguments:
    """KL divergence regularization configuration"""
    
    use_kl_loss: bool = field(
        default=False,
        metadata={"help": "Whether to enable KL divergence loss regularization to prevent the trained model from deviating too far from the pretrained model"}
    )
    
    kl_coef: float = field(
        default=0.01,
        metadata={"help": "Weight coefficient for KL divergence loss"}
    )
    
    kl_type: str = field(
        default="forward",
        metadata={"help": "KL divergence type: 'forward' (KL(policy||ref)), 'reverse' (KL(ref||policy)), 'symmetric' (JSD)"}
    )
    
    kl_weight_version: str = field(
        default="v2",
        metadata={"help": "KL weight calculation version: 'v1' (Δt / η² · σ²) or 'v2' (Δt / 2η²) · (1 + η²(1-σ) / 2σ)²"}
    )



