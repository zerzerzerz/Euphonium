"""Reward model arguments configuration."""

from dataclasses import dataclass, field


@dataclass
class RewardModelArguments:
    """Reward model configuration"""
    
    # Basic reward model configuration
    reward_type: str = field(
        default="trl",
        metadata={"help": "Reward model type: trl"}
    )
    
    reward_device: str = field(
        default="cuda",
        metadata={"help": "Device to run the reward model on"}
    )
    
    # Reward weight configuration
    score_coef: float = field(
        default=1.0,
        metadata={"help": "Reward weight for TRL model"}
    )
    
    trl_core_enabled: bool = field(
        default=True,
        metadata={"help": "Whether to enable the TRL Core reward model"}
    )
    
    # TRL model configuration
    trl_model_path: str = field(
        default="",
        metadata={"help": "Path to the TRL model"}
    )
    
    trl_model_type: str = field(
        default="InternVL3",
        metadata={"help": "TRL model type: InternVL3, InternVL3-5, Qwen2.5-VL, Qwen2-VL"}
    )
    
    trl_tokenizer_path: str = field(
        default="",
        metadata={"help": "Path to the TRL tokenizer"}
    )
    
    trl_dtype: str = field(
        default="bf16",
        metadata={"help": "TRL model precision: bf16, fp16, fp32"}
    )
    
    trl_path: str = field(
        default="",
        metadata={"help": "Path to the SoliReward package for import. If empty, it's not added to sys.path"}
    )
    
    # TRL prompt configuration
    trl_system_prompt: str = field(
        default="你是一个专业的视频质量评估专家。请根据提供的视频内容，判断是否同时满足以下所有问题的合格标准：\n\n1. 物理规律是否合格？\n2. 是否存在人物或动物畸形？\n\n回答要求：\n- 只有当所有问题的答案都是\"合格\"时，才输出：good\n- 如果任何一个问题的答案是\"部分合格\"或\"不合格\"，则输出：bad\n- 不要输出任何其他内容\n- 答案要准确、客观\n",
        metadata={"help": "System prompt for TRL model"}
    )
    
    trl_user_prompt: str = field(
        default="请评估以下视频：",
        metadata={"help": "User prompt for TRL model"}
    )
    
    # TRL prompt variant configuration
    trl_prompt_variant: str = field(
        default="deformity_physics",
        metadata={"help": "TRL prompt variant: deformity_physics (deformity and physics laws) or ta (text alignment)"}
    )
    
    # Generation prompt configuration
    trl_include_generation_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to add the video generation prompt before the user prompt (automatically enabled for 'ta' variant)"}
    )
    
    trl_generation_prompt_prefix: str = field(
        default="生成视频的文本提示词是: ",
        metadata={"help": "Prefix for the generation prompt"}
    )
    
@dataclass
class VideoAlignInTRLRewardArguments:
    """Configuration for using VideoAlign in the TRL reward model"""
    
    video_align_in_trl_reward_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to enable VideoAlign in the TRL reward model"}
    )
    
    video_align_in_trl_reward_pretrained_path: str = field(
        default="/path/to/videoalign/pretrained/model/with/updated/keys",
        metadata={"help": "Pretrained weight path for the VideoAlign model used in TRL reward"}
    )
    
    video_align_in_trl_reward_use_norm: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the output when enabling VideoAlign in TRL reward"}
    )
    
    video_align_in_trl_reward_vq_coef: float = field(
        default=0.5,
        metadata={"help": "VQ weight when fusing VideoAlign in TRL reward"}
    )
    
    video_align_in_trl_reward_mq_coef: float = field(
        default=0.5,
        metadata={"help": "MQ weight when fusing VideoAlign in TRL reward"}
    )
    
    video_align_in_trl_reward_ta_coef: float = field(
        default=0.5,
        metadata={"help": "TA weight when fusing VideoAlign in TRL reward"}
    )


@dataclass
class LatentRewardInTRLRewardArguments:
    """Configuration for using Latent Reward Model in the TRL reward model"""
    
    latent_reward_in_trl_reward_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to enable Latent Reward Model in the TRL reward model"}
    )
    
    latent_reward_in_trl_reward_checkpoint_dir: str = field(
        default="",
        metadata={"help": "Path to the checkpoint directory of the Latent Reward Model, containing config.json and model.safetensors"}
    )
    
    latent_reward_in_trl_reward_coef: float = field(
        default=1.0,
        metadata={"help": "Weight coefficient for fusing Latent Reward in TRL reward"}
    )
    
    latent_reward_in_trl_reward_dtype: str = field(
        default="bf16",
        metadata={"help": "Data type for Latent Reward Model: bf16, fp16, fp32"}
    )
    
    latent_prm_path: str = field(
        default="",
        metadata={"help": "Path to the Latent_PRM package for import. If empty, it's not added to sys.path"}
    )
    
    latent_reward_use_sigmoid: bool = field(
        default=False,
        metadata={"help": "Whether to use sigmoid on the Latent Reward output (Outcome and Process share this parameter)"}
    )
    
    # Process Latent Reward configuration
    process_latent_reward_enabled: bool = field(
        default=False,
        metadata={"help": "Main switch to control whether to load the Process Latent Reward model"}
    )
    
    process_latent_reward_sampling_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to use reward gradient guidance during the data collection (sampling) process"}
    )
    
    process_latent_reward_training_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to use reward gradient guidance during the training process"}
    )
    
    process_latent_reward_guidance_scale: float = field(
        default=0.1,
        metadata={"help": "Guidance strength of the Process Reward gradient. Larger values mean the gradient has a greater impact on denoising."}
    )
    
    process_latent_reward_start_step: int = field(
        default=0,
        metadata={"help": "Which denoising step to start using Process Reward from"}
    )
    
    process_latent_reward_end_step: int = field(
        default=-1,
        metadata={"help": "Which denoising step to stop using Process Reward at. -1 means until the last step."}
    )
    
    process_latent_reward_interval: int = field(
        default=1,
        metadata={"help": "How many steps to skip between using Process Reward. 1 means every step."}
    )
    
    process_latent_reward_normalize_gradient: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the reward gradient. Default is True."}
    )
    
    process_latent_reward_kl_beta: float = field(
        default=0.1,
        metadata={"help": "KL divergence coefficient beta. When guidance_scale < 0, the actual scale = eta^2 / (2 * kl_beta)."}
    )
    
    use_delta_t_for_gradient_scaling: bool = field(
        default=False,
        metadata={"help": "Whether to multiply by delta t in reward/SPSA gradient"}
    )
    
    # Process Reward for Advantage configuration
    process_reward_advantage_mode: str = field(
        default="none",
        metadata={"help": "Advantage calculation mode: 'none' - only use ORA; 'only' - only use PRA; 'both' - weighted combination of ORA and PRA"}
    )
    
    process_reward_advantage_coef: float = field(
        default=1.0,
        metadata={"help": "Weight coefficient for PRA (Process Reward Advantage)"}
    )
    
    outcome_reward_advantage_coef: float = field(
        default=1.0,
        metadata={"help": "Weight coefficient for ORA (Outcome Reward Advantage)"}
    )
    
    # SPSA Reward Gradient Guidance configuration
    spsa_reward_enabled: bool = field(
        default=False,
        metadata={"help": "Whether to enable SPSA reward gradient guidance"}
    )
    
    spsa_reward_sigma: float = field(
        default=0.01,
        metadata={"help": "SPSA perturbation coefficient σ"}
    )
    
    spsa_reward_guidance_scale: float = field(
        default=0.1,
        metadata={"help": "Scaling coefficient for SPSA estimated gradient"}
    )
    
    spsa_reward_normalize_gradient: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the SPSA estimated gradient"}
    )
    
    spsa_reward_num_samples: int = field(
        default=1,
        metadata={"help": "How many sets of random samples SPSA uses to estimate the gradient"}
    )
    
    spsa_reward_start_step: int = field(
        default=0,
        metadata={"help": "Which denoising step to start using SPSA Reward from"}
    )
    
    spsa_reward_end_step: int = field(
        default=-1,
        metadata={"help": "Which denoising step to stop using SPSA Reward at"}
    )
    
    spsa_reward_interval: int = field(
        default=1,
        metadata={"help": "How many steps to skip between using SPSA Reward"}
    )
    
    spsa_reward_kl_beta: float = field(
        default=0.1,
        metadata={"help": "KL divergence coefficient beta"}
    )    
    # Log Prob computation control parameters
    use_reward_guided_mean_for_logprob: bool = field(
        default=True,
        metadata={"help": "[Sampling Phase] Whether to use the mean containing the reward gradient in log prob calculation. True: use mean with reward gradient; False: use mean without reward gradient."}
    )
    
    use_reward_guided_mean_for_logprob_training: bool = field(
        default=True,
        metadata={"help": "[Training Phase] Whether to use the mean containing the reward gradient in log prob calculation. True: use mean with reward gradient; False: use mean without reward gradient."}
    )