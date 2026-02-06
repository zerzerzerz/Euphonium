"""
TRL Core Reward Model

Core reward model trained based on the TRL framework, supporting various model types such as InternVL and QwenVL.
"""

import os
import sys
import torch
from typing import Dict, List, Any, Optional

from .base_sub_reward import BaseSubReward


class TRLCoreReward(BaseSubReward):
    """
    TRL Core Reward Model.
    
    Video quality assessment model trained based on the TRL framework.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        super().__init__(config, device)
        
        # Parse configuration
        self._parse_config()
        
        # Model-related (deferred initialization)
        self.model = None
        self.data_collator = None
        self.model_type = None
        self.model_path = None
        
        # Message printing counter (for debugging)
        self._message_print_count = 0
        self._max_message_prints = 3
    
    def _parse_config(self) -> None:
        """Parse configuration related to the TRL core model."""
        # Control whether TRL Core is enabled via an explicit parameter
        self._enabled = self._get_config_value(
            "trl_core_enabled", True, bool  # Enabled by default for backward compatibility
        )
        
        self._coef = self._get_config_value(
            "score_coef", 1.0, float
        )
        
        self.append_generation_prompt = self.config.get("append_generation_prompt", False)
        self.generation_prompt_prefix = self.config.get(
            "generation_prompt_prefix", "生成视频的文本提示词是: "
        )
        self.prompt_variant = self.config.get("prompt_variant", "deformity_physics")
        
        self.trl_path = self.config.get('trl_path', 'SoliReward')
        
        self.system_prompt = self.config.get('system_prompt', 
            "你是一个专业的视频质量评估专家。请根据提供的视频内容，判断是否同时满足以下所有问题的合格标准：\n\n"
            "1. 物理规律是否合格？\n"
            "2. 是否存在人物或动物畸形？\n\n"
            "回答要求：\n"
            "- 只有当所有问题的答案都是\"合格\"时，才输出：good\n"
            "- 如果任何一个问题的答案是\"部分合格\"或\"不合格\"，则输出：bad\n"
            "- 不要输出任何其他内容\n"
            "- 答案要准确、客观\n"
        )
        self.user_prompt = self.config.get('user_prompt', "请评估以下视频：")
    
    def _import_modules(self) -> None:
        """Add SoliReward package path to sys.path and import modules."""
        # Use the specified path if provided
        if self.trl_path:
            solireward_path = os.path.abspath(self.trl_path)
        else:
            # Default to relative path: from euphonium/reward_models/trl_reward/ back to third_party/SoliReward/ in project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            solireward_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "third_party", "SoliReward"))
        
        if solireward_path not in sys.path:
            sys.path.insert(0, solireward_path)
            print(f"[TRLCoreReward] Adding SoliReward path to sys.path: {solireward_path}")
        
        from solireward.models import load_reward_model_and_collator
        self.load_reward_model_and_collator = load_reward_model_and_collator
    
    def initialize(self) -> None:
        """Initialize TRL reward model."""
        if not self._enabled:
            print(f"[TRLCoreReward] Not enabled, skipping initialization")
            return
        
        # Import modules
        self._import_modules()
        
        # Load model
        model_path = self.config['model_name_or_path']
        print(f"[TRLCoreReward] Loading model: {model_path}")
        model, data_collator, model_type = self.load_reward_model_and_collator(model_path)
        print(f"[TRLCoreReward] Model loading complete")
        
        self.model = model.bfloat16().cuda().eval()
        self.model_path = model_path
        self.data_collator = data_collator
        self.model_type = model_type
        
        print(f"[TRLCoreReward] Initialization successful")
        print(f"  - model_path: {model_path}")
        print(f"  - model_type: {model_type}")
        print(f"  - coef: {self._coef}")
    
    def _build_messages(self, 
                        video_paths: List[str], 
                        captions: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Build message format.
        
        Args:
            video_paths: A list of paths to video files.
            captions: A list of corresponding text descriptions.
            
        Returns:
            A list of messages.
        """
        batch_messages = []
        for idx, video_path in enumerate(video_paths):
            # Construct user prompt
            user_text = self.user_prompt
            
            # If generation prompt feature is enabled, prepend generation prompt to user prompt
            if self.append_generation_prompt and idx < len(captions):
                generation_prompt = f"{self.generation_prompt_prefix}{captions[idx]}"
                user_text = generation_prompt + user_text
            
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "video", "video": video_path}
                    ]
                }
            ]
            
            # Only print messages for initial calls to aid debugging
            if self._message_print_count < self._max_message_prints:
                print(f"[TRLCoreReward] Message {self._message_print_count + 1}/{self._max_message_prints}:")
                print(messages)
                self._message_print_count += 1
            
            batch_messages.append(messages)
        
        return batch_messages
    
    def _get_target_dtype(self) -> Optional[torch.dtype]:
        """Get the target data type."""
        dtype_str = self.config.get('dtype', 'bf16')
        if dtype_str == 'bf16':
            return torch.bfloat16
        elif dtype_str == 'fp16':
            return torch.float16
        elif dtype_str == 'fp32':
            return torch.float32
        return None
    
    @torch.no_grad()
    def _predict_batch(self, batch_messages: List[List[Dict[str, Any]]]) -> List[float]:
        """Predict reward scores in batches."""
        # Process data
        if self.model_type in ['InternVL3', 'InternVL3-5']:
            # InternVL processing
            fake_pair_batch = []
            for messages in batch_messages:
                fake_pair_batch.append({
                    'win': messages,
                    'lose': messages,
                    'meta': {'win': {'quality': 1.0}, 'lose': {'quality': 0.0}}
                })
            batch_data = self.data_collator(fake_pair_batch)
            model_input = {k: v for k, v in batch_data['win'].items() 
                          if k not in ['quality', 'num_patches_list']}
        else:
            # QwenVL processing
            fake_pair_batch = []
            for messages in batch_messages:
                fake_pair_batch.append({
                    'win': messages,
                    'lose': messages,
                    'meta': {'win': {'quality': 1.0}, 'lose': {'quality': 0.0}}
                })
            batch_data = self.data_collator(fake_pair_batch)
            batch_size = batch_data['batch_size']
            inputs_concat = batch_data['inputs_concat']
            
            model_input = {}
            for key, value in inputs_concat.items():
                if isinstance(value, torch.Tensor):
                    # Check if the first dimension of the tensor is 2 * batch_size
                    if value.dim() > 0 and value.shape[0] == 2 * batch_size:
                        model_input[key] = value[:batch_size]
                    elif key == 'pixel_values_videos' and 'video_grid_thw' in inputs_concat:
                        # Special handling for pixel_values_videos: calculate split position based on video_grid_thw
                        video_grid_thw = inputs_concat['video_grid_thw']
                        if video_grid_thw.shape[0] == 2 * batch_size:
                            # Calculate total token count for the first batch_size videos
                            num_tokens_per_video = video_grid_thw[:, 0] * video_grid_thw[:, 1] * video_grid_thw[:, 2]
                            total_tokens_first_half = num_tokens_per_video[:batch_size].sum().item()
                            model_input[key] = value[:int(total_tokens_first_half)]
                        else:
                            model_input[key] = value
                    else:
                        model_input[key] = value
                elif isinstance(value, (list, tuple)):
                    # For lists/tuples, check if the length matches
                    if len(value) == 2 * batch_size:
                        model_input[key] = value[:batch_size]
                    else:
                        model_input[key] = value
                else:
                    model_input[key] = value
        
        # Move to device and set precision
        target_dtype = self._get_target_dtype()
        
        for key, value in model_input.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point and target_dtype is not None:
                    value = value.to(target_dtype)
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    model_input[key] = value.cuda()
                else:
                    model_input[key] = value.to(torch.device(self.device))
        
        # Model inference
        with torch.autocast(
            device_type='cuda' if self.device.startswith('cuda') else 'cpu',
            dtype=target_dtype if target_dtype else torch.float32
        ):
            outputs = self.model(**model_input)
        
        # Extract scores
        reward_scores = None
        if isinstance(outputs, dict):
            if 'reward_scores' in outputs:
                reward_scores = outputs['reward_scores'].squeeze().cpu().tolist()
            elif 'reward' in outputs:
                reward_scores = outputs['reward'].squeeze().cpu().tolist()
            elif 'logits' in outputs:
                reward_scores = outputs['logits'].squeeze().cpu().tolist()
        else:
            if hasattr(outputs, 'reward_scores'):
                reward_scores = outputs.reward_scores.squeeze().cpu().tolist()
            elif hasattr(outputs, 'reward'):
                reward_scores = outputs.reward.squeeze().cpu().tolist()
            elif hasattr(outputs, 'logits'):
                reward_scores = outputs.logits.squeeze().cpu().tolist()
        
        if reward_scores is None:
            raise ValueError("No valid reward score field found in model output")
        
        # Ensure a list is returned
        if isinstance(reward_scores, float):
            reward_scores = [reward_scores]
        
        return [float(score) for score in reward_scores]
    
    @torch.no_grad()
    def compute(self, 
                video_paths: List[str], 
                captions: List[str]) -> torch.Tensor:
        """
        Compute TRL reward scores.
        
        Args:
            video_paths: A list of paths to video files.
            captions: A list of corresponding text descriptions.
            
        Returns:
            torch.Tensor: (B,) Reward scores.
        """
        if not self._enabled:
            # Return zero scores if not enabled
            return torch.zeros(len(video_paths), device=self.device, dtype=torch.float32)
        
        if self.model is None:
            raise RuntimeError("TRL reward model has not been initialized")
        
        # Build messages
        batch_messages = self._build_messages(video_paths, captions)
        
        # Prediction
        scores = self._predict_batch(batch_messages)
        
        return torch.tensor(scores, device=self.device, dtype=torch.float32)
