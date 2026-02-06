"""Constants for GRPO configuration."""

# TRL prompt variant configurations
TRL_PROMPT_VARIANTS = {
    "deformity_physics": {
        "system_prompt": "你是一个专业的视频质量评估专家。请根据提供的视频内容，判断是否同时满足以下所有问题的合格标准：\n\n1. 物理规律是否合格？\n2. 是否存在人物或动物畸形？\n\n回答要求：\n- 只有当所有问题的答案都是\"合格\"时，才输出：good\n- 如果任何一个问题的答案是\"部分合格\"或\"不合格\"，则输出：bad\n- 不要输出任何其他内容\n- 答案要准确、客观\n",
        "user_prompt": "请评估以下视频：",
        "include_generation_prompt": False,
    },
    "ta": {
        "system_prompt": "你是一个专业的视频质量评估专家。请根据提供的视频内容，判断是否同时满足以下所有问题的合格标准：\n\n1. 视频是否符合prompt的语义？\n\n回答要求：\n- 只有当所有问题的答案都是\"合格\"时，才输出：good\n- 如果任何一个问题的答案是\"部分合格\"或\"不合格\"，则输出：bad\n- 不要输出任何其他内容\n- 答案要准确、客观\n",
        "user_prompt": "。根据该提示词生成的视频为：",
        "include_generation_prompt": True,
    },
}

# Default prompt variant
DEFAULT_TRL_PROMPT_VARIANT = "deformity_physics"

# Generation prompt prefix template
DEFAULT_TRL_GENERATION_PROMPT_PREFIX = "生成视频的文本提示词是: "
