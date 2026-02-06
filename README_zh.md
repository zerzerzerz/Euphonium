
# Euphonium: Steering Video Flow Matching via Process Reward Gradient Guided Stochastic Dynamics

Euphonium 是一个基于 GRPO (Group Relative Policy Optimization) 算法和奖励梯度引导 (RGG) 对 HunyuanVideo 视频生成模型进行强化学习微调的框架。

[English Version](README.md)

## 🚀 快速开始

### 克隆仓库

```bash
git clone --recursive https://github.com/your-org/Euphonium.git
cd Euphonium
```

> **注意**: 必须使用 `--recursive` 参数来正确初始化子模块 (Latent_PRM 和 SoliReward)。

## 🔧 环境配置

### 基础要求

- 推荐使用 **CUDA 12.4** 以获得最佳性能和兼容性。

### 安装步骤

```bash
# 运行环境配置脚本，自动安装所有依赖
bash scripts/env_setup.sh
```

### 下载预训练模型

需要下载以下模型：

- **HunyuanVideo**: 主生成模型
  - 可从 [Hugging Face](https://huggingface.co/tencent/HunyuanVideo) 下载

- **奖励模型** (必需):
  - **SoliReward** (像素空间 ORM, 如 InternVL3-1B) - **必需**
  - **Latent PRM** (潜空间过程奖励模型) - **必需**
  - VideoAlign 奖励模型 - **可选**

---

## 📊 数据预处理

数据预处理的目标是将原始的 text prompts 转换为预计算的 text embeddings，以加速训练。输出索引文件将使用**绝对路径**，方便在不同工作目录下直接使用。

### 输入输出格式

**输入示例** (`prompts.txt`):
```text
A dog running in the park.
A cat sleeping on the couch.
A person playing basketball.
```

**输出结构**:
```
output_dir/
├── prompt_embed/           # 文本嵌入向量 (.pt 文件)
├── prompt_attention_mask/  # 注意力掩码 (.pt 文件)
└── videos2caption.json     # 数据索引文件 (包含绝对路径)
```

### 使用预处理脚本

1. **准备 prompt 文件**: 创建一个文本文件，每行一个 prompt。
2. **配置并运行**:
   编辑 `scripts/preprocess_hunyuan_text_embeddings.sh`，设置 `WORKDIR`、`MODEL_PATH` 和 `PROMPT_PATH`。

   ```bash
   bash scripts/preprocess_hunyuan_text_embeddings.sh
   ```

---

## 🚀 GRPO 训练

### 训练配置

编辑 `scripts/train_grpo_hunyuan.sh` 配置相关变量：
- `HUNYUAN_VIDEO_PATH`: 基础模型路径。
- `TRL_MODEL_PATH`: 奖励模型路径。
- `data_json_path`: 指向预处理生成的 `videos2caption.json`。

### 多机训练

**推荐**: 使用 `pssh` (parallel-ssh) 在多个节点上启动训练。

1. **创建 hostfile**: 每行一个 IP。
   ```text
   192.168.1.1
   192.168.1.2
   ```

2. **在每台机器上运行**:
   ```bash
   export hostfile=/path/to/hostfile
   bash scripts/train_grpo_hunyuan.sh
   ```

   或使用 `pssh` 并行启动:
   ```bash
   pssh -h hostfile -i "cd /path/to/Euphonium && bash scripts/train_grpo_hunyuan.sh"
   ```

### 训练日志与监控

- **TensorBoard**: 日志保存在 `${output_base_dir}/tensorboard/`。
- **日志**: 详细运行日志位于 `${output_base_dir}/logs/`。

---

## ⚙️ 重要训练参数

以下环境变量控制奖励模型和 RGG (奖励梯度引导) 的行为:

### 奖励模型配置

- **`TRL_CORE_ENABLED`**: 启用像素空间 ORM (SoliReward)
- **`LATENT_REWARD_IN_TRL_REWARD_ENABLED`**: 启用潜空间 PRM
- **`LATENT_REWARD_IN_TRL_REWARD_COEF`**: 计算总奖励时潜空间奖励的系数 (默认: 0,在 PRM advantage 中单独计算)

### 过程奖励模型 (PRM) 设置

- **`PROCESS_LATENT_REWARD_ENABLED`**: 启用过程潜空间奖励模型计算
- **`PROCESS_LATENT_REWARD_SAMPLING_ENABLED`**: 在采样时启用 RGG (奖励梯度引导)
- **`PROCESS_LATENT_REWARD_TRAINING_ENABLED`**: ⚠️ **已弃用** - 未使用

### RGG 配置

- **`USE_REWARD_GUIDED_MEAN_FOR_LOGPROB`**: 采样时计算对数概率的均值时使用 RGG
- **`USE_REWARD_GUIDED_MEAN_FOR_LOGPROB_TRAINING`**: 训练时计算对数概率的均值时使用 RGG
- **`PROCESS_LATENT_REWARD_GUIDANCE_SCALE`**: RGG 强度的引导尺度系数
- **`USE_DELTA_T_FOR_GRADIENT_SCALING`**: RGG 系数是否乘以 delta_t (步长)

### Advantage 计算

- **`PROCESS_REWARD_ADVANTAGE_MODE`**: 计算双奖励 advantage 的模式
  - `both`: 同时使用 PRM 和 ORM
  - `none`: 仅使用 ORM (结果奖励模型)
  - `only`: 仅使用 PRM (过程奖励模型)

### 梯度估计

- **`SPSA_REWARD_ENABLED`**: 使用 SPSA 估计的 ORM 对潜变量的梯度

---

## 🎬 推理/可视化

### 运行推理

编辑 `scripts/vis_hunyuanvideo.sh` 设置模型和数据路径，然后运行：

```bash
bash scripts/vis_hunyuanvideo.sh
```

---

## 📚 引用

如果你使用了 Euphonium,请引用我们的论文:

```bibtex
@article{zhong2026euphonium,
  title={Euphonium: Steering Video Flow Matching via Process Reward Gradient Guided Stochastic Dynamics},
  author={Zhong, Ruizhe and Lian, Jiesong and Mi, Xiaoyue and Zhou, Zixiang and Zhou, Yuan and Lu, Qinglin and Yan, Junchi},
  journal={arXiv preprint arXiv:2602.04928},
  year={2026}
}
```

**论文链接**: [arXiv:2602.04928](https://arxiv.org/abs/2602.04928)

## 📄 License

Apache License 2.0

## 致谢

我们向以下项目表示感谢：
- https://github.com/XueZeyue/DanceGRPO
- https://github.com/huggingface/diffusers
- https://github.com/hao-ai-lab/FastVideo
- https://github.com/Tencent-Hunyuan/HunyuanVideo