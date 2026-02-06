#!/bin/bash
conda create -n euphonium python=3.10 -y
conda activate euphonium

# install torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# transformers
pip install transformers==4.56.2 trl==0.22.2 datasets==4.1.0 timm==1.0.19 qwen-vl-utils==0.0.11 math_verify==0.5.2 decord==0.6.0 protobuf==5.29.5
pip install deepspeed==0.16.9

# diffusers and others
pip install diffusers==0.34.0 peft==0.15.0 liger-kernel==0.6.4
pip install easydict tensorboard ml-collections inflect wandb einops matplotlib seaborn opencv-python-headless imageio pandas scipy scikit-learn
