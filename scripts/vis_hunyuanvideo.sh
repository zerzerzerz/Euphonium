#!/bin/bash
####################################################################################
# HunyuanVideo Inference/Visualization Script
# 
# Generate videos using pretrained or fine-tuned models
# Supports FSDP distributed inference
#
# Usage:
#   1. Modify the configuration parameters below
#   2. Run: bash scripts/vis_hunyuanvideo.sh
####################################################################################

####################################################################################
# Key Parameter Definitions - Please confirm before running
####################################################################################

# ==================== Root Directory Configuration (Must Modify) ====================
# Project root directory (the root of the Euphonium project)
WORKDIR="/path/to/Euphonium"
# ================================================================

# Conda environment configuration (please modify as needed)
conda_activate="/path/to/miniconda3/bin/activate"
conda_env_name="euphonium"

# Model path (please modify as needed)
HUNYUAN_VIDEO_PATH="/path/to/HunyuanVideo"
export PYTHONPATH=$PYTHONPATH:$WORKDIR

# Dataset configuration (videos2caption.json in the preprocessed embedding directory)
data_json_path="${WORKDIR}/assets/wan_short_captions_merged_val.json"

# Model weight configuration
pretrained_model_name_or_path="$HUNYUAN_VIDEO_PATH"
vae_model_path="$HUNYUAN_VIDEO_PATH"

# Checkpoint configuration (leave empty to use base model, or specify fine-tuned weights)
# Example: dit_model_name_or_path=/path/to/checkpoint-250-0/diffusion_pytorch_model.safetensors
dit_model_name_or_path="/path/to/checkpoint"
exp_name="euphonium"

# Output directory
output_base_dir="${WORKDIR}/result-inference/${exp_name}"
video_dir="${output_base_dir}/videos"

####################################################################################
# Video Generation Parameters
####################################################################################

VIDEO_HEIGHT=640      # Video height
VIDEO_WIDTH=640       # Video width
VIDEO_FRAMES=91       # Video frames (91 frames @ 18fps ≈ 5 seconds)
VIDEO_FPS=18          # Output video frame rate
SAMPLING_STEPS=30     # Sampling steps (more = better quality but slower)
ETA=0.0               # SDE solver eta (0 for deterministic DDIM)
SHIFT=7               # Noise schedule time shift
SEED=42               # Random seed

# Number of videos generated per prompt
NUM_VIDEOS_PER_PROMPT=1

####################################################################################
# Distributed Configuration
####################################################################################

# Automatically detect number of GPUs per node
export nproc_per_node=$(nvidia-smi --list-gpus | wc -l)
echo "Detected number of GPUs: $nproc_per_node"
export MASTER_PORT=19006
hostfile="${hostfile:-/root/cur.hosts}"

####################################################################################
# Environment Preparation
####################################################################################

unset http_proxy
unset https_proxy

cd "$WORKDIR"
echo "Work dir: $WORKDIR"

# Activate conda environment
if [ -f "$conda_activate" ]; then
    echo "Using conda activate: $conda_activate"
    echo "Using conda env name: $conda_env_name"
    source "$conda_activate" "$conda_env_name"
else
    echo "Warning: conda activate script not found at $conda_activate"
    echo "Please ensure the correct Python environment is activated"
fi

# Network adapter configuration (to get the local IP address)
network_adapter="bond1"  # Specify network adapter name, modify according to actual environment

echo "Master port: $MASTER_PORT"

# Multi-node inference configuration
if [ -f "$hostfile" ]; then
    # Read hostfile, filter valid lines (non-empty, non-comment), and sort
    readarray -t ips < <(grep -v '^#' "$hostfile" | grep -v '^$' | sort)
    MASTER_ADDR=${ips[0]}
    echo "master_addr: $MASTER_ADDR"
    
    nnodes=${#ips[@]}
    echo "nnodes: $nnodes"
    
    # Generate ip_to_rank mapping
    declare -A ip_to_rank
    for i in "${!ips[@]}"; do
        ip_to_rank[${ips[$i]}]=$i
    done
    # Get the local IP address (via specified adapter)
    current_ip=$(ip addr show $network_adapter | grep "inet " | awk '{print $2}' | cut -d/ -f1)
    echo "current ip: $current_ip"
    
    if [ -z "${ip_to_rank[$current_ip]}" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "WARNING: Current IP ($current_ip) not found in hostfile!"
        echo "This node may not be correctly identified in the distributed setup."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    fi
    
    node_rank=${ip_to_rank[$current_ip]:-0}
    echo "node_rank = ${node_rank}"
else
    # Single-node inference
    echo "No hostfile found, using single node configuration"
    MASTER_ADDR="localhost"
    nnodes=1
    node_rank=0
fi

export nnodes=$nnodes
export MASTER_ADDR=$MASTER_ADDR

launch_args="
    --nproc_per_node $nproc_per_node \
    --nnodes $nnodes \
    --node_rank $node_rank \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

# Create output directories
mkdir -p "${output_base_dir}"
mkdir -p "${output_base_dir}/logs"
mkdir -p "${video_dir}"

# Log path
logs_path="${output_base_dir}/logs/inference_${current_ip}.txt"

####################################################################################
# NCCL Environment Variable Configuration
####################################################################################
# Basic NCCL settings
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

####################################################################################
# Run Inference
####################################################################################

echo ""
echo "=========================================="
echo "Starting Video Generation Inference"
echo "=========================================="
echo "Model path: $pretrained_model_name_or_path"
echo "Checkpoint: ${dit_model_name_or_path:-'(using base model)'}"
echo "Data path: $data_json_path"
echo "Output directory: $video_dir"
echo "Video size: ${VIDEO_WIDTH}x${VIDEO_HEIGHT}, ${VIDEO_FRAMES} frames @ ${VIDEO_FPS}fps"
echo "=========================================="
echo ""

nohup torchrun $launch_args \
    "${WORKDIR}/scripts/vis_hunyuanvideo.py" \
    $(if [ -n "$dit_model_name_or_path" ] && [ "$dit_model_name_or_path" != "" ]; then echo "--dit_model_name_or_path $dit_model_name_or_path"; fi) \
    --pretrained_model_name_or_path "$pretrained_model_name_or_path" \
    --data_json_path "$data_json_path" \
    --vae_model_path "$vae_model_path" \
    --output_dir "${output_base_dir}" \
    --video_dir "${video_dir}" \
    --h $VIDEO_HEIGHT \
    --w $VIDEO_WIDTH \
    --t $VIDEO_FRAMES \
    --fps $VIDEO_FPS \
    --sampling_steps $SAMPLING_STEPS \
    --eta $ETA \
    --shift $SHIFT \
    --seed $SEED \
    --num_videos_per_prompt $NUM_VIDEOS_PER_PROMPT \
    > "$logs_path" 2>&1 &

echo "Inference started in background."
echo "Logs are being saved to: $logs_path"
echo "You can monitor the progress with: tail -f $logs_path"

echo ""
# Inference complete message removed as process runs in background.
