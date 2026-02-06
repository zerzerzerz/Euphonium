#!/bin/bash
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

# Model and data paths (please modify as needed)
HUNYUAN_VIDEO_PATH="/path/to/HunyuanVideo"
MODEL_PATH="$HUNYUAN_VIDEO_PATH"
OUTPUT_DIR="${WORKDIR}/assets/embeddings/vbench2"
PROMPT_PATH="${WORKDIR}/assets/VBench2_full_text.txt"

export PYTHONPATH=$PYTHONPATH:$WORKDIR

# Torchrun launch configuration
USE_RDZV=false  # Whether to use rdzv mode for torchrun, default: false
export nproc_per_node=$(nvidia-smi --list-gpus | wc -l)
echo "Detected number of GPUs: $nproc_per_node"
export MASTER_PORT=19006  # Use different ports to avoid conflicts

# hostfile path (used for multi-node training)
hostfile="${hostfile:-/root/cur.hosts}"

####################################################################################
# Pre-training Operations and Environment Preparation
####################################################################################
unset http_proxy
unset https_proxy

cd "$WORKDIR"
echo "work dir: $WORKDIR"

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

echo "master_port: $MASTER_PORT"

# Multi-node training configuration
if [ -f "$hostfile" ]; then
    # Read hostfile, filter valid lines (non-empty, non-comment), and sort
    readarray -t ips < <(grep -v '^#' "$hostfile" | grep -v '^$' | sort)
    
    # Use the first one as master
    MASTER_ADDR=${ips[0]}
    echo "master_addr: $MASTER_ADDR"
    
    # Set nnodes to the number of IPs
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
    # Single-node training
    echo "No hostfile found, using single node configuration"
    MASTER_ADDR="localhost"
    nnodes=1
    node_rank=0
fi

export nnodes=$nnodes
export MASTER_ADDR=$MASTER_ADDR

# Launch arguments for torchrun
if [ "$USE_RDZV" = "true" ]; then
    launch_args="
        --nproc_per_node $nproc_per_node \
        --nnodes $nnodes \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv_id=euphonium_embedding \
        --rdzv_backend=c10d"
else
    launch_args="
        --nproc_per_node $nproc_per_node \
        --nnodes $nnodes \
        --node_rank $node_rank \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT"
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"

# Log path
logs_path="${OUTPUT_DIR}/logs/preprocess_${current_ip}.txt"

####################################################################################
# Environment Variable Configuration - Adjust communication and debugging settings before training as needed
####################################################################################
# Basic NCCL settings
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

####################################################################################
# Run Preprocessing Script
####################################################################################

echo ""
echo "=========================================="
echo "Starting Prompt Embedding Preprocessing"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Prompt file: $PROMPT_PATH"
echo ""
echo "After preprocessing, set in the training script:"
echo "  data_json_path=${OUTPUT_DIR}/videos2caption.json"
echo "=========================================="
echo ""

nohup torchrun $launch_args \
    "${WORKDIR}/scripts/preprocess_hunyuan_text_embeddings.py" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --prompt_dir "$PROMPT_PATH" \
    --model_type hunyuan_hf \
    > "$logs_path" 2>&1 &

echo "Preprocessing started in background."
echo "Logs are being saved to: $logs_path"
echo "You can monitor the progress with: tail -f $logs_path"

echo ""
# Preprocessing complete message removed as process runs in background.
