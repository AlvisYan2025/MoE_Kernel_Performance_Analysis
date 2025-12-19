#!/bin/bash
# Script to launch a model with BASELINE/DEFAULT configuration (no FlashInfer MoE)
# Example:
# ./launch_baseline.sh mistralai/Mixtral-8x7B-Instruct-v0.1 8001 4096 16 4096 4 1

MODEL=${1:-"mistralai/Mixtral-8x7B-v0.1"}
PORT=${2:-8000}  
BATCH_TOKENS=${3:-8192}     
MAX_SEQS=${4:-32}          
MAX_MODEL_LEN=${5:-4096}
TP_SIZE=${6:-4}
PROFILE=${7:-1}  

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/vllm_baseline_${TIMESTAMP}_tp${TP_SIZE}_batch${BATCH_TOKENS}_seq${MAX_SEQS}_modellen${MAX_MODEL_LEN}.log"

# Use cache -- comment off if not using cache
HF_CACHE_DIR="${SCRIPT_DIR}/../hf_cache"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
echo "Using HuggingFace cache at: $HF_CACHE_DIR"

echo "========================================="
echo "Launching vLLM with BASELINE Configuration"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel: $TP_SIZE"
echo "Batch Tokens:    $BATCH_TOKENS"
echo "Max Seqs:        $MAX_SEQS"
echo "Max Model Len:   $MAX_MODEL_LEN"
echo "Profiling:       $PROFILE"
echo "Log File:        $LOG_FILE"
echo "========================================="

# Standard logging 
export VLLM_LOGGING_LEVEL=INFO

# Profiling setup
PROFILER=""
if [ "$PROFILE" -eq 1 ]; then
    PROFILE_DIR="${SCRIPT_DIR}/profiles"
    mkdir -p "$PROFILE_DIR"
    PROFILE_FILE="${PROFILE_DIR}/nsys_baseline_${TIMESTAMP}_tp${TP_SIZE}.qdrep"
    
    echo "Enabling Nsight Systems profiling..."
    echo "Profile output: $PROFILE_FILE"
    
    PROFILER="nsys profile \
        --output=$PROFILE_FILE \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage=true \
        --gpu-metrics-device=all \
        --delay=30 \
        --duration=60 \
        --sample=cpu \
        --backtrace=dwarf"
fi

# Standard configuration
# Uses default Triton MoE kernels, no expert parallelism
$PROFILER vllm serve $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size $TP_SIZE \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-batched-tokens $BATCH_TOKENS \
  --max-num-seqs $MAX_SEQS \
  --enforce-eager \
  --disable-log-requests \
  2>&1 | tee "$LOG_FILE"