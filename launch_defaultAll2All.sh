#!/bin/bash
# Modified script - remove FlashInfer all2all, keep FlashInfer MoE backend

MODEL=${1:-"mistralai/Mixtral-8x7B-v0.1"}
PORT=${2:-8000}
BATCH_TOKENS=${3:-8192}     
MAX_SEQS=${4:-32}          
MAX_MODEL_LEN=${5:-4096}
EP_SIZE=${6:-4}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/vllm_defualt_${TIMESTAMP}_batch${BATCH_TOKENS}_seq${MAX_SEQS}_modellen${MAX_MODEL_LEN}.log"

HF_CACHE_DIR="${SCRIPT_DIR}/hf_cache"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"

echo "========================================="
echo "Launching vLLM with FlashInfer MoE (default all2all)"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Expert Parallel: $EP_SIZE"
echo "========================================="

export VLLM_FLASHINFER_MOE_BACKEND="throughput"
export VLLM_LOGGING_LEVEL=INFO


####--------------------
# export VLLM_USE_V1=0 
# export VLLM_TORCH_COMPILE_LEVEL=0
# export TORCH_COMPILE_DISABLE=1
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export TMPDIR=/tmp
# export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_$$"
# mkdir -p "$TORCH_EXTENSIONS_DIR"
# export NCCL_DEBUG=WARN
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=5
#does not fix the race condition 


#removed --all2all-backend flashinfer_all2allv
vllm serve $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 1 \
  --data-parallel-size $EP_SIZE \
  --enable-expert-parallel \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-batched-tokens $BATCH_TOKENS \
  --max-num-seqs $MAX_SEQS \
  --trust-remote-code \
  --enforce-eager
  --disable-log-requests \
  2>&1 | tee "$LOG_FILE"



rm -rf "$TORCH_EXTENSIONS_DIR" #clean up 