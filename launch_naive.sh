#!/bin/bash

MODEL="mistralai/Mixtral-8x7B-v0.1"

PORT=${1:-8000}
MAX_MODEL_LEN=4096
MAX_SEQS=32
BATCH_TOKENS=8192
EP_SIZE=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/naive_${TIMESTAMP}.log"

HF_CACHE_DIR="${SCRIPT_DIR}/hf_cache"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"

echo "==========================================="
echo "Launching TRUE NAIVE MoE Backend"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "==========================================="

# -----------------------------
# Disable all optimized MoE paths
# -----------------------------

# Disable FlashInfer completely
export VLLM_FLASHINFER_MOE_BACKEND="none"

# Disable Triton fused MoE kernels
export VLLM_USE_TRITON_MOE=0
export VLLM_USE_FUSED_MOE=0

# Force naive all-to-all (slow)
export VLLM_ALLTOALL_BACKEND="naive"

# Disable CUDA Graphs
export VLLM_USE_CUDA_GRAPH=0

# Disable continuous batching optimization
export VLLM_CONTINUOUS_BATCHING=0

# Disable chunked prefill
export VLLM_DISABLE_CHUNKED_PREFILL=1

# Disable speculative scheduling
export VLLM_SCHEDULER_POLICY="naive"

export VLLM_LOGGING_LEVEL=INFO

vllm serve $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 1 \
  --data-parallel-size $EP_SIZE \
  --enable-expert-parallel \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-batched-tokens $BATCH_TOKENS \
  --max-num-seqs $MAX_SEQS \
  --disable-log-requests \
  2>&1 | tee "$LOG_FILE"