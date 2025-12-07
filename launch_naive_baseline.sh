#!/bin/bash

MODEL="mistralai/Mixtral-8x7B-v0.1"

PORT=${1:-8001}
MAX_MODEL_LEN=4096
MAX_SEQS=32
BATCH_TOKENS=8192

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGS_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/baseline_${TIMESTAMP}.log"

HF_CACHE_DIR="${SCRIPT_DIR}/hf_cache"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"

echo "=============================="
echo "Launching BASELINE vLLM"
echo "Model: $MODEL"
echo "Port:  $PORT"
echo "=============================="

# ---- Remove any leftover acceleration flags ----
unset VLLM_FLASHINFER_MOE_BACKEND
unset VLLM_USE_FUSED_MOE
unset VLLM_ALLTOALL_BACKEND
unset VLLM_USE_TRITON_MOE

# ---- Enable naive Triton MoE ----
export VLLM_USE_TRITON_MOE=1
export VLLM_LOGGING_LEVEL=INFO

# ---- Launch model (single-GPU baseline) ----
vllm serve $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 1 \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-batched-tokens $BATCH_TOKENS \
  --max-num-seqs $MAX_SEQS \
  --disable-log-requests \
  2>&1 | tee "$LOG_FILE"