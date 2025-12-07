#!/bin/bash

# ===========================================================
# SUPER-NAIVE MoE BASELINE (EXTREMELY SLOW ON PURPOSE)
# ===========================================================

MODEL="mistralai/Mixtral-8x7B-v0.1"
PORT=${1:-8000}

# Very small batching → even slower
MAX_MODEL_LEN=4096
MAX_SEQS=1
BATCH_TOKENS=128
EP_SIZE=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGS_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/super_naive_${TS}.log"

HF_CACHE_DIR="${SCRIPT_DIR}/hf_cache"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"

echo "=============================================="
echo "Launching SUPER-NAIVE MoE baseline"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "=============================================="

# ------------------------------------------------------
# 🔥 Make everything extremely slow
# ------------------------------------------------------

# Disable ANY fused MoE kernels
export VLLM_USE_FUSED_MOE=0

# Disable FlashInfer backend
export VLLM_FLASHINFER_MOE_BACKEND="none"

# Disable Triton MoE kernels (force PyTorch fallback)
export VLLM_USE_TRITON_MOE=0

# Force naive all-to-all (worst communication backend)
export VLLM_ALLTOALL_BACKEND="naive"

# Force blocking GPU sync at every step (super slow)
export CUDA_LAUNCH_BLOCKING=1

# Disable multi-stream scheduling → all compute serialized
export VLLM_SCHEDULER_MAX_PARALLELISM=1

# Disable parallel expert execution → run each expert sequentially
export VLLM_MOE_GROUPED_FC_PARALLELISM=1

# Logging
export VLLM_LOGGING_LEVEL=INFO

# ===========================================================
# Launch vLLM engine
# ===========================================================
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