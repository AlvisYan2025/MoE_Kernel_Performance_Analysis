#!/bin/bash

MODEL=${1:-"mistralai/Mixtral-8x7B-Instruct-v0.1"}
PORT=${2:-8000}

echo "========================================="
echo "Launching vLLM with FlashInfer MoE"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "========================================="

export VLLM_FLASHINFER_MOE_BACKEND="throughput"
export VLLM_LOGGING_LEVEL=INFO

vllm serve $MODEL \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend flashinfer_all2allv \
  --trust-remote-code \
  --max-model-len 4096 \
  2>&1 | tee logs/vllm_flashinfer.log
