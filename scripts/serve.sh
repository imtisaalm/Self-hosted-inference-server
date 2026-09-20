#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${SERVED_MODEL_NAME:=local-model}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_SEQS:=64}"
: "${KV_CACHE_DTYPE:=auto}"
: "${ENABLE_PREFIX_CACHING:=1}"
: "${TENSOR_PARALLEL_SIZE:=1}"

args=(
  serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
)

if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  args+=(--enable-prefix-caching)
fi

if [[ -n "${VLLM_API_KEY:-}" ]]; then
  args+=(--api-key "$VLLM_API_KEY")
fi

exec vllm "${args[@]}"
