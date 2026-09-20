#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:=Qwen/Qwen3-0.6B}"
: "${SERVED_MODEL_NAME:=local-model}"
: "${PORT:=8000}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_SEQS:=64}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${KV_CACHE_DTYPE:=auto}"

args=(
  serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port 8000
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
)

if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
  args+=(--enable-prefix-caching)
fi

if [[ -n "${VLLM_API_KEY:-}" ]]; then
  args+=(--api-key "$VLLM_API_KEY")
fi

docker_args=(
  run --rm --gpus all --ipc=host
  -p "$PORT:8000"
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface"
  --entrypoint vllm
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  docker_args+=(--env "HF_TOKEN=$HF_TOKEN")
fi

exec docker "${docker_args[@]}" vllm/vllm-openai:latest "${args[@]}"
