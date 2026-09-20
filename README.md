# Self-hosted inference server

A reproducible configuration for serving an open-weight language model with vLLM behind an OpenAI-compatible HTTP API.

The repository does not wrap vLLM in an additional application server. vLLM owns request scheduling, continuous batching, KV-cache management, and model execution. This repository contains launch configuration, a streaming verification client, scheduler-metric inspection, and a concurrent-request smoke test.

## Runtime boundary

The control plane is Python-facing because vLLM exposes its server, configuration, and scheduler through Python. The performance-critical execution path is lower level: CUDA/C++/Triton kernels execute attention, GEMM, cache movement, sampling, and other GPU operations. This repository intentionally configures and measures that runtime rather than reimplementing kernels in Python.

```text
Python configuration / API / scheduler
              |
              v
      vLLM execution engine
              |
              v
   C++ / CUDA / Triton kernels
              |
              v
             GPU
```

This separation is typical of modern inference systems: a high-level control layer drives a compiled numerical runtime.

## Architecture

```text
OpenAI-compatible client
        |
        v
/v1/chat/completions
        |
        v
+---------------------------+
| vLLM API server           |
| scheduler / request queue |
| continuous batching       |
| KV cache                  |
| model runner              |
+---------------------------+
        |
        +----> /metrics
                 |
                 v
          verification tools
```

## Requirements

A CUDA-capable NVIDIA GPU is required for the default configuration. The Docker launch path requires Docker with the NVIDIA container runtime. The native launch path requires a working vLLM installation.

The default model is `Qwen/Qwen3-0.6B` so the configuration can be exercised on comparatively modest hardware. Change `MODEL`, context length, cache dtype, and parallelism for the target system.

## Configuration

Copy the example environment file and adjust it for the available hardware:

```bash
cp config/server.env.example .env
set -a
source .env
set +a
```

Key scheduler parameters:

| Variable | Meaning |
| --- | --- |
| `MAX_MODEL_LEN` | Maximum sequence length exposed by the server |
| `MAX_NUM_SEQS` | Maximum number of sequences processed in an iteration |
| `MAX_NUM_BATCHED_TOKENS` | Optional token budget scheduled in one iteration |
| `GPU_MEMORY_UTILIZATION` | Fraction of device memory made available to the vLLM executor |
| `TENSOR_PARALLEL_SIZE` | Number of tensor-parallel ranks |
| `KV_CACHE_DTYPE` | KV-cache storage dtype |
| `ENABLE_PREFIX_CACHING` | Enable automatic prefix-cache reuse |

## Start the server

Native vLLM installation:

```bash
bash scripts/serve.sh
```

Official vLLM Docker image:

```bash
bash scripts/docker_serve.sh
```

Both launch paths expose the server on port 8000 by default.

## Verify the API

Install the repository utilities:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

List served models:

```bash
inference-server check
```

Issue one streamed request:

```bash
inference-server generate \
  --model local-model \
  "Give a concise definition of continuous batching."
```

Inspect scheduler state:

```bash
inference-server metrics
```

The metrics command reads:

- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:kv_cache_usage_perc`

## Concurrent-request smoke test

```bash
python scripts/concurrency_smoke.py \
  --model local-model \
  --concurrency 8 \
  --max-tokens 128
```

The script starts concurrent streamed requests while sampling vLLM scheduler metrics. It reports completed requests, wall time, mean request latency, mean first-chunk latency, peak running requests, peak waiting requests, and peak KV-cache utilization.

A `peak scheduler running` value above one is direct evidence that requests overlapped in the execution scheduler during the sample. This is a smoke test rather than a benchmark; controlled latency/throughput measurement belongs in the separate TTFT/ITL benchmark suite.

## Continuous batching

vLLM performs scheduling at the engine level. Requests can enter and leave the active batch as sequences progress rather than waiting for a fixed batch to finish. `MAX_NUM_SEQS` and `MAX_NUM_BATCHED_TOKENS` expose two useful capacity controls for experiments in this repository.

The server's own metrics are used to observe request admission and queueing instead of inferring batching from client-side wall-clock timing alone.

## Tests

```bash
pytest -q
```

Unit tests cover configuration validation, streamed OpenAI response parsing, and scheduler-metric parsing. GPU execution is intentionally excluded from CI.

## Security

vLLM's API-key option protects selected API paths, but it is not a complete network security boundary. Do not expose a development server directly to an untrusted network. Use an appropriate reverse proxy, firewall, or private network boundary for remote deployments.

## References

- OpenAI-compatible vLLM server: https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/
- vLLM scheduler configuration: https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- vLLM Docker deployment: https://docs.vllm.ai/en/latest/deployment/docker/
- vLLM production metrics: https://docs.vllm.ai/en/latest/usage/metrics/
