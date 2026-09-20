#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import statistics
import threading
import time

from inference_server.client import stream_chat
from inference_server.metrics import fetch_scheduler_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue concurrent streamed requests and sample vLLM scheduler state."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt",
        default="Explain why continuous batching improves GPU utilization in two paragraphs.",
    )
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")

    stop = threading.Event()
    observed: list[tuple[float, float, float]] = []

    def sample() -> None:
        while not stop.is_set():
            try:
                m = fetch_scheduler_metrics(args.url, timeout=1.0)
                observed.append((m.running or 0.0, m.waiting or 0.0, m.kv_cache_usage or 0.0))
            except Exception:
                pass
            stop.wait(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    api_key = os.getenv("VLLM_API_KEY")
    results = []
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    stream_chat,
                    args.url,
                    model=args.model,
                    prompt=f"request={i}\n{args.prompt}",
                    max_tokens=args.max_tokens,
                    api_key=api_key,
                )
                for i in range(args.concurrency)
            ]
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        stop.set()
        sampler.join(timeout=1.0)

    elapsed = time.perf_counter() - started
    latencies = [r.total_latency_ms for r in results]
    first_chunks = [
        r.time_to_first_chunk_ms
        for r in results
        if r.time_to_first_chunk_ms is not None
    ]
    peak_running = max((x[0] for x in observed), default=0.0)
    peak_waiting = max((x[1] for x in observed), default=0.0)
    peak_kv = max((x[2] for x in observed), default=0.0)

    print(f"requests: {len(results)}/{args.concurrency}")
    print(f"wall time: {elapsed:.3f} s")
    print(f"mean request latency: {statistics.fmean(latencies):.1f} ms")
    if first_chunks:
        print(f"mean first-chunk latency: {statistics.fmean(first_chunks):.1f} ms")
    print(f"peak scheduler running: {peak_running:.0f}")
    print(f"peak scheduler waiting: {peak_waiting:.0f}")
    print(f"peak KV utilization: {peak_kv * 100:.1f}%")


if __name__ == "__main__":
    main()
