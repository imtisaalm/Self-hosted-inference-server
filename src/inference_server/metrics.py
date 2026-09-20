from __future__ import annotations

from dataclasses import dataclass
import re

import httpx


_SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")


@dataclass(frozen=True)
class SchedulerMetrics:
    running: float | None
    waiting: float | None
    kv_cache_usage: float | None


def scheduler_metrics(text: str) -> SchedulerMetrics:
    values: dict[str, list[float]] = {}
    for raw in text.splitlines():
        match = _SAMPLE.match(raw.strip())
        if match:
            values.setdefault(match.group(1), []).append(float(match.group(2)))

    def total(name: str) -> float | None:
        samples = values.get(name)
        return None if samples is None else sum(samples)

    return SchedulerMetrics(
        running=total("vllm:num_requests_running"),
        waiting=total("vllm:num_requests_waiting"),
        kv_cache_usage=total("vllm:kv_cache_usage_perc"),
    )


def fetch_scheduler_metrics(base_url: str, timeout: float = 5.0) -> SchedulerMetrics:
    response = httpx.get(f"{base_url.rstrip('/')}/metrics", timeout=timeout)
    response.raise_for_status()
    return scheduler_metrics(response.text)
