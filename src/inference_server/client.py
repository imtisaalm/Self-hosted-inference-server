from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class StreamResult:
    text: str
    time_to_first_chunk_ms: float | None
    total_latency_ms: float
    chunks: int


def _content_from_chunk(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return ""

    delta = first.get("delta")
    if not isinstance(delta, dict):
        return ""

    content = delta.get("content")
    return content if isinstance(content, str) else ""


def stream_chat(
    base_url: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int = 64,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> StreamResult:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    start = time.perf_counter()
    first: float | None = None
    chunks = 0
    output: list[str] = []

    with httpx.stream(
        "POST",
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue

            data = line[6:]
            if data == "[DONE]":
                break

            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError("server emitted malformed SSE JSON") from exc

            if not isinstance(payload, dict):
                raise ValueError("server emitted a non-object SSE payload")

            content = _content_from_chunk(payload)
            if not content:
                continue

            now = time.perf_counter()
            if first is None:
                first = now
            chunks += 1
            output.append(content)

    end = time.perf_counter()
    return StreamResult(
        text="".join(output),
        time_to_first_chunk_ms=None if first is None else (first - start) * 1000,
        total_latency_ms=(end - start) * 1000,
        chunks=chunks,
    )


def list_models(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> list[str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = httpx.get(
        f"{base_url.rstrip('/')}/v1/models",
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("model-list response must be a JSON object")

    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("model-list response is missing the data array")

    models: list[str] = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("model-list response contains an invalid model entry")
        models.append(item["id"])
    return models
