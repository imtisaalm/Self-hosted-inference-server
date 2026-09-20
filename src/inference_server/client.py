from __future__ import annotations

from dataclasses import dataclass
import json
import time

import httpx


@dataclass(frozen=True)
class StreamResult:
    text: str
    time_to_first_chunk_ms: float | None
    total_latency_ms: float
    chunks: int


def _content_from_chunk(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


def stream_chat(
    base_url: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int = 64,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> StreamResult:
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
            payload = json.loads(data)
            content = _content_from_chunk(payload)
            if content:
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


def list_models(base_url: str, api_key: str | None = None, timeout: float = 10.0) -> list[str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = httpx.get(f"{base_url.rstrip('/')}/v1/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    return [item["id"] for item in response.json().get("data", [])]
