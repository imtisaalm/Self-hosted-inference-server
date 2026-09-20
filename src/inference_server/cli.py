from __future__ import annotations

import os

import typer

from .client import list_models, stream_chat
from .metrics import fetch_scheduler_metrics

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def check(url: str = "http://127.0.0.1:8000") -> None:
    """Verify the OpenAI-compatible model-list endpoint."""
    models = list_models(url, os.getenv("VLLM_API_KEY"))
    if not models:
        raise typer.Exit(code=1)
    for model in models:
        typer.echo(model)


@app.command()
def generate(
    prompt: str,
    url: str = "http://127.0.0.1:8000",
    model: str = "local-model",
    max_tokens: int = 64,
) -> None:
    """Issue one streamed chat request and report basic timing."""
    result = stream_chat(
        url,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        api_key=os.getenv("VLLM_API_KEY"),
    )
    typer.echo(result.text)
    if result.time_to_first_chunk_ms is None:
        typer.echo("first chunk: n/a")
    else:
        typer.echo(f"first chunk: {result.time_to_first_chunk_ms:.1f} ms")
    typer.echo(f"total: {result.total_latency_ms:.1f} ms")
    typer.echo(f"stream chunks: {result.chunks}")


@app.command()
def metrics(url: str = "http://127.0.0.1:8000") -> None:
    """Print a scheduler snapshot from vLLM metrics."""
    snapshot = fetch_scheduler_metrics(url)
    kv = "n/a" if snapshot.kv_cache_usage is None else f"{snapshot.kv_cache_usage * 100:.1f}%"
    typer.echo(f"running={snapshot.running} waiting={snapshot.waiting} kv_cache={kv}")


if __name__ == "__main__":
    app()
