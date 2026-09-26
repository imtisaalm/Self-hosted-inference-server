"""Context-local spans and W3C traceparent propagation with redacted JSONL output."""
from __future__ import annotations

import re
import secrets
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .core import canonical, positive_int
from .guardrails import scrub


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str
    sampled: bool = True

    def header(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-{'01' if self.sampled else '00'}"

    @classmethod
    def parse(cls, value: str) -> TraceContext:
        match = re.fullmatch(r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})", value)
        if not match or int(match[1], 16) == 0 or int(match[2], 16) == 0:
            raise ValueError("invalid W3C version-00 traceparent")
        return cls(match[1], match[2], bool(int(match[3], 16) & 1))


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    started_ns: int
    ended_ns: int = 0
    duration_ns: int = 0
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """An educational span exporter, NOT an OpenTelemetry SDK or OTLP endpoint."""

    def __init__(self, path: str | Path | None = None, capacity: int = 1000) -> None:
        self.path = Path(path) if path else None
        self.capacity = positive_int(capacity, "capacity")
        self.spans: list[Span] = []
        self.dropped = 0
        self.export_errors = 0
        self.lock = threading.Lock()
        self.current: ContextVar[TraceContext | None] = ContextVar(f"trace-{id(self)}", default=None)

    @contextmanager
    def span(self, name: str, *, parent: TraceContext | None = None, **attributes: Any) -> Iterator[Span]:
        if not name or len(name) > 128:
            raise ValueError("span name must contain 1 to 128 characters")
        previous = parent or self.current.get()
        context = TraceContext(previous.trace_id if previous else secrets.token_hex(16), secrets.token_hex(8), previous.sampled if previous else True)
        token = self.current.set(context)
        value = Span(name, context.trace_id, context.span_id, previous.span_id if previous else None, time.time_ns(), attributes=scrub(attributes))
        started = time.perf_counter_ns()
        try:
            yield value
        except BaseException as exc:
            value.status = "error"
            value.attributes["exception.type"] = type(exc).__name__
            # Exception messages can contain credentials; they are intentionally omitted.
            raise
        finally:
            value.ended_ns = time.time_ns()
            value.duration_ns = max(0, time.perf_counter_ns() - started)
            value.attributes = scrub(value.attributes)
            self.current.reset(token)
            if context.sampled:
                with self.lock:
                    if len(self.spans) >= self.capacity:
                        self.spans.pop(0)
                        self.dropped += 1
                    self.spans.append(value)
                    if self.path:
                        try:
                            with self.path.open("a", encoding="utf-8") as stream:
                                stream.write(canonical(asdict(value)) + "\n")
                        except OSError:
                            self.export_errors += 1  # Telemetry must not mask application errors.

    def traceparent(self) -> str | None:
        context = self.current.get()
        return context.header() if context else None
