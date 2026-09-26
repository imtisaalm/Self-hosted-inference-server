"""Bounded, transparent heuristics; authorization remains a separate boundary."""
from __future__ import annotations

import re
import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from .core import positive_int

PII_PATTERNS = (
    ("EMAIL", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\w)")),
    ("SECRET", re.compile(r"\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}\b")),
)
INJECTION_PATTERNS = (
    ("instruction_override", re.compile(r"\b(?:ignore|disregard|override)\b.{0,60}\b(?:previous|prior|system|developer)\b.{0,40}\binstructions?\b", re.I)),
    ("secret_extraction", re.compile(r"\b(?:reveal|print|show|exfiltrate)\b.{0,50}\b(?:system prompt|api key|password|secret token)\b", re.I)),
    ("role_spoofing", re.compile(r"(?:<\|(?:system|developer)\|>|\[INST\]|<<SYS>>)", re.I)),
)


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join("".join(ch for ch in text if unicodedata.category(ch) != "Cf").split())


def redact(text: str) -> str:
    result = unicodedata.normalize("NFKC", text)
    result = "".join(ch for ch in result if unicodedata.category(ch) != "Cf")
    for label, pattern in PII_PATTERNS:
        result = pattern.sub(f"[{label}]", result)
    return result


def scrub(value: Any, depth: int = 0) -> Any:
    """Bounded trace-safe projection, including sensitive key names."""
    if depth > 6:
        return "[DEPTH_LIMIT]"
    if isinstance(value, str):
        return redact(value[:8192])
    if isinstance(value, dict):
        out = {}
        for key, item in list(value.items())[:64]:
            key = str(key)
            sensitive = any(part in key.casefold() for part in ("password", "secret", "authorization", "api_key"))
            out[redact(key)] = "[REDACTED]" if sensitive else scrub(item, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [scrub(item, depth + 1) for item in value[:64]]
    if isinstance(value, float) and not math.isfinite(value):
        return "[NONFINITE]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


@dataclass(frozen=True)
class Inspection:
    text: str
    flags: tuple[str, ...]
    redacted: bool


class GuardrailBlocked(ValueError):
    def __init__(self, flags: tuple[str, ...]) -> None:
        super().__init__("input blocked: " + ", ".join(flags))
        self.flags = flags


class Guardrails:
    def __init__(self, max_chars: int = 32_000) -> None:
        self.max_chars = positive_int(max_chars, "max_chars")

    def inspect(self, text: str) -> Inspection:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if len(text) > self.max_chars:
            raise GuardrailBlocked(("input_size",))
        cleaned = normalized(text)
        flags = tuple(name for name, pattern in INJECTION_PATTERNS if pattern.search(cleaned))
        safe = redact(text)
        return Inspection(safe, flags, safe != text)

    def inbound(self, text: str) -> str:
        result = self.inspect(text)
        if result.flags:
            raise GuardrailBlocked(result.flags)
        return result.text

    def call(self, text: str, handler: Callable[[str], str]) -> str:
        answer = handler(self.inbound(text))
        # Output has its own size check and redaction, without interpreting quoted instructions.
        return self.inspect(answer).text
