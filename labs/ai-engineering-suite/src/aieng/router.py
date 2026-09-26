"""Constraint-based routing with charged-attempt budgets and circuit breakers."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar

from .core import number, positive_int

T = TypeVar("T")


@dataclass(frozen=True)
class Model:
    name: str
    input_per_million: float
    output_per_million: float
    latency_ms: float
    quality: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model name required")
        for field in ("input_per_million", "output_per_million", "latency_ms", "quality"):
            number(getattr(self, field), field)
        if self.quality > 1:
            raise ValueError("quality must be in [0,1]")

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_million + output_tokens * self.output_per_million) / 1_000_000


@dataclass(frozen=True)
class RouteResult(Generic[T]):
    model: str
    value: T
    attempts: tuple[str, ...]
    reserved_cost: float


class RoutingFailed(RuntimeError):
    def __init__(self, attempts: list[str], reserved_cost: float) -> None:
        super().__init__("no eligible model completed within the request constraints")
        self.attempts, self.reserved_cost = tuple(attempts), reserved_cost


class ModelRouter:
    """Providers must cooperate with cancellation and honor the output-token cap.

    Costs are estimates from operator-supplied prices, never a billing guarantee.
    Failed attempts consume reservations because providers may charge for them.
    One router instance belongs to one asyncio event loop.
    """

    def __init__(self, models: list[Model], failures: int = 2, cooldown: float = 30, clock: Callable[[], float] = time.monotonic) -> None:
        if not models or len({model.name for model in models}) != len(models):
            raise ValueError("models must be nonempty and uniquely named")
        positive_int(failures, "failures")
        number(cooldown, "cooldown", 0.000001)
        self.models, self.failures, self.cooldown, self.clock = models, failures, cooldown, clock
        self.health: dict[str, tuple[int, float]] = {}

    async def run(
        self, provider: Callable[[Model, int], Awaitable[T]], *, input_tokens: int,
        output_tokens: int, budget: float, deadline_s: float = 10,
        min_quality: float = 0, max_latency_ms: float = 10_000,
    ) -> RouteResult[T]:
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0:
            raise ValueError("input_tokens must be a nonnegative integer")
        positive_int(output_tokens, "output_tokens")
        number(budget, "budget")
        number(deadline_s, "deadline_s", 0.000001)
        number(max_latency_ms, "max_latency_ms", 0.000001)
        if not 0 <= number(min_quality, "min_quality") <= 1:
            raise ValueError("min_quality must be <= 1")
        candidates = sorted(
            (model for model in self.models if model.quality >= min_quality and model.latency_ms <= max_latency_ms),
            key=lambda model: (model.estimate(input_tokens, output_tokens), model.latency_ms, -model.quality, model.name),
        )
        start, spent, attempted = self.clock(), 0.0, []
        for model in candidates:
            count, until = self.health.get(model.name, (0, 0))
            if until > self.clock():
                continue
            if until:
                count = 0
            remaining = deadline_s - (self.clock() - start)
            cost = model.estimate(input_tokens, output_tokens)
            if remaining <= 0:
                break
            if spent + cost > budget + 1e-12:
                continue
            spent += cost
            attempted.append(model.name)
            try:
                value = await asyncio.wait_for(provider(model, output_tokens), timeout=remaining)
            except (TimeoutError, ConnectionError, OSError):
                count += 1
                self.health[model.name] = (count, self.clock() + self.cooldown if count >= self.failures else 0)
                continue
            # Programming and validation errors propagate instead of being hidden as provider failures.
            self.health[model.name] = (0, 0)
            return RouteResult(model.name, value, tuple(attempted), spent)
        raise RoutingFailed(attempted, spent)
