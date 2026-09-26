"""Budgeted context selection with indivisible dependency groups."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .core import number, positive_int


def byte_count(text: str) -> int:
    """Conservative byte count, NOT a model tokenizer. Inject a model-specific counter."""
    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class ContextItem:
    id: str
    text: str
    role: str = "user"
    priority: float = 1
    pinned: bool = False
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssembledContext:
    items: tuple[ContextItem, ...]
    used: int
    available: int
    omitted: tuple[str, ...]


class ContextAssembler:
    """Greedy priority selection, preserving input order and all dependencies.

    A tool result can require its call and schema. None is truncated mid-JSON.
    Budget covers every selected item's content plus explicit framing overhead.
    Optimal knapsack utility is deliberately traded for inspectable O(n^2) selection.
    """

    def __init__(self, count: Callable[[str], int] = byte_count, framing: int = 8) -> None:
        if not isinstance(framing, int) or isinstance(framing, bool) or framing < 0:
            raise ValueError("framing must be a nonnegative integer")
        self.count, self.framing = count, framing

    def assemble(self, items: Iterable[ContextItem], limit: int, reserve: int = 0) -> AssembledContext:
        positive_int(limit, "limit")
        if not isinstance(reserve, int) or isinstance(reserve, bool) or not 0 <= reserve < limit:
            raise ValueError("reserve must be an integer in [0, limit)")
        values = tuple(items)
        index = {item.id: item for item in values}
        if len(index) != len(values) or any(not item.id for item in values):
            raise ValueError("context IDs must be nonempty and unique")
        costs: dict[str, int] = {}
        for item in values:
            number(item.priority, "priority")
            cost = self.count(item.text)
            if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
                raise ValueError("counter must return a nonnegative integer")
            costs[item.id] = cost + self.framing
        closures: dict[str, set[str]] = {}

        def dependencies(key: str, visiting: frozenset[str] = frozenset()) -> set[str]:
            if key not in index:
                raise ValueError(f"missing context dependency: {key}")
            if key in visiting:
                raise ValueError("cyclic context dependencies")
            if key not in closures:
                result = {key}
                for dep in index[key].requires:
                    result |= dependencies(dep, visiting | {key})
                closures[key] = result
            return closures[key]

        for item in values:
            dependencies(item.id)
        selected: set[str] = set()
        for item in values:
            if item.pinned:
                selected |= closures[item.id]
        available = limit - reserve
        used = sum(costs[key] for key in selected)
        if used > available:
            raise ValueError("pinned context exceeds available budget")
        # Python's stable sort preserves source order for ties.
        for item in sorted(values, key=lambda value: -value.priority):
            extra = closures[item.id] - selected
            cost = sum(costs[key] for key in extra)
            if used + cost <= available:
                selected |= extra
                used += cost
        return AssembledContext(
            tuple(item for item in values if item.id in selected), used, available,
            tuple(item.id for item in values if item.id not in selected),
        )
