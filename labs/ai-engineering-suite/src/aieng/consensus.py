"""Weighted quorum, independent agent identities, judging, and explicit escalation."""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Sequence

from .core import number
from .guardrails import normalized


@dataclass(frozen=True)
class Ballot:
    agent: str
    answer: str


@dataclass(frozen=True)
class Decision:
    answer: str | None
    status: str
    support: float
    participation: float
    candidates: tuple[str, ...]


class Consensus:
    """Weights are operator-configured, never taken from self-reported confidence.

    Agreement is an aggregation signal, not evidence of factual correctness.
    Missing agents remain in the denominator; duplicate identities are rejected.
    """

    def __init__(self, weights: Mapping[str, float], threshold: float = 2 / 3, quorum: float = 2 / 3) -> None:
        if not weights or any(not name for name in weights):
            raise ValueError("nonempty agent weights required")
        self.weights = {name: number(weight, "weight", 0.000001) for name,weight in weights.items()}
        if not math.isfinite(sum(self.weights.values())):
            raise ValueError("total weight must be finite")
        self.threshold, self.quorum = number(threshold, "threshold"), number(quorum, "quorum")
        if not 0.5 < threshold <= 1 or not 0 < quorum <= 1:
            raise ValueError("threshold must be in (.5,1], quorum in (0,1]")

    def decide(self, ballots: Sequence[Ballot], judge: Callable[[tuple[str, ...]], str | None] | None = None) -> Decision:
        seen: set[str] = set()
        scores: dict[str, float] = {}
        display: dict[str, str] = {}
        for ballot in ballots:
            if ballot.agent not in self.weights or ballot.agent in seen:
                raise ValueError("unknown or duplicate agent identity")
            seen.add(ballot.agent)
            answer = normalized(ballot.answer).casefold()
            if not answer:
                continue  # Abstentions do not add participation.
            scores[answer] = scores.get(answer,0) + self.weights[ballot.agent]
            display.setdefault(answer,ballot.answer.strip())
        total = sum(self.weights.values())
        participation = sum(scores.values()) / total
        ranked = sorted(scores, key=lambda answer: (-scores[answer],answer))
        candidates = tuple(display[answer] for answer in ranked)
        support = scores[ranked[0]] / total if ranked else 0.0
        if participation < self.quorum or not ranked:
            return Decision(None,"escalated",support,participation,candidates)
        if support >= self.threshold:
            return Decision(candidates[0],"consensus",support,participation,candidates)
        if judge:
            choice = judge(candidates)
            if choice in candidates:
                return Decision(choice,"judged",support,participation,candidates)
        return Decision(None,"escalated",support,participation,candidates)

    async def collect(self, agents: Mapping[str, Callable[[], Awaitable[str]]], timeout: float = 3) -> list[Ballot]:
        number(timeout,"timeout",0.000001)
        if any(name not in self.weights for name in agents):
            raise ValueError("unknown agent")

        async def one(name: str, call: Callable[[], Awaitable[str]]) -> Ballot | None:
            try:
                return Ballot(name,await asyncio.wait_for(call(),timeout))
            except (TimeoutError, ConnectionError, OSError):
                return None

        results = await asyncio.gather(*(one(name,call) for name,call in agents.items()))
        return [ballot for ballot in results if ballot is not None]
