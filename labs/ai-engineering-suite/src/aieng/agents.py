"""An explicit bounded state machine; the planner cannot choose arbitrary functions."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping


from .core import canonical, positive_int
from .guardrails import Guardrails
from .tracing import Tracer
from .mcp import validator


@dataclass(frozen=True)
class Tool:
    schema: dict[str, Any]
    run: Callable[[dict[str, Any]], Any]
    sensitive: bool = False


@dataclass(frozen=True)
class Action:
    kind: str  # "tool" or "finish"
    name: str = ""
    arguments: dict[str, Any] | None = None
    answer: str = ""


@dataclass(frozen=True)
class AgentResult:
    answer: str
    trajectory: tuple[dict[str, Any], ...]


class AgentStopped(RuntimeError):
    pass


class Agent:
    def __init__(self, tools: Mapping[str, Tool], *, max_steps: int = 8,
                 guardrails: Guardrails | None = None, tracer: Tracer | None = None):
        self.tools = dict(tools)
        self.max_steps = positive_int(max_steps, "max_steps")
        self.guardrails = guardrails or Guardrails()
        self.tracer = tracer or Tracer()
        for name, tool in self.tools.items():
            if not name:
                raise ValueError("tool name required")
            validator(tool.schema)

    def run(self, question: str, planner: Callable[[str, tuple[dict[str, Any], ...]], Action],
            *, approve: Callable[[str, dict[str, Any]], bool] | None = None) -> AgentResult:
        safe_question = self.guardrails.inbound(question)
        history: list[dict[str, Any]] = []
        with self.tracer.span("agent"):
            for _ in range(self.max_steps):
                action = planner(safe_question, tuple(copy.deepcopy(history)))
                if not isinstance(action, Action):
                    raise AgentStopped("planner returned an invalid action")
                if action.kind == "finish":
                    answer = self.guardrails.inbound(action.answer)
                    history.append({"state": "done"})
                    return AgentResult(answer, tuple(history))
                if action.kind != "tool" or action.name not in self.tools:
                    raise AgentStopped("tool is outside the allowlist")
                tool = self.tools[action.name]
                args = copy.deepcopy(action.arguments)
                if not isinstance(args, dict) or len(canonical(args)) > 8192:
                    raise AgentStopped("invalid tool arguments")
                validator(tool.schema).validate(args)
                if tool.sensitive and (approve is None or not approve(action.name, copy.deepcopy(args))):
                    raise AgentStopped("explicit approval required")
                with self.tracer.span("tool", tool_name=action.name):
                    value = tool.run(args)
                    # Treat tool output as untrusted. Never promote it into system instructions.
                    text = self.guardrails.inbound(canonical(value))
                history.append({"state": "observed", "tool": action.name, "observation": text})
        raise AgentStopped("step budget exhausted")
