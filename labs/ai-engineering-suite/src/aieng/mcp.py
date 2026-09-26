"""Raw JSON-RPC stdio tools subset of MCP 2026-07-28, without an MCP SDK."""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from jsonschema import Draft202012Validator
from referencing import Registry

from .core import canonical
from .guardrails import scrub
from .sandbox import arithmetic

VERSION = "2026-07-28"
PREFIX = "io.modelcontextprotocol/"
MAX_MESSAGE = 65536
INFO = {"name": "aieng-stdio", "version": "0.1.0"}


def loads(text: str | bytes) -> Any:
    def invalid(value: str) -> None:
        raise ValueError("non-finite JSON number")
    return json.loads(text, parse_constant=invalid)


def validator(schema: dict[str, Any]) -> Draft202012Validator:
    """Empty registry disables remote reference retrieval. Internal refs still work."""
    Draft202012Validator.check_schema(schema)
    if schema.get("$schema", "https://json-schema.org/draft/2020-12/schema") != \
            "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("only JSON Schema draft 2020-12 is supported")
    return Draft202012Validator(schema, registry=Registry())


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    schema: dict[str, Any]
    run: Callable[[dict[str, Any]], Any]
    output_schema: dict[str, Any] | None = None


class Server:
    def __init__(self, tools: list[MCPTool], *, clock: Callable[[], float] = time.monotonic):
        if len({t.name for t in tools}) != len(tools):
            raise ValueError("duplicate tool name")
        self.tools = {t.name: t for t in tools}
        for tool in tools:
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", tool.name):
                raise ValueError("invalid tool name")
            validator(tool.schema)
            if tool.output_schema is not None:
                validator(tool.output_schema)
        self.calls: deque[float] = deque()
        self.clock = clock

    @staticmethod
    def error(identity: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": identity, "error": error}

    @staticmethod
    def result(identity: Any, **body: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": identity, "result": {
            "resultType": "complete", "_meta": {PREFIX + "serverInfo": INFO}, **body}}

    def handle(self, line: str | bytes) -> dict[str, Any] | None:
        if len(line) > MAX_MESSAGE:
            return self.error(None, -32600, "message too large")
        try:
            request = loads(line)
        except (ValueError, UnicodeError, RecursionError):
            return self.error(None, -32700, "invalid JSON")
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" \
                or not isinstance(request.get("method"), str):
            return self.error(None, -32600, "invalid request")
        # Notifications never receive responses; this server registers no notification handlers.
        if "id" not in request:
            return None
        identity = request["id"]
        if type(identity) not in (int, str):
            return self.error(None, -32600, "request ID must be an integer or string")
        params = request.get("params", {})
        if not isinstance(params, dict) or not isinstance(params.get("_meta"), dict):
            return self.error(identity, -32602, "per-request metadata required")
        meta = params["_meta"]
        if not isinstance(meta.get(PREFIX + "protocolVersion"), str) or \
                not isinstance(meta.get(PREFIX + "clientCapabilities"), dict):
            return self.error(identity, -32602, "protocolVersion and clientCapabilities required")
        version = meta[PREFIX + "protocolVersion"]
        if version != VERSION:
            return self.error(identity, -32022, "unsupported protocol version",
                              {"supported": [VERSION], "requested": version})
        method = request["method"]
        if method == "server/discover":
            return self.result(identity, supportedVersions=[VERSION], capabilities={"tools": {}})
        if method == "ping":
            return self.result(identity)
        if method == "tools/list":
            if params.get("cursor") is not None:
                return self.error(identity, -32602, "this bounded catalog has no cursors")
            tools = [{"name": t.name, "description": t.description, "inputSchema": t.schema,
                      **({"outputSchema": t.output_schema} if t.output_schema is not None else {})}
                     for t in sorted(self.tools.values(), key=lambda t: t.name)]
            return self.result(identity, tools=tools)
        if method != "tools/call":
            return self.error(identity, -32601, "unknown method; supported version: " + VERSION)
        name, args = params.get("name"), params.get("arguments", {})
        if not isinstance(name, str) or name not in self.tools or not isinstance(args, dict):
            return self.error(identity, -32602, "unknown tool or invalid call structure")
        now = self.clock()
        while self.calls and self.calls[0] <= now - 60:
            self.calls.popleft()
        if len(self.calls) >= 100:
            return self.result(identity, isError=True,
                               content=[{"type": "text", "text": "tool rate limit exceeded"}])
        self.calls.append(now)
        tool = self.tools[name]
        try:
            validator(tool.schema).validate(args)
            value = scrub(tool.run(args))
            if tool.output_schema is not None:
                validator(tool.output_schema).validate(value)
            text = canonical(value)
            if len(text.encode()) > MAX_MESSAGE // 4:
                raise ValueError("tool output too large")
            return self.result(identity, content=[{"type": "text", "text": text}],
                               structuredContent=value, isError=False)
        except Exception as exc:
            # Do not leak arguments, tracebacks, or exception strings across the boundary.
            return self.result(identity, isError=True,
                               content=[{"type": "text", "text": "tool failed: " + type(exc).__name__}])

    def serve(self) -> None:
        while True:
            line = sys.stdin.buffer.readline(MAX_MESSAGE + 1)
            if not line:
                break
            if len(line) > MAX_MESSAGE:
                response = self.error(None, -32600, "message too large")
                print(canonical(response), flush=True)
                break  # Do not parse the remainder of an oversized frame as another request.
            response = self.handle(line)
            if response is not None:
                print(canonical(response), flush=True)


class RPCError(RuntimeError):
    def __init__(self, error: dict[str, Any]):
        self.code = error.get("code")
        super().__init__(str(error.get("message", "RPC error")))


class Client:
    """One outstanding request at a time. Commands are trusted application configuration."""
    def __init__(self, command: list[str], *, timeout: float = 5):
        if not command or timeout <= 0:
            raise ValueError("command and positive timeout required")
        self.command, self.timeout = command, timeout
        self.process = None
        self.sequence = 0
        self.lock = asyncio.Lock()

    async def __aenter__(self) -> Client:
        self.process = await asyncio.create_subprocess_exec(
            *self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=MAX_MESSAGE)
        try:
            await self.request("server/discover")
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.process is not None:
            if self.process.returncode is None:
                self.process.kill()
            await self.process.wait()
            self.process = None

    async def request(self, method: str, **params: Any) -> dict[str, Any]:
        async with self.lock:
            if self.process is None or self.process.returncode is not None:
                raise RuntimeError("client is closed")
            self.sequence += 1
            params["_meta"] = {PREFIX + "protocolVersion": VERSION,
                               PREFIX + "clientCapabilities": {},
                               PREFIX + "clientInfo": {"name": "aieng-client", "version": "0.1.0"}}
            request = canonical({"jsonrpc": "2.0", "id": self.sequence,
                                 "method": method, "params": params}).encode() + b"\n"
            if len(request) > MAX_MESSAGE:
                raise ValueError("request too large")

            async def exchange() -> dict[str, Any]:
                self.process.stdin.write(request)
                await self.process.stdin.drain()
                response = loads(await self.process.stdout.readline())
                if not isinstance(response, dict) or response.get("jsonrpc") != "2.0" or \
                        response.get("id") != self.sequence:
                    raise ValueError("invalid or mismatched response")
                if ("result" in response) == ("error" in response):
                    raise ValueError("response must contain exactly one result or error")
                return response

            try:
                response = await asyncio.wait_for(exchange(), self.timeout)
            except BaseException:
                # A timed-out channel cannot safely be reused with unread frames.
                await self.__aexit__(None, None, None)
                raise
            if "error" in response:
                raise RPCError(response["error"])
            result = response["result"]
            if not isinstance(result, dict) or result.get("resultType") != "complete":
                raise ValueError("unsupported result shape")
            return result


def default_server() -> Server:
    return Server([MCPTool("arithmetic", "Bounded numeric arithmetic; no Python execution.",
                          {"type": "object", "properties": {"expression": {"type": "string", "maxLength": 512}},
                           "required": ["expression"], "additionalProperties": False},
                          lambda args: {"value": arithmetic(args["expression"])},
                          {"type": "object", "properties": {"value": {"type": "number"}},
                           "required": ["value"], "additionalProperties": False})])


if __name__ == "__main__":
    default_server().serve()
