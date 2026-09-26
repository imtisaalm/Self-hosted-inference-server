"""A bounded arithmetic language and a fail-closed Docker Python executor."""
from __future__ import annotations

import ast
import math
import operator
import selectors
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass

from .core import number, positive_int


class SandboxUnavailable(RuntimeError):
    pass


class ExecutionLimit(RuntimeError):
    pass


def arithmetic(expression: str) -> float:
    """Interpret a deliberately tiny language. Never calls eval/exec or arbitrary names."""
    if not isinstance(expression, str) or len(expression) > 512:
        raise ValueError("expression exceeds 512 characters")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 96:
        raise ValueError("expression too complex")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Mod: operator.mod}
    unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def visit(node: ast.AST, depth: int = 0) -> float:
        if depth > 20:
            raise ValueError("expression too deep")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            value = binary[type(node.op)](visit(node.left, depth + 1), visit(node.right, depth + 1))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            value = unary[type(node.op)](visit(node.operand, depth + 1))
        else:
            raise ValueError("only finite numeric arithmetic is allowed")
        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("numeric bound exceeded")
        return value

    return visit(tree.body)


@dataclass(frozen=True)
class Execution:
    returncode: int
    stdout: str
    stderr: str


class DockerExecutor:
    """Host Docker must be administered separately. No host fallback, mounts, or secrets."""
    def __init__(self, *, image: str = "python:3.12-slim", timeout_s: float = 3,
                 memory_mb: int = 128, output_bytes: int = 65536):
        if not image or image.startswith("-") or any(c.isspace() for c in image):
            raise ValueError("invalid image")
        self.image = image
        self.timeout = number(timeout_s, "timeout_s")
        if not self.timeout:
            raise ValueError("positive timeout required")
        self.memory = positive_int(memory_mb, "memory_mb")
        self.output_bytes = positive_int(output_bytes, "output_bytes")

    def command(self, name: str) -> list[str]:
        return ["docker", "run", "--rm", "--pull=never", "--name", name,
                "--network=none", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--user=65534:65534",
                "--pids-limit=32", "--cpus=0.5", f"--memory={self.memory}m",
                f"--memory-swap={self.memory}m", "--ulimit=nofile=64:64",
                "--ulimit=fsize=1048576:1048576", "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
                "-i", self.image, "python", "-I", "-B", "-"]

    def execute(self, code: str) -> Execution:
        if not isinstance(code, str) or len(code.encode()) > 8192:
            raise ValueError("code must be UTF-8 text of at most 8192 bytes")
        if not shutil.which("docker"):
            raise SandboxUnavailable("Docker is unavailable; refusing host execution")
        name = "aieng-" + uuid.uuid4().hex
        process = subprocess.Popen(self.command(name), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        deadline = time.monotonic() + self.timeout
        try:
            # Input is bounded below pipe capacity; the child consumes it immediately.
            process.stdin.write(code.encode())
            process.stdin.close()
            with selectors.DefaultSelector() as selector:
                for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
                    selector.register(stream, selectors.EVENT_READ, label)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ExecutionLimit("wall time exceeded")
                    for selected, _ in selector.select(min(remaining, 0.1)):
                        chunk = selected.fileobj.read1(4096)
                        if not chunk:
                            selector.unregister(selected.fileobj)
                            continue
                        buffers[selected.data].extend(chunk)
                        if sum(map(len, buffers.values())) > self.output_bytes:
                            raise ExecutionLimit("output limit exceeded")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ExecutionLimit("wall time exceeded")
            process.wait(timeout=remaining)
            return Execution(process.returncode, buffers["stdout"].decode("utf-8", "replace"),
                             buffers["stderr"].decode("utf-8", "replace"))
        except subprocess.TimeoutExpired as exc:
            raise ExecutionLimit("wall time exceeded") from exc
        finally:
            # Killing the CLI alone does not guarantee that the container stops.
            try:
                subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=5, check=False)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream and not stream.closed:
                        stream.close()
