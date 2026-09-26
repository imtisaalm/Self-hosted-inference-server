# AI Engineering Systems

Small implementations of common LLM infrastructure components, built without agent frameworks.

## Projects

- Context assembler
- Retrieval stack
- Model router
- Semantic cache
- Agent orchestrator
- MCP server and client
- Multi-agent consensus
- Sandboxed tool executor
- Guardrails middleware
- Durable workflow engine
- Streaming proxy
- LLM tracer
- Eval harness
- Prompt registry
- Data flywheel

Python code lives in `src/aieng`. The streaming proxy is in `go/streamproxy`.

## Setup

Requires Python 3.11+ and Go 1.23+.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Test

```sh
make check
```

Docker integration tests are separate:

```sh
docker pull python:3.12-slim
AIENG_DOCKER_TESTS=1 python -m pytest -q tests/test_docker_integration.py
```

## Run

Run the full local demo:

```sh
python -m aieng all
```

Or run one component:

```sh
python -m aieng retrieval-stack
python -m aieng model-router
python -m aieng mcp
python -m aieng durable-workflows
```

For the Go streaming proxy:

```sh
cd go/streamproxy
go run . -demo
```

Implementation notes are in `docs/PROJECTS.md`.
