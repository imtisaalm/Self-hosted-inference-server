# Projects

## Context assembler
Selects context under a fixed budget. Pinned items are kept, dependencies are included together, and remaining items are selected by priority.

## Retrieval stack
Chunks documents, runs BM25 and vector search, merges the rankings with reciprocal-rank fusion, then reranks the candidates.

## Model router
Filters models by quality and latency, estimates request cost, retries transient provider failures, and opens a circuit after repeated failures.

## Semantic cache
Stores responses in SQLite and matches queries by cosine similarity. Cache keys are scoped by tenant, model, prompt version, embedding model, and corpus version.

## Agent orchestrator
Runs a bounded tool loop. Tools are allowlisted and schema-validated. Sensitive tools require an approval callback.

## MCP server and client
Implements the stdio tools path with JSON-RPC directly. The client handles request IDs, timeouts, and subprocess cleanup.

## Multi-agent consensus
Aggregates weighted ballots with a quorum. Split decisions can be passed to a judge or escalated.

## Sandboxed tool executor
Provides a restricted arithmetic evaluator and optional Python execution in a resource-limited Docker container.

## Guardrails middleware
Normalizes input, flags a small set of prompt-injection patterns, and redacts common PII and secret formats.

## Durable workflow engine
Persists workflow state in SQLite. Checkpoints support restart/resume, while leases and fencing prevent stale workers from committing.

## Streaming proxy
A Go SSE proxy with authentication, concurrency limits, cancellation, and first-content/inter-event timing.

## LLM tracer
Records nested spans, propagates W3C trace context, and writes redacted JSONL traces.

## Eval harness
Runs named cases, grades results and tool trajectories, and compares them with a checked-in baseline.

## Prompt registry
Stores immutable prompt versions, tracks the active version, supports rollback, and assigns stable weighted experiments.

## Data flywheel
Stores approved feedback, splits data by source, optionally augments training prompts, and trains a small LoRA example.
