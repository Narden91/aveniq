# AVENIQ Technologies

## Core language

### Python 3.11+

Primary implementation language for:

- runtime
- policies
- planning
- execution
- benchmarks
- research tooling
- APIs

## Policy model

### Laya

Used as the current System-1 decision model.

Relevant properties:

- non-autoregressive
- typed decisions
- local inference
- calibrated probabilities
- supports choice / score / binary-style decisions
- compatible with LangGraph integration
- supports task-specific fine-tuning

Current baseline checkpoint:

```text
convaiinnovations/laya-typed-decisions
```

Research baseline:

```text
convaiinnovations/laya
```

## Deep learning runtime

### PyTorch

Used by Laya for model inference and training.

Current research hardware:

```text
NVIDIA GeForce RTX 5080 Laptop GPU
Blackwell
cuda:0
```

Observed real Laya inference in v0.2.1 separated preprocessing from neural forward-pass timing.

## Agent workflow

### LangGraph

Used as an execution/orchestration adapter.

Role in AVENIQ:

- route between policy paths
- preserve existing graph workflow
- manage execution state
- support System-2 fallback flow

LangGraph is infrastructure, not the project identity.

## Validation and data models

### Pydantic

Used for typed schemas such as:

- `PolicyDecision`
- `ExecutionPlan`
- `OutcomeTrace`
- benchmark dataset records

## Provider layer

The inherited project supports external LLM providers.

Current research has used Groq-backed execution.

Provider usage should remain isolated from policy logic.

All research comparisons must use identical downstream provider configuration across competing policies.

## System-2 orchestration

Inherited components include:

- generative orchestrator
- code generation
- AST analysis
- sandboxed execution
- expert dispatch
- asynchronous execution
- retry behavior

In AVENIQ these components form the expensive fallback path.

## Testing

### pytest

Used for:

- policy tests
- plan tests
- graph regression tests
- research integrity checks

### Ruff

Used for linting and code-quality checks.

## Environment / dependency tooling

### uv

Used for:

- virtual environments
- dependency installation
- lockfile management
- reproducible Python environments

## Repository analysis

### Graft

Used during development to inspect and refresh the repository context graph.

It is a development aid, not a runtime dependency.

## Storage formats

### JSONL

Preferred for:

- routing datasets
- counterfactual outcomes
- training examples
- execution traces

Benefits:

- append-friendly
- line-addressable
- easy to validate
- easy to hash
- easy to stream

### JSON

Used for:

- benchmark summaries
- manifests
- frozen hashes
- aggregate metrics
- configuration snapshots

## Research artifact policy

Canonical artifacts should include:

```text
dataset
split manifest
configuration
checkpoint identifier
raw predictions
raw route outcomes
aggregate results
environment metadata
hashes
```

Partial or interrupted runs should remain preserved but must be stored separately from canonical results.
