# AVENIQ Documentation Index

This folder contains the high-level project framing and research handoff.

## Start here

### `HANDOFF.md`

Current state, next milestone, preserved assumptions, research guardrails, and the exact next work sequence.

Use this for:

- Codex
- a new engineer
- a new research session
- project continuation after a break

### `PROJECT_VISION.md`

Explains:

- what AVENIQ is
- what problem it solves
- why it is not another multi-agent framework
- the System-1 / System-2 thesis
- long-term product and research value

### `ARCHITECTURE.md`

Explains:

- PolicyEngine
- PolicyDecision
- ExecutionPlan
- System2Planner
- Executor
- Verifier
- OutcomeTrace
- research data flow

### `TECHNOLOGIES.md`

Documents:

- Python
- Laya
- PyTorch
- RTX 5080 research environment
- LangGraph
- Pydantic
- pytest
- Ruff
- uv
- JSONL research artifacts
- inherited agent/runtime infrastructure

### `RESEARCH_METHOD.md`

Defines:

- research question
- counterfactual labeling
- false bypass
- unnecessary escalation
- primary metrics
- split discipline
- threshold selection
- reproducibility requirements

### `ROADMAP.md`

Tracks:

- v0.3 typed-policy baseline
- v0.4 learned compute policy
- later stages only if v0.4 succeeds

## Recommended repository placement

Suggested layout:

```text
docs/
  PROJECT_VISION.md
  ARCHITECTURE.md
  TECHNOLOGIES.md
  RESEARCH_METHOD.md
  ROADMAP.md

HANDOFF.md
```

`HANDOFF.md` should remain at repository root so coding agents see it immediately.

`DOCS_INDEX.md` may be renamed to `docs/README.md` when copied into the repository.
