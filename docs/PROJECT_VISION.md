# AVENIQ Project Vision

## One-sentence definition

**AVENIQ is an adaptive compute runtime that learns the minimum amount of agentic reasoning required to solve each request reliably.**

## Problem

Modern agent systems often default to expensive execution:

- always call a large model
- always invoke a planner
- always use multiple agents
- always run verification
- always construct a dynamic workflow

This produces unnecessary:

- latency
- token usage
- provider cost
- orchestration overhead
- failure surface

Many requests do not require the full agent stack.

The core AVENIQ hypothesis is:

> A fast local policy can identify when expensive planning is unnecessary and safely route requests to cheaper execution paths.

## Product thesis

AVENIQ should not compete as another general agent framework.

It should sit above or beside agent runtimes and answer:

```text
What is the minimum sufficient execution path for this request?
```

Possible paths include:

```text
direct
single expert
parallel experts
System-2 planner
```

The project currently focuses on:

```text
direct
single expert
System-2
```

to keep the research problem controlled.

## System-1 / System-2 framing

AVENIQ uses a two-level control model.

### System-1

Fast, typed, local decision policy.

Responsibilities:

- classify execution class
- choose specialist when relevant
- estimate confidence
- decide whether to escalate

Current implementation candidate:

- Laya

### System-2

Expensive generative reasoning path.

Responsibilities:

- synthesize complex plans
- generate orchestration logic
- manage dependency-heavy tasks
- recover from uncertainty
- solve tasks requiring adaptive planning

Current implementation:

- inherited programmatic multi-agent orchestrator

## Long-term value proposition

The system should eventually demonstrate:

```text
similar task quality
lower System-2 invocation rate
lower cost per successful task
lower p95 latency
fewer unnecessary agent calls
```

The core business value is not “more agents”.

It is:

> Avoid paying for reasoning that the request does not need.

## Research thesis

A suitable research framing is:

> Learning when not to orchestrate: adaptive compute allocation for agentic systems.

The central scientific question is:

> Can a specialized local decision policy preserve task success while reducing unnecessary use of expensive generative planning?

## What AVENIQ is not

AVENIQ is not intended to be primarily:

- a LangGraph wrapper
- a generic multi-agent framework
- a prompt library
- a visual workflow builder
- a provider aggregator
- a generated-Python demo
- a model benchmark repository

Those may exist as implementation components, but they are not the project identity.

## Success criteria

AVENIQ is successful when it can show, on held-out tasks:

1. very low false bypass rate;
2. materially fewer unnecessary escalations;
3. lower System-2 invocation rate;
4. preserved task quality;
5. lower cost per successful task;
6. lower end-to-end latency.

## Current position

The current typed Laya baseline is promising but still conservative:

```text
held-out class accuracy:   64.0%
System-2 invocation:       60.0%
false bypass:              0/15
unnecessary escalation:    15/35
```

The next phase is specialization, not feature expansion.
