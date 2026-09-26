# AVENIQ Roadmap

## Current stage

AVENIQ is in the **policy-learning research phase**.

The architecture has already demonstrated:

- deterministic bypass of System-2
- real Laya inference
- shadow and control modes
- fixed dataset splits
- typed routing metrics
- outcome tracing
- zero-shot checkpoint comparison

The next priority is not feature expansion.

The next priority is to learn a better compute policy.

---

# v0.3 — Typed Policy Baseline

Status: baseline established.

Canonical result:

```text
Base Laya:
    accuracy: 40.0%
    System-2 rate: 90.0%
    false bypass: 0/15
    unnecessary escalation: 30/35

Laya typed-decisions:
    accuracy: 64.0%
    System-2 rate: 60.0%
    false bypass: 0/15
    unnecessary escalation: 15/35

RulePolicy:
    accuracy: 52.0%
    System-2 rate: 2.0%
    false bypass: 14/15
    unnecessary escalation: 0/35
```

Decision:

```text
convaiinnovations/laya-typed-decisions
```

becomes the default zero-shot System-1 research checkpoint.

Provider-backed specialization is not considered complete.

---

# v0.4 — Learned Compute Policy

## Goal

Train an AVENIQ-specific execution-class policy.

Target labels:

```text
direct
single_expert
system2
```

## Dataset

Initial target:

```text
>= 500 counterfactual candidates
```

Preferred longer-term target:

```text
1,500–3,000 verified examples
```

Prioritize deterministic validation tasks.

## Training

1. derive minimum-sufficient-compute labels;
2. fine-tune Laya;
3. calibrate probabilities;
4. select confidence threshold on validation;
5. run untouched final test;
6. run provider-backed downstream comparison.

## Success criteria

Compared with typed Laya:

```text
false bypass: remains near zero
unnecessary escalation: materially lower
System-2 rate: materially lower
task quality: preserved
cost per successful task: lower
```

---

# v0.5 — Adaptive Cost Policy

Only if v0.4 succeeds.

Potential additions:

```text
model-tier selection
parallel-expert routing
verification policy
budget-aware decisions
```

Research question:

> Can AVENIQ choose not only the execution class, but also the amount of model capacity and verification required?

---

# v0.6 — Continuous Policy Improvement

Only after offline specialization is stable.

Potential work:

```text
policy regret
drift detection
selective exploration
counterfactual replay
periodic retraining
```

Do not introduce uncontrolled online learning.

---

# v1.0 candidate

A credible v1.0 should demonstrate:

```text
stable typed policy API
measured cost reduction
measured latency reduction
strong false-bypass control
reproducible benchmarks
portable runtime adapters
clear provider separation
production traceability
```

The release should be based on measurable adaptive-compute value, not feature count.

---

## Deferred until evidence exists

Do not prioritize before v0.4:

- major frontend redesign
- MCP expansion
- additional workflow DSL features
- more memory systems
- additional agent personas
- broad provider proliferation
- autonomous online learning
- complex model-tier routing
- compression research
