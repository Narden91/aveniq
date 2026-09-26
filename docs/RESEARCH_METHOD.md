# AVENIQ Research Method

## Research question

> Can a specialized local decision policy identify the minimum sufficient execution path for a request while preserving task quality and reducing expensive System-2 planning?

## Execution classes

Current controlled research space:

```text
direct
single_expert
system2
```

The smaller action space is intentional.

## Primary risk

A cheap route may be selected for a task that actually needs System-2.

This is a **false bypass**.

False bypass is more serious than unnecessary escalation.

### False bypass

Expected route:

```text
system2
```

Policy route:

```text
direct
or
single_expert
```

### Unnecessary escalation

Expected route:

```text
direct
or
single_expert
```

Policy route:

```text
system2
```

## Primary metrics

Report raw counts and percentages.

Required:

```text
false bypass
unnecessary escalation
System-2 invocation rate
task success
cost per successful task
p50 latency
p95 latency
```

Secondary:

```text
execution-class accuracy
expert-selection accuracy
Brier score
ECE
precision
recall
confusion matrix
```

## Counterfactual labeling

Each training candidate should be run through all applicable routes:

```text
direct
single_expert
system2
```

Use identical downstream configuration.

Each route produces:

```text
task result
task-check result
latency
input tokens
output tokens
estimated cost
failure category
```

The training label is:

> the lowest-cost route that passes the configured quality check.

If no route passes, the example is not used as a policy label.

If the run is interrupted by provider limits or infrastructure failure, the example remains incomplete and is excluded.

## Quality checking

Prefer deterministic checks.

Priority order:

1. exact answer
2. executable code tests
3. structured field validation
4. numerical tolerance
5. deterministic transformation checks
6. semantic judge

Open-ended semantic tasks should not dominate early training because noisy labels directly degrade routing quality.

## Data splits

Keep distinct:

```text
training
calibration
validation
final test
```

Rules:

- final-test examples never enter training
- calibration data fits probability calibration
- validation selects confidence thresholds
- final test is used only after design decisions are frozen

## Threshold selection

AVENIQ uses asymmetric risk.

A threshold should be selected to strongly penalize false bypass.

Conceptually:

```text
loss =
    high_penalty * false_bypass
    + lower_penalty * unnecessary_escalation
```

Do not optimize threshold solely for classification accuracy.

## Current baseline

Held-out zero-shot routing:

| Policy | Accuracy | System-2 rate | False bypass | Unnecessary escalation |
| --- | ---: | ---: | ---: | ---: |
| Base Laya | 40.0% | 90.0% | 0/15 | 30/35 |
| Laya typed-decisions | 64.0% | 60.0% | 0/15 | 15/35 |
| RulePolicy | 52.0% | 2.0% | 14/15 | 0/35 |

This establishes three characteristic policies:

```text
Base Laya:
    safe but highly conservative

RulePolicy:
    cheap but unsafe

Typed Laya:
    useful middle ground
```

## v0.4 hypothesis

An AVENIQ-specific fine-tuned Laya policy can:

- retain near-zero false bypass
- reduce unnecessary escalation below 15/35
- reduce System-2 invocation below 60%
- preserve downstream task quality

## Provider-backed experiments

Provider-backed comparisons must use identical:

```text
provider
model
max output tokens
timeout
retry settings
task checker
```

Provider quota failures must be reported separately.

Estimated costs must be labeled as estimates unless they come from actual billing data.

## Reproducibility

Every canonical experiment should save:

```text
dataset hash
split IDs
configuration
checkpoint identifier
threshold
package versions
GPU metadata
raw predictions
raw outcomes
aggregate metrics
timestamp
```

Do not overwrite previous canonical results.
