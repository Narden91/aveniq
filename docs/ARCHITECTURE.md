# AVENIQ Architecture

## Design objective

AVENIQ separates **policy**, **planning**, **execution**, and **measurement**.

The runtime must not depend on a specific System-1 model or a specific orchestration framework.

## Logical architecture

```text
                     Request
                        |
                        v
                  RequestState
                        |
                        v
                  PolicyEngine
                 /      |      \
                /       |       \
             direct   single   system2
               |        |        |
               |        |        v
               |        |   System2Planner
               |        |        |
               +--------+--------+
                        |
                        v
                  ExecutionPlan
                        |
                        v
                     Executor
                        |
                        v
                     Verifier
                        |
                        v
                   OutcomeTrace
```

## Core abstractions

### PolicyEngine

A replaceable decision layer.

Expected implementations:

```text
RulePolicy
LayaPolicy
future learned policies
```

The rest of the runtime should depend on the `PolicyEngine` contract rather than Laya directly.

### PolicyDecision

Typed policy output.

Current primary research fields:

```text
execution_class
primary_expert
confidence
```

The current execution-class space is intentionally small:

```text
direct
single_expert
system2
```

Additional dimensions should be introduced only after the primary routing problem is solved.

### ExecutionPlan

Typed representation of execution.

Current minimal plan operations:

```text
CALL
PARALLEL
SEQUENCE
VERIFY
RETURN
```

Generated Python is not the primary runtime representation.

### System2Planner

The inherited generative programmatic orchestrator.

It should be invoked only when the policy decides that deterministic execution is insufficient.

### Executor

Runs deterministic plans or the output of the System-2 path.

The inherited sandbox, AST validation, asynchronous expert execution, provider adapters, and retry logic remain useful infrastructure.

### Verifier

Determines whether the output satisfies the task requirement.

Verification quality is central because AVENIQ training labels depend on knowing which execution paths actually succeed.

Preferred verification order:

1. deterministic checks
2. executable tests
3. structured schema checks
4. reference checks
5. semantic / judge-based checks

### OutcomeTrace

Every execution should generate structured telemetry.

Typical fields:

```text
request_id
policy checkpoint
policy prediction
confidence
selected route
actual route
System-2 invoked
experts called
latency
provider tokens
estimated cost
task-check result
failure category
```

These traces form the basis for later specialization.

## System-1 policy

Current preferred zero-shot baseline:

```text
convaiinnovations/laya-typed-decisions
```

Current baselines:

```text
convaiinnovations/laya
RulePolicy
```

The architecture must remain model-agnostic.

## System-2 path

The original project generated async Python programs dynamically.

In AVENIQ, that mechanism remains useful but moves behind the policy layer.

Conceptually:

```text
request
   |
   v
PolicyEngine
   |
   +--> deterministic execution
   |
   +--> System2Planner
            |
            v
      generated program
            |
            v
       AST validation
            |
            v
          sandbox
```

## Research data flow

```text
candidate query
      |
      +--> direct ---------> outcome
      |
      +--> single expert --> outcome
      |
      +--> System-2 -------> outcome
                               |
                               v
                         quality checks
                               |
                               v
                     minimum sufficient route
                               |
                               v
                         training example
                               |
                               v
                        specialized policy
```

## Critical architectural rule

Training labels must be derived from observed route outcomes, not from:

- RulePolicy
- the System-2 planner's chosen route
- Laya's own prediction

Otherwise AVENIQ would imitate an existing router rather than learn minimum sufficient compute.

## Failure taxonomy

Keep failures distinct:

```text
routing_failure
runtime_failure
provider_token_limit
provider_timeout
task_check_failure
inference_error
```

Do not collapse provider failures into routing quality.

## Current implementation focus

The architecture currently optimizes for research validity rather than feature breadth.

Deferred:

- model-tier routing
- online learning
- MCP expansion
- richer workflow language
- UI redesign
- dynamic memory policy
- multi-GPU training
