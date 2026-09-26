"""Typed Execution Plan Intermediate Representation (IR) for AVENIQ.

Supports only:
- CALL
- PARALLEL
- SEQUENCE
- VERIFY
- RETURN
"""

from typing import Any, Dict, List, Literal, Optional
import uuid
from pydantic import BaseModel, Field

PlanOp = Literal["CALL", "PARALLEL", "SEQUENCE", "VERIFY", "RETURN"]


class PlanStep(BaseModel):
    """A single atomic or composite step in the deterministic execution IR.

    Operations:
    - CALL: Invokes a single expert with a prompt.
    - PARALLEL: Concurrently runs nested steps (typically CALLs).
    - SEQUENCE: Sequentially runs nested steps in order.
    - VERIFY: Evaluates prior output against criteria.
    - RETURN: Produces the final user-facing response.
    """

    op: PlanOp = Field(..., description="Operation type: CALL | PARALLEL | SEQUENCE | VERIFY | RETURN")
    expert: Optional[str] = Field(default=None, description="Expert identifier for CALL")
    prompt: Optional[str] = Field(default=None, description="Task instruction or query passed to CALL")
    output_key: Optional[str] = Field(default=None, description="Context variable name to store step result")
    steps: Optional[List["PlanStep"]] = Field(
        default=None,
        description="Nested child steps for PARALLEL or SEQUENCE",
    )
    verifier_expert: Optional[str] = Field(
        default=None,
        description="Verifier expert type (e.g. 'critical-thinker' or 'analytical')",
    )
    verification_criteria: Optional[str] = Field(
        default=None,
        description="Criteria or rubric for VERIFY step",
    )
    target_step_output: Optional[str] = Field(
        default=None,
        description="Context key of the output to be verified",
    )
    value: Optional[str] = Field(
        default=None,
        description="Static return string or formatted template for RETURN",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra step metadata",
    )


class ExecutionPlan(BaseModel):
    """Complete typed execution plan compiled from a PolicyDecision."""

    plan_id: str = Field(
        default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}",
        description="Unique execution plan ID",
    )
    execution_class: str = Field(
        ...,
        description="Source execution class: direct | single_expert | parallel_experts | system2",
    )
    steps: List[PlanStep] = Field(
        default_factory=list,
        description="Top-level steps of the plan",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Plan-level metadata and compilation trace",
    )

    def is_deterministic(self) -> bool:
        """Return True if this plan executes deterministically without System-2."""
        return self.execution_class in {"direct", "single_expert", "parallel_experts"}

    def is_system2(self) -> bool:
        """Return True if this plan delegates to the generative code orchestrator."""
        return self.execution_class == "system2"
