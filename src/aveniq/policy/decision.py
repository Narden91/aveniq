"""Typed policy decision schema for AVENIQ adaptive compute runtime."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ExecutionClass = Literal["direct", "single_expert", "parallel_experts", "system2"]
ModelTier = Literal["local", "fast", "strong"]


class PolicyDecision(BaseModel):
    """Structured decision made by a PolicyEngine.

    Attributes
    ----------
    execution_class:
        Routing class:
        - 'direct': Immediate answer without expert dispatch.
        - 'single_expert': Route to a single specialized expert deterministically.
        - 'parallel_experts': Route to multiple experts in parallel deterministically.
        - 'system2': Fall back to generative programmatic code orchestrator.
    primary_expert:
        Expert identifier (e.g., 'technical', 'creative', 'analytical', 'general')
        when applicable.
    model_tier:
        Resource tier for model inference: 'local', 'fast', or 'strong'.
    needs_verification:
        Whether the output requires a verification step before returning.
    max_agent_calls:
        Upper bound on the number of expert invocations (must be > 0).
    confidence:
        Confidence score in [0.0, 1.0].
    parallel_experts:
        List of expert identifiers to invoke when execution_class is 'parallel_experts'.
    reasoning:
        Human-readable explanation of why this policy decision was chosen.
    raw_predictions:
        Raw backend scores, logits, or probabilities (e.g., from Laya or classifiers).
    metadata:
        Additional policy metadata or feature values.
    """

    execution_class: ExecutionClass = Field(
        ...,
        description="Routing class: direct | single_expert | parallel_experts | system2",
    )
    primary_expert: Optional[str] = Field(
        default=None,
        description="Primary expert type to call for single_expert or lead expert",
    )
    model_tier: ModelTier = Field(
        default="fast",
        description="Model tier: local | fast | strong",
    )
    needs_verification: bool = Field(
        default=False,
        description="Whether a verification step is required",
    )
    max_agent_calls: int = Field(
        default=1,
        gt=0,
        description="Maximum number of agent calls allowed (must be > 0)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in [0.0, 1.0]",
    )
    parallel_experts: List[str] = Field(
        default_factory=list,
        description="Expert types to invoke in parallel for parallel_experts",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning or rule explanation for this decision",
    )
    raw_predictions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw probabilities/scores from backend or classifier",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary policy engine metadata",
    )
    token_usage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Token usage for generative policies (null for non-generative Laya)",
    )

