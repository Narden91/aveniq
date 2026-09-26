"""OutcomeTrace schema and local persistence for AVENIQ (Phase 5)."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class OutcomeTrace(BaseModel):
    """Complete trace of an execution outcome for evaluation and dataset building.

    Attributes
    ----------
    request_id:
        Unique identifier for the request.
    policy_decision:
        Dictionary representation of the PolicyDecision.
    execution_path:
        Actual path taken: 'direct' | 'single_expert' | 'parallel_experts' | 'system2'.
    generated_plan_or_script:
        Plan ID or Python script content if System-2 was invoked.
    experts_called:
        List of expert agents called during execution.
    system2_invoked:
        Whether the generative System-2 orchestrator was executed.
    latency_seconds:
        End-to-end execution latency in seconds.
    input_tokens:
        Input/prompt token count.
    output_tokens:
        Completion/output token count.
    total_tokens:
        Sum of input and output tokens.
    estimated_cost_usd:
        Estimated monetary cost of provider calls.
    success:
        Whether the request succeeded without unhandled fatal error.
    verification_score:
        Verification score if verification was performed.
    error:
        Error message or exception string if failed.
    timestamp:
        Unix timestamp when trace was generated.
    query:
        Original user query prompt.
    final_answer_snippet:
        Snippet of the final produced answer.
    shadow_predictions:
        Predictions recorded in shadow mode (e.g. Laya).
    """

    request_id: str = Field(
        default_factory=lambda: f"req-{uuid.uuid4().hex[:10]}",
        description="Unique request ID",
    )
    policy_decision: Dict[str, Any] = Field(
        default_factory=dict,
        description="PolicyDecision dictionary",
    )
    execution_path: str = Field(
        ...,
        description="Execution path: direct | single_expert | parallel_experts | system2",
    )
    generated_plan_or_script: Optional[str] = Field(
        default=None,
        description="Generated plan ID or orchestration script reference",
    )
    experts_called: List[str] = Field(
        default_factory=list,
        description="Experts called during execution",
    )
    system2_invoked: bool = Field(
        default=False,
        description="True if OrchestratorAgent was invoked",
    )
    latency_seconds: float = Field(
        default=0.0,
        description="End-to-end execution duration in seconds",
    )
    input_tokens: int = Field(
        default=0,
        description="Total input tokens consumed",
    )
    output_tokens: int = Field(
        default=0,
        description="Total output tokens produced",
    )
    total_tokens: int = Field(
        default=0,
        description="Total tokens consumed",
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        description="Estimated monetary cost in USD",
    )
    success: bool = Field(
        default=True,
        description="Whether the request succeeded",
    )
    verification_score: Optional[float] = Field(
        default=None,
        description="Verification score if verified",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details if execution encountered an error",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Timestamp of the trace",
    )
    query: str = Field(
        default="",
        description="The user request prompt",
    )
    final_answer_snippet: str = Field(
        default="",
        description="Prefix of final answer",
    )
    shadow_predictions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Shadow predictor outputs (e.g. Laya)",
    )


class OutcomeTraceStore:
    """Persists OutcomeTrace records to a local JSONL file."""

    DEFAULT_TRACE_PATH = ".aveniq_traces.jsonl"

    def __init__(self, trace_file: Optional[str | Path] = None):
        self.trace_file = Path(trace_file or self.DEFAULT_TRACE_PATH)

    def append(self, trace: OutcomeTrace) -> None:
        """Append a trace to the local storage."""
        try:
            line = trace.model_dump_json() + "\n"
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            # Failure to log trace must never crash the pipeline
            pass

    def load_all(self) -> List[OutcomeTrace]:
        """Read all saved traces."""
        if not self.trace_file.exists():
            return []
        traces = []
        with open(self.trace_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        traces.append(OutcomeTrace.model_validate_json(stripped))
                    except Exception:
                        continue
        return traces
