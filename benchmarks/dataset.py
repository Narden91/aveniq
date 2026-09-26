"""Dataset models and loaders for AVENIQ adaptive routing research benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


ExecutionClassType = Literal["direct", "single_expert", "system2"]
ExpertType = Literal["technical", "analytical", "creative", "general"]


class AdaptiveRoutingRecord(BaseModel):
    """Schema for a gold-labeled adaptive routing benchmark record."""

    id: str = Field(..., description="Unique scenario identifier")
    query: str = Field(..., description="User query text")
    expected_execution_class: ExecutionClassType = Field(
        ...,
        description="Ground-truth minimum sufficient execution class: direct, single_expert, or system2",
    )
    expected_primary_expert: Optional[ExpertType] = Field(
        None,
        description="Ground-truth primary expert for single_expert routing (null for direct or system2)",
    )
    tags: List[str] = Field(default_factory=list, description="Categorical and task tags")
    deterministic_validation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for deterministic validation",
    )

    @field_validator("expected_primary_expert")
    @classmethod
    def validate_expert_for_class(cls, v: Optional[str], info: Any) -> Optional[str]:
        # For single_expert, expected_primary_expert is expected to be present
        return v

    def to_benchmark_case(self) -> Dict[str, Any]:
        """Convert to dictionary representation for benchmark harnesses."""
        return {
            "name": self.id,
            "query": self.query,
            "expected_execution_class": self.expected_execution_class,
            "expected_primary_expert": self.expected_primary_expert,
            "tags": list(self.tags),
        }


def default_dataset_path() -> Path:
    """Return the default path to adaptive_routing.jsonl."""
    return Path(__file__).parent / "data" / "adaptive_routing.jsonl"


def load_adaptive_routing_dataset(
    path: Optional[str | Path] = None,
) -> List[AdaptiveRoutingRecord]:
    """Load and validate the gold-labeled adaptive routing benchmark dataset.

    Parameters
    ----------
    path : str | Path, optional
        Path to the JSONL dataset. Defaults to benchmarks/data/adaptive_routing.jsonl.

    Returns
    -------
    List[AdaptiveRoutingRecord]
        Validated list of benchmark scenarios.
    """
    file_path = Path(path) if path is not None else default_dataset_path()
    if not file_path.exists():
        raise FileNotFoundError(f"Adaptive routing dataset not found at: {file_path}")

    records: List[AdaptiveRoutingRecord] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                record = AdaptiveRoutingRecord.model_validate(data)
                records.append(record)
            except Exception as exc:
                raise ValueError(
                    f"Invalid record at line {line_num} in {file_path}: {exc}"
                ) from exc

    return records
