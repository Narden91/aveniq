"""Dataset models and loaders for AVENIQ adaptive routing research benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


ExecutionClassType = Literal["direct", "single_expert", "system2"]
ExpertType = Literal["technical", "analytical", "creative", "general"]
SplitType = Literal["development", "validation", "test"]


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
    split: SplitType
    provenance: Dict[str, str]
    deterministic_validation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for deterministic validation",
    )

    @model_validator(mode="after")
    def check_label(self) -> "AdaptiveRoutingRecord":
        if (self.expected_execution_class == "single_expert") != (self.expected_primary_expert is not None):
            raise ValueError("primary expert is required exactly for single_expert cases")
        if not self.provenance.get("origin") or not self.provenance.get("label_basis"):
            raise ValueError("every case needs an origin and label basis")
        return self

    def to_benchmark_case(self) -> Dict[str, Any]:
        """Convert to dictionary representation for benchmark harnesses."""
        return {
            "name": self.id,
            "query": self.query,
            "expected_execution_class": self.expected_execution_class,
            "expected_primary_expert": self.expected_primary_expert,
            "tags": list(self.tags),
            "split": self.split,
            "provenance": dict(self.provenance),
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

    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate routing case ID in {file_path}")
    queries = [record.query.strip().casefold() for record in records]
    if len(queries) != len(set(queries)):
        raise ValueError(f"Duplicate routing query in {file_path}")
    return records
