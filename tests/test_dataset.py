"""Unit tests for adaptive routing dataset loading and schema validation."""

import pytest
from pathlib import Path
from benchmarks.dataset import AdaptiveRoutingRecord, load_adaptive_routing_dataset, default_dataset_path


def test_load_default_dataset():
    """Verify that the checked-in adaptive routing dataset loads and validates successfully."""
    dataset = load_adaptive_routing_dataset()
    assert len(dataset) >= 20

    # Ensure every record matches the schema
    for record in dataset:
        assert isinstance(record, AdaptiveRoutingRecord)
        assert record.id.startswith("route_")
        assert len(record.query) > 0
        assert record.expected_execution_class in {"direct", "single_expert", "system2"}
        if record.expected_execution_class == "single_expert":
            assert record.expected_primary_expert in {"technical", "analytical", "creative", "general"}
        else:
            assert record.expected_primary_expert is None


def test_dataset_class_distributions():
    """Verify that the dataset contains examples for direct, single_expert, and system2."""
    dataset = load_adaptive_routing_dataset()
    classes = {r.expected_execution_class for r in dataset}
    assert "direct" in classes
    assert "single_expert" in classes
    assert "system2" in classes

    experts = {r.expected_primary_expert for r in dataset if r.expected_primary_expert}
    assert "technical" in experts
    assert "analytical" in experts
    assert "creative" in experts
    assert "general" in experts
