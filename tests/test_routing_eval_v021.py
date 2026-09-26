"""Checks for routing counts, fixed splits, and separate success terms."""

import asyncio
import hashlib
from collections import Counter
from types import SimpleNamespace

from benchmarks.dataset import load_adaptive_routing_dataset
from benchmarks.suite import BenchmarkCase, BenchmarkReport, BenchmarkResult, BenchmarkSuite
from research.downstream_eval import RecordingLayaPolicy
from research.routing_eval import CONFIG, choose_threshold, routing_metrics
from src.aveniq.policy.laya_policy import LayaPolicy


def mock_prediction(label, confidence, probabilities):
    return {
        "execution_class": label,
        "primary_expert": None,
        "confidence": confidence,
        "probabilities": {"execution_class": probabilities},
    }


def test_corpus_counts_and_provenance():
    records = load_adaptive_routing_dataset()
    assert len(records) == 200
    assert Counter(record.expected_execution_class for record in records) == {
        "direct": 40,
        "single_expert": 100,
        "system2": 60,
    }
    assert Counter(record.split for record in records) == {
        "development": 100,
        "validation": 50,
        "test": 50,
    }
    assert all(
        record.provenance["origin"] and record.provenance["label_basis"] for record in records
    )
    for execution_class in ("direct", "single_expert", "system2"):
        group = [record for record in records if record.expected_execution_class == execution_class]
        group.sort(
            key=lambda record: hashlib.sha256(
                f"{CONFIG['corpus_split_seed']}:{record.id}".encode()
            ).digest()
        )
        development_end = len(group) // 2
        validation_end = development_end + len(group) // 4
        expected = (
            ["development"] * development_end
            + ["validation"] * (validation_end - development_end)
            + ["test"] * (len(group) - validation_end)
        )
        assert [record.split for record in group] == expected


def test_false_bypass_uses_system2_denominator():
    rows = [
        {
            "expected_execution_class": "system2",
            "expected_primary_expert": None,
            "prediction": mock_prediction(
                "single_expert", 0.8, {"direct": 0.1, "single_expert": 0.8, "system2": 0.1}
            ),
        },
        {
            "expected_execution_class": "direct",
            "expected_primary_expert": None,
            "prediction": mock_prediction(
                "direct", 0.8, {"direct": 0.8, "single_expert": 0.1, "system2": 0.1}
            ),
        },
    ]
    metrics = routing_metrics(rows, 0.5)
    assert metrics["false_bypass"] == {"count": 1, "total": 1, "percent": 100.0}
    assert metrics["execution_class_accuracy"] == {"count": 1, "total": 2, "percent": 50.0}
    assert 0 <= metrics["multiclass_brier_score_raw_laya"] <= 2
    assert 0 <= metrics["ece_raw_laya"] <= 1


def test_validation_threshold_and_unchecked_task_success():
    rows = [
        {
            "expected_execution_class": "system2",
            "prediction": mock_prediction(
                "direct", 0.3, {"direct": 0.5, "single_expert": 0.2, "system2": 0.3}
            ),
        },
    ]
    assert choose_threshold(rows)["threshold"] > 0.3
    report = BenchmarkReport(
        results=[
            BenchmarkResult(
                case=BenchmarkCase(name="case", query="Hi"), success=True, elapsed_seconds=0.1
            )
        ]
    )
    assert report.success_rate_pct == 100.0
    assert report.task_success_rate_pct is None
    assert report.summary()["task_success_rate_pct"] is None


def test_provider_usage_reads_real_tracker_fields():
    class MockGraph:
        async def ainvoke(self, state):
            return {
                **state,
                "metadata": {"execution_path": "direct"},
                "token_usage": {
                    "total_input_tokens": 13,
                    "total_output_tokens": 7,
                    "estimated_cost_usd": 0.001,
                },
                "final_answer": "OK",
            }

    suite = BenchmarkSuite()
    suite.add(BenchmarkCase(name="usage", query="Reply with OK"))
    result = asyncio.run(suite.run_all(MockGraph(), is_mock_provider=False)).results[0]
    assert result.actual_provider_input_tokens == 13
    assert result.actual_provider_output_tokens == 7
    assert result.execution_success is True
    assert result.task_success is None


def test_recording_policy_keeps_metadata_before_downstream_error(monkeypatch):
    record = {"checkpoint": "mock-checkpoint", "forward_latency_ms": 3.0}
    mock_decision = SimpleNamespace(metadata={"laya_metadata": record})
    monkeypatch.setattr(LayaPolicy, "evaluate", lambda self, query, context=None: mock_decision)
    policy = RecordingLayaPolicy(mode="disabled")
    assert policy.evaluate("test request") is mock_decision
    assert policy.last_record == record
