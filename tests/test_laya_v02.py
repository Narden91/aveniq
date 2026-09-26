"""Policy-mode tests use explicitly named mocks; hardware checks live in research/laya_smoke.py."""

import pytest

from src.aveniq.policy.laya_policy import DEFAULT_CHECKPOINT, LayaPolicy
from src.aveniq.policy.rule_policy import RulePolicy


@pytest.fixture
def mock_laya_prediction():
    return {
        "checkpoint": DEFAULT_CHECKPOINT,
        "device": "cuda:0",
        "latency_ms": 12.0,
        "forward_latency_ms": 12.0,
        "preprocessing_latency_ms": 2.0,
        "total_policy_latency_ms": 15.0,
        "execution_class": "single_expert",
        "primary_expert": "technical",
        "confidence": 0.85,
        "probabilities": {
            "execution_class": {"direct": 0.05, "single_expert": 0.85, "system2": 0.10},
            "primary_expert": {
                "technical": 0.85,
                "analytical": 0.05,
                "creative": 0.05,
                "general": 0.05,
            },
        },
        "backend": "mock_laya_prediction",
        "token_usage": None,
    }


def test_disabled_mode_does_not_load_model(monkeypatch):
    policy = LayaPolicy(mode="disabled")
    monkeypatch.setattr(policy._engine, "predict", lambda _query: pytest.fail("Laya was called"))
    decision = policy.evaluate("Hello")
    assert decision.execution_class == "direct"
    assert decision.metadata["laya_metadata"]["checkpoint"] is None
    assert decision.metadata["laya_metadata"]["token_usage"] is None


def test_shadow_keeps_rule_route(monkeypatch, mock_laya_prediction):
    policy = LayaPolicy(mode="shadow", fallback_policy=RulePolicy())
    monkeypatch.setattr(policy._engine, "predict", lambda _query: mock_laya_prediction)
    decision = policy.evaluate("Hello")
    assert decision.execution_class == "direct"
    assert decision.metadata["laya_shadow"]["execution_class"] == "single_expert"
    assert decision.metadata["laya_shadow"]["backend"] == "mock_laya_prediction"
    assert decision.metadata["laya_shadow"]["controlled_execution"] is False


def test_control_uses_mock_laya_decision(monkeypatch, mock_laya_prediction):
    policy = LayaPolicy(mode="control", confidence_threshold=0.8)
    monkeypatch.setattr(policy._engine, "predict", lambda _query: mock_laya_prediction)
    decision = policy.evaluate("Explain Python threading")
    assert decision.execution_class == "single_expert"
    assert decision.primary_expert == "technical"
    assert decision.metadata["laya_metadata"]["controlled_execution"] is True
    assert decision.token_usage is None


def test_control_escalates_below_threshold(monkeypatch, mock_laya_prediction):
    policy = LayaPolicy(mode="control", confidence_threshold=0.9)
    monkeypatch.setattr(policy._engine, "predict", lambda _query: mock_laya_prediction)
    decision = policy.evaluate("Explain Python threading")
    assert decision.execution_class == "system2"
    assert decision.metadata["laya_metadata"]["fallback_reason"] == "low_confidence"


def test_inference_error_is_recorded_as_fallback(monkeypatch):
    policy = LayaPolicy(mode="control")

    def mock_failing_laya(_query):
        raise RuntimeError("simulated device failure")

    monkeypatch.setattr(policy._engine, "predict", mock_failing_laya)
    decision = policy.evaluate("Hello")
    assert decision.execution_class == "direct"
    assert decision.metadata["laya_metadata"]["fallback_occurred"] is True
    assert "simulated device failure" in decision.metadata["laya_metadata"]["fallback_reason"]
