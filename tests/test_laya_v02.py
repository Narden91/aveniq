"""Regression tests for AVENIQ v0.2 LayaPolicy semantics (disabled, shadow, control)."""

import pytest
from src.aveniq.policy.laya_policy import LayaPolicy
from src.aveniq.policy.rule_policy import RulePolicy
from src.aveniq.policy.decision import PolicyDecision


class TestLayaPolicyV02:
    """Validate Laya disabled, shadow, control modes and runtime metadata."""

    def test_disabled_mode_semantics(self):
        """In disabled mode, Laya is not called and fallback policy governs."""
        policy = LayaPolicy(mode="disabled")
        decision = policy.evaluate("Hello there")
        assert decision.execution_class == "direct"
        meta = decision.metadata.get("laya_metadata", {})
        assert meta["latency_ms"] is None
        assert meta["controlled_execution"] is False
        assert meta["fallback_occurred"] is False
        assert meta["token_usage"] is None  # Non-generative null tokens

    def test_shadow_mode_does_not_alter_routing(self):
        """In shadow mode, Laya runs inference but does NOT alter the execution path."""
        # Query that RulePolicy routes to single_expert:technical
        query = "Explain Python GIL and multi-threading"
        policy = LayaPolicy(mode="shadow", fallback_policy=RulePolicy())
        decision = policy.evaluate(query)

        # Must strictly follow fallback policy
        assert decision.execution_class == "single_expert"
        assert decision.primary_expert == "technical"

        # Shadow prediction must be recorded
        assert "laya_shadow" in decision.metadata
        shadow = decision.metadata["laya_shadow"]
        assert shadow["checkpoint"] == "laya-router-v0.2"
        assert shadow["device"] is not None
        assert isinstance(shadow["latency_ms"], float)
        assert shadow["latency_ms"] >= 0.0
        assert shadow["confidence"] is not None
        assert "probabilities" in shadow
        assert "execution_class" in shadow["probabilities"]
        assert "primary_expert" in shadow["probabilities"]
        assert shadow["controlled_execution"] is False
        assert shadow["fallback_occurred"] is False
        assert shadow["fallback_reason"] is None
        assert shadow["token_usage"] is None  # Null tokens, not 0

    def test_control_mode_high_confidence_routes_directly(self):
        """In control mode with confidence >= threshold, Laya controls the execution path."""
        policy = LayaPolicy(mode="control", confidence_threshold=0.70)
        # Direct greeting query
        decision = policy.evaluate("Hello, what can you do?")
        assert decision.execution_class == "direct"
        meta = decision.metadata.get("laya_metadata", {})
        assert meta["controlled_execution"] is True
        assert meta["fallback_occurred"] is False
        assert meta["fallback_reason"] is None

        # Technical single expert query
        tech_decision = policy.evaluate("Write a Python function to sort a list using quicksort")
        assert tech_decision.execution_class == "single_expert"
        assert tech_decision.primary_expert == "technical"
        tech_meta = tech_decision.metadata.get("laya_metadata", {})
        assert tech_meta["controlled_execution"] is True
        assert tech_meta["fallback_occurred"] is False

    def test_control_mode_low_confidence_escalates_to_system2(self):
        """When confidence < threshold, Laya control mode escalates to System-2."""
        # Set an artificially high threshold to guarantee fallback trigger
        policy = LayaPolicy(mode="control", confidence_threshold=0.99)
        decision = policy.evaluate("Write an algorithm to sort numbers")
        assert decision.execution_class == "system2"
        meta = decision.metadata.get("laya_metadata", {})
        assert meta["controlled_execution"] is True
        assert meta["fallback_occurred"] is True
        assert meta["fallback_reason"] == "low_confidence"

    def test_inference_error_fallback_resilience(self):
        """If Laya inference raises an exception, the pipeline falls back safely."""
        policy = LayaPolicy(mode="control")
        # Mock engine to simulate an unexpected runtime error
        policy._engine.predict = lambda q: (_ for _ in ()).throw(RuntimeError("Simulated device failure"))

        decision = policy.evaluate("Hello")
        assert decision.execution_class == "direct"  # Fallback to RulePolicy
        meta = decision.metadata.get("laya_metadata", {})
        assert meta["controlled_execution"] is False
        assert meta["fallback_occurred"] is True
        assert "Simulated device failure" in meta["fallback_reason"]

    def test_runtime_metadata_no_fabricated_tokens(self):
        """Verify that Laya metadata uses null (None) rather than 0 for non-applicable token counts."""
        policy = LayaPolicy(mode="control")
        decision = policy.evaluate("Hi")
        meta = decision.metadata.get("laya_metadata", {})
        assert meta["token_usage"] is None

    def test_probability_distributions_sum_to_one(self):
        """Probabilities for execution_class and primary_expert must be valid distributions."""
        policy = LayaPolicy(mode="control")
        pred = policy.predict_laya("Compare SQL and NoSQL databases for metrics")
        class_probs = pred["probabilities"]["execution_class"]
        expert_probs = pred["probabilities"]["primary_expert"]

        assert pytest.approx(sum(class_probs.values()), abs=0.01) == 1.0
        assert pytest.approx(sum(expert_probs.values()), abs=0.01) == 1.0
        assert pred["execution_class"] in {"direct", "single_expert", "system2"}
        if pred["execution_class"] == "single_expert":
            assert pred["primary_expert"] in {"technical", "analytical", "creative", "general"}
