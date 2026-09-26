"""Unit tests for AVENIQ policy abstraction (Phase 1)."""

import pytest
from pydantic import ValidationError

from src.aveniq.policy import (
    ExecutionClass,
    LayaPolicy,
    PolicyDecision,
    PolicyEngine,
    RulePolicy,
)


class TestPolicyDecision:
    """Test validation and behavior of PolicyDecision Pydantic model."""

    def test_valid_decision_creation(self):
        decision = PolicyDecision(
            execution_class="single_expert",
            primary_expert="technical",
            model_tier="fast",
            needs_verification=False,
            max_agent_calls=1,
            confidence=0.92,
        )
        assert decision.execution_class == "single_expert"
        assert decision.primary_expert == "technical"
        assert decision.model_tier == "fast"
        assert decision.needs_verification is False
        assert decision.max_agent_calls == 1
        assert decision.confidence == 0.92

    def test_confidence_boundary_validation(self):
        # Valid bounds: 0.0 and 1.0
        d_min = PolicyDecision(
            execution_class="direct",
            needs_verification=False,
            max_agent_calls=1,
            confidence=0.0,
        )
        assert d_min.confidence == 0.0

        d_max = PolicyDecision(
            execution_class="system2",
            needs_verification=True,
            max_agent_calls=3,
            confidence=1.0,
        )
        assert d_max.confidence == 1.0

        # Invalid bounds
        with pytest.raises(ValidationError):
            PolicyDecision(
                execution_class="direct",
                needs_verification=False,
                max_agent_calls=1,
                confidence=-0.1,
            )

        with pytest.raises(ValidationError):
            PolicyDecision(
                execution_class="direct",
                needs_verification=False,
                max_agent_calls=1,
                confidence=1.01,
            )

    def test_max_agent_calls_must_be_positive(self):
        with pytest.raises(ValidationError):
            PolicyDecision(
                execution_class="single_expert",
                primary_expert="technical",
                needs_verification=False,
                max_agent_calls=0,  # invalid, must be > 0
                confidence=0.5,
            )

        with pytest.raises(ValidationError):
            PolicyDecision(
                execution_class="single_expert",
                primary_expert="technical",
                needs_verification=False,
                max_agent_calls=-2,
                confidence=0.5,
            )

    def test_invalid_execution_class(self):
        with pytest.raises(ValidationError):
            PolicyDecision(
                execution_class="invalid_class",  # type: ignore
                needs_verification=False,
                max_agent_calls=1,
                confidence=0.5,
            )


class TestPolicyEngineProtocol:
    """Verify that concrete implementations satisfy the PolicyEngine protocol."""

    def test_rule_policy_implements_protocol(self):
        policy = RulePolicy()
        assert isinstance(policy, PolicyEngine)

    def test_laya_policy_implements_protocol(self):
        policy = LayaPolicy()
        assert isinstance(policy, PolicyEngine)


class TestRulePolicy:
    """Test deterministic heuristics in RulePolicy."""

    def setup_method(self):
        self.policy = RulePolicy()

    def test_direct_routing_for_greetings(self):
        decision = self.policy.evaluate("Hello")
        assert decision.execution_class == "direct"
        assert decision.confidence == 1.0
        assert decision.model_tier == "local"

    def test_single_expert_technical(self):
        decision = self.policy.evaluate("Write a Python function to sort a list using quicksort")
        assert decision.execution_class == "single_expert"
        assert decision.primary_expert == "technical"
        assert decision.confidence >= 0.7
        assert decision.model_tier == "fast"

    def test_single_expert_creative(self):
        decision = self.policy.evaluate("Write a haiku poem about an autumn evening")
        assert decision.execution_class == "single_expert"
        assert decision.primary_expert == "creative"
        assert decision.confidence >= 0.7

    def test_single_expert_analytical(self):
        decision = self.policy.evaluate("Compare PostgreSQL and MongoDB pros and cons and tradeoffs")
        assert decision.execution_class == "single_expert"
        assert decision.primary_expert == "analytical"
        assert decision.confidence >= 0.7

    def test_system2_for_complex_query(self):
        decision = self.policy.evaluate("Plan and execute a multi-step workflow pipeline across all agents")
        assert decision.execution_class == "system2"
        assert decision.model_tier == "strong"
        assert decision.needs_verification is True

    def test_verification_requirement_keyword(self):
        decision = self.policy.evaluate("Write a python function to compute primes and verify correctness strictly")
        assert decision.execution_class == "single_expert"
        assert decision.needs_verification is True

    @pytest.mark.asyncio
    async def test_async_evaluate(self):
        decision = await self.policy.aevaluate("Hello there")
        assert decision.execution_class == "direct"


class TestLayaPolicy:
    """Test LayaPolicy mode behavior with explicit mock predictions."""

    @staticmethod
    def mock_laya_prediction(_query):
        return {"execution_class": "single_expert", "primary_expert": "technical",
                "confidence": 0.9, "probabilities": {"execution_class": {
                    "direct": 0.05, "single_expert": 0.9, "system2": 0.05}},
                "backend": "mock_laya_prediction"}

    def test_shadow_mode_execution(self, monkeypatch):
        policy = LayaPolicy(shadow_mode=True)
        monkeypatch.setattr(policy, "predict_laya", self.mock_laya_prediction)
        decision = policy.evaluate("Write a Python script to parse JSON")
        # In shadow mode, execution path is governed by fallback policy (RulePolicy)
        assert decision.execution_class == "single_expert"
        assert decision.primary_expert == "technical"
        # Shadow predictions are recorded
        assert "laya_shadow" in decision.raw_predictions
        laya_info = decision.raw_predictions["laya_shadow"]
        assert "confidence" in laya_info
        assert "probabilities" in laya_info
        assert "backend" in laya_info

    def test_active_mode_execution(self, monkeypatch):
        policy = LayaPolicy(shadow_mode=False)
        monkeypatch.setattr(policy, "predict_laya", self.mock_laya_prediction)
        decision = policy.evaluate("Write a simple function")
        assert decision.execution_class == "single_expert"
        assert "laya" in decision.raw_predictions
