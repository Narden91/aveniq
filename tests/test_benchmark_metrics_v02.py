"""Regression tests for AVENIQ v0.2 research benchmark metrics and semantics."""

import pytest
from benchmarks.suite import (
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkResult,
)


class TestBenchmarkMetricsSemantics:
    """Validate that metrics adhere strictly to research-valid definitions."""

    def test_shadow_mode_does_not_alter_routing_or_sys2_stats(self):
        """Shadow mode must not alter actual execution path or System-2 invocation stats."""
        # Baseline policy routed to single_expert, while Laya shadow predicted system2
        case = BenchmarkCase(
            name="test_1",
            query="Explain Python GIL",
            expected_execution_class="single_expert",
            expected_primary_expert="technical",
        )
        res = BenchmarkResult(
            case=case,
            success=True,
            elapsed_seconds=0.012,
            execution_class="single_expert",  # Actual controlling path
            execution_path="single_expert",
            system2_invoked=False,  # Actual execution did not invoke System-2
            laya_shadow_prediction={
                "execution_class": "system2",
                "confidence": 0.88,
            },
        )
        report = BenchmarkReport(results=[res])

        # System-2 invocation rate must reflect actual execution (0%), not Laya prediction
        assert report.system2_invocation_rate_pct == 0.0
        # False bypass must be 0% since expected is single_expert and routed to single_expert
        assert report.false_bypass_rate == 0.0
        assert report.false_bypass_count == 0
        assert report.unnecessary_escalation_rate == 0.0
        assert report.execution_class_accuracy == 100.0

    def test_laya_tokens_are_null_not_zero(self):
        """Laya is non-generative; token usage must be null/None, never fabricated as 0."""
        from src.aveniq.policy import LayaPolicy

        policy = LayaPolicy(mode="control")
        decision = policy.evaluate("What is 2 + 2?")
        laya_meta = decision.metadata.get("laya_metadata", {})

        assert laya_meta.get("token_usage") is None
        assert decision.token_usage is None

    def test_mock_provider_tokens_separated_from_real_savings(self):
        """Mock provider tokens must be reported as null/None rather than real savings."""
        case = BenchmarkCase(name="mock_case", query="query")
        # Result from mock provider
        res_mock = BenchmarkResult(
            case=case,
            success=True,
            elapsed_seconds=0.05,
            token_summary={"prompt_tokens": 120, "completion_tokens": 40},
            is_mock_provider=True,
            actual_provider_input_tokens=None,
            actual_provider_output_tokens=None,
            estimated_cost_usd=None,
        )
        report = BenchmarkReport(results=[res_mock])

        # Provider tokens must be None (null in JSON export)
        assert report.actual_provider_input_tokens is None
        assert report.actual_provider_output_tokens is None
        assert report.estimated_provider_cost is None

        # Result from real provider
        res_real = BenchmarkResult(
            case=case,
            success=True,
            elapsed_seconds=0.25,
            token_summary={"prompt_tokens": 100, "completion_tokens": 50, "estimated_cost_usd": 0.0015},
            is_mock_provider=False,
            actual_provider_input_tokens=100,
            actual_provider_output_tokens=50,
            estimated_cost_usd=0.0015,
        )
        report_real = BenchmarkReport(results=[res_real])
        assert report_real.actual_provider_input_tokens == 100
        assert report_real.actual_provider_output_tokens == 50
        assert report_real.estimated_provider_cost == 0.0015

    def test_sub_100ms_timing_precision(self):
        """Timing must report in milliseconds with sub-millisecond precision, not rounded seconds."""
        case = BenchmarkCase(name="fast_case", query="fast query")
        # 14.2 ms execution
        res = BenchmarkResult(case=case, success=True, elapsed_seconds=0.0142)
        report = BenchmarkReport(results=[res])

        assert pytest.approx(report.latency_ms_mean, rel=1e-3) == 14.2
        assert pytest.approx(report.latency_ms_p50, rel=1e-3) == 14.2
        assert pytest.approx(report.latency_ms_p95, rel=1e-3) == 14.2

    def test_policy_prediction_vs_actual_execution_semantics(self):
        """In shadow mode, report metrics must evaluate actual controlling execution."""
        case_sys2 = BenchmarkCase(
            name="c_sys2",
            query="multi-agent complex pipeline",
            expected_execution_class="system2",
        )
        # Controlling policy incorrectly sent it to direct (false bypass)
        # while Laya shadow correctly predicted system2
        res = BenchmarkResult(
            case=case_sys2,
            success=True,
            elapsed_seconds=0.01,
            execution_class="direct",
            execution_path="direct",
            system2_invoked=False,
            laya_shadow_prediction={"execution_class": "system2", "confidence": 0.90},
        )
        report = BenchmarkReport(results=[res])

        # Actual execution was direct for a system2 query -> False bypass!
        assert report.false_bypass_count == 1
        assert report.false_bypass_rate == 100.0
        # Classification accuracy of controlling policy is 0%
        assert report.execution_class_accuracy == 0.0

    def test_expert_routing_accuracy_requires_gold_labels(self):
        """Cases without gold expert labels must not artificially inflate expert accuracy."""
        # Case with NO expected expert (e.g. direct response)
        case_no_gold = BenchmarkCase(
            name="direct_1",
            query="hello",
            expected_execution_class="direct",
            expected_primary_expert=None,
            expected_experts=[],
        )
        res_no_gold = BenchmarkResult(
            case=case_no_gold,
            success=True,
            elapsed_seconds=0.005,
            execution_class="direct",
            experts_used=[],
        )
        report_empty = BenchmarkReport(results=[res_no_gold])
        assert report_empty.expert_routing_accuracy == 0.0

        # Case WITH gold label
        case_gold = BenchmarkCase(
            name="tech_1",
            query="code",
            expected_execution_class="single_expert",
            expected_primary_expert="technical",
        )
        res_gold_correct = BenchmarkResult(
            case=case_gold,
            success=True,
            elapsed_seconds=0.01,
            execution_class="single_expert",
            primary_expert="technical",
            experts_used=["technical"],
        )
        report_with_gold = BenchmarkReport(results=[res_no_gold, res_gold_correct])
        # Only 1 case had gold label, and it was correct -> 100.0%
        assert report_with_gold.expert_routing_accuracy == 100.0

    def test_false_bypass_and_unnecessary_escalation_definitions(self):
        """Verify false bypass and unnecessary escalation definitions."""
        c1 = BenchmarkCase(
            name="c1", query="q1", expected_execution_class="system2"
        )
        c2 = BenchmarkCase(
            name="c2", query="q2", expected_execution_class="direct"
        )
        c3 = BenchmarkCase(
            name="c3", query="q3", expected_execution_class="single_expert", expected_primary_expert="creative"
        )

        r1 = BenchmarkResult(
            case=c1, success=True, elapsed_seconds=0.01,
            execution_class="single_expert",  # System-2 falsely bypassed to single_expert!
        )
        r2 = BenchmarkResult(
            case=c2, success=True, elapsed_seconds=0.02,
            execution_class="system2",  # Direct unnecessarily escalated to system2!
            system2_invoked=True,
        )
        r3 = BenchmarkResult(
            case=c3, success=True, elapsed_seconds=0.015,
            execution_class="single_expert",  # Correct!
            primary_expert="creative",
        )

        report = BenchmarkReport(results=[r1, r2, r3])

        # Expected system2 cases: c1 (1 total). r1 was bypassed -> 1 / 1 = 100%
        assert report.false_bypass_count == 1
        assert report.false_bypass_rate == 100.0

        # Expected non-system2 cases: c2, c3 (2 total). r2 was escalated -> 1 / 2 = 50%
        assert report.unnecessary_escalation_count == 1
        assert report.unnecessary_escalation_rate == 50.0

    def test_brier_score_and_calibration_error(self):
        """Test Brier score and Expected Calibration Error computation."""
        case = BenchmarkCase(
            name="c_brier",
            query="q",
            expected_execution_class="single_expert",
        )
        # Perfect calibrated prediction: P(single_expert)=1.0
        res_perfect = BenchmarkResult(
            case=case,
            success=True,
            elapsed_seconds=0.01,
            execution_class="single_expert",
            laya_metadata={
                "execution_class": "single_expert",
                "confidence": 1.0,
                "probabilities": {
                    "execution_class": {
                        "direct": 0.0,
                        "single_expert": 1.0,
                        "system2": 0.0,
                    }
                },
            },
        )
        report = BenchmarkReport(results=[res_perfect])
        # Multi-class Brier score should be 0.0 (perfect)
        assert report.brier_score == 0.0
        # Expected calibration error should be 0.0 (perfect)
        assert report.expected_calibration_error == 0.0
