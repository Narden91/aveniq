import asyncio
from pathlib import Path

import pytest

from research.calibrate_policy import _fit_temperature, _rule_metrics, _soften
from research.counterfactual import (
    Candidate,
    RouteOutcome,
    SpecializationExample,
    derive_example,
    failure_category,
    load_candidates,
    run,
)
from research.downstream_eval import escalation_reason, summarize


def candidate():
    return Candidate(
        id="case",
        query="Say hello",
        split="training",
        primary_expert="general",
        task_check_regex="hello",
        provenance={"origin": "test", "task_check": "test"},
    )


def outcome(route, passed, cost):
    return RouteOutcome(
        route=route,
        task_success=passed,
        verification_score=float(passed),
        latency_ms=1,
        input_tokens=1,
        output_tokens=1,
        estimated_cost_usd=cost,
        failure_category=None if passed else "task_check_failure",
        answer="hello" if passed else "",
        error=None,
    )


def test_label_comes_from_cheapest_passing_route():
    example = derive_example(
        candidate(),
        [
            outcome("direct", False, 0),
            outcome("single_expert", True, 0.02),
            outcome("system2", True, 0.03),
        ],
    )
    assert example.execution_class == "single_expert"
    assert example.primary_expert == "general"
    assert example.cost_measurements["selected_estimated_cost_usd"] == 0.02
    assert (
        derive_example(
            candidate(),
            [
                outcome("direct", False, 0),
                outcome("single_expert", False, 0),
                outcome("system2", False, None),
            ],
        )
        is None
    )
    tampered = example.model_dump()
    tampered["execution_class"] = "system2"
    tampered["primary_expert"] = None
    with pytest.raises(ValueError, match="cheapest"):
        SpecializationExample.model_validate(tampered)


def test_provider_interruption_never_supplies_a_label():
    outcomes = [
        outcome("direct", True, 0.01),
        outcome("single_expert", False, None).model_copy(
            update={"failure_category": "provider_token_limit", "error": "quota exhausted"}
        ),
        outcome("system2", True, 0.03),
    ]
    assert derive_example(candidate(), outcomes) is None
    good = derive_example(candidate(), [
        outcome("direct", False, 0), outcome("single_expert", True, 0.02),
        outcome("system2", True, 0.03),
    ])
    tampered = good.model_dump()
    tampered["candidate_route_outcomes"] = [row.model_dump() for row in outcomes]
    with pytest.raises(ValueError, match="Incomplete provider run"):
        SpecializationExample.model_validate(tampered)


def test_failure_categories_keep_provider_limits_separate():
    assert failure_category("tokens per minute exceeded", False) == "provider_token_limit"
    assert failure_category("request timed out", False) == "provider_timeout"
    assert failure_category("invalid response", False) == "runtime_failure"
    assert failure_category(None, False) == "task_check_failure"
    assert failure_category(None, False, True) == "routing_failure"


def test_original_test_query_cannot_enter_training(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        candidate().model_copy(update={"query": "Goodbye."}).model_dump_json() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        load_candidates(path)


def test_archived_outcomes_cannot_supply_canonical_examples():
    with pytest.raises(ValueError, match="Archived"):
        asyncio.run(run(
            Path("research/v03_candidates.jsonl"),
            Path("research/archive/v03_incomplete/aborted.jsonl"),
            Path("research/v03_examples.jsonl"),
        ))


def test_calibration_uses_only_passed_rows():
    rows = [
        {
            "expected_execution_class": "direct",
            "prediction": {
                "probabilities": {
                    "execution_class": {"direct": 0.8, "single_expert": 0.1, "system2": 0.1}
                }
            },
        }
    ]
    assert _fit_temperature(rows, "execution_class", "expected_execution_class") == 0.5
    probabilities = _soften({"direct": 0.8, "single_expert": 0.1, "system2": 0.1}, 2.0)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-12


def test_escalation_has_one_reason_and_provider_failures_stay_separate():
    assert (
        escalation_reason("fine_tuned_laya", "system2", {"execution_class": "system2"})
        == "predicted_system2"
    )
    assert (
        escalation_reason(
            "fine_tuned_laya",
            "system2",
            {"execution_class": "direct", "fallback_reason": "low_confidence"},
        )
        == "low_confidence"
    )
    assert (
        escalation_reason(
            "fine_tuned_laya", "system2", {"fallback_reason": "inference_error: failed"}
        )
        == "inference_error"
    )
    assert escalation_reason("rule_policy", "system2", None) == "policy_override"
    row = {
        "task_success": False,
        "estimated_cost_usd": None,
        "input_tokens": None,
        "output_tokens": None,
        "latency_ms": 10,
        "expected_execution_class": "system2",
        "system2_invoked": True,
        "failure_category": "provider_token_limit",
        "escalation_reason": "low_confidence",
    }
    summary = summarize([row])
    assert summary["failure_categories"]["provider_token_limit"] == 1
    assert summary["failure_categories"]["routing_failure"] == 0
    assert summary["false_bypass"] == {"count": 0, "total": 1, "rate": 0.0}
    assert summary["unnecessary_escalation"] == {"count": 0, "total": 0, "rate": None}
    assert summary["system2_invocation"] == {"count": 1, "total": 1, "rate": 1.0}


def test_rule_parallel_route_is_counted_as_its_own_output():
    rows = [
        {
            "expected_execution_class": "system2",
            "expected_primary_expert": None,
            "prediction": {
                "execution_class": "parallel_experts",
                "primary_expert": None,
                "total_policy_latency_ms": 1.0,
            },
        }
    ]
    metrics = _rule_metrics(rows)
    assert metrics["confusion_matrix"]["system2"]["parallel_experts"] == 1
    assert metrics["false_bypass"]["count"] == 1
    assert metrics["system2_invocation_rate"]["count"] == 0
