"""Provider-backed held-out comparison with one fixed provider configuration."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.dataset import load_adaptive_routing_dataset
from research.calibrate_policy import _soften
from research.counterfactual import (
    failure_category,
    fixed_provider_model,
    load_candidates,
    provider_settings,
)
from research.downstream_baseline import CHECKS, RecordingLayaPolicy
from src.aveniq.policy import AlwaysSystem2Policy, RulePolicy
from src.core.config import MoEConfig
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder
from src.utils.metrics import reset_token_tracker


class CalibratedLayaPolicy(RecordingLayaPolicy):
    def __init__(self, checkpoint: str, temperatures: dict[str, float], threshold: float):
        super().__init__(
            mode="control", confidence_threshold=threshold, checkpoint=checkpoint, device="cuda"
        )
        self.temperatures = temperatures

    def predict_laya(self, query: str) -> dict[str, Any]:
        prediction = super().predict_laya(query)
        prediction["probabilities"] = {
            name: _soften(prediction["probabilities"][name], self.temperatures[name])
            for name in ("execution_class", "primary_expert")
        }
        execution = prediction["execution_class"]
        prediction["confidence"] = prediction["probabilities"]["execution_class"][execution]
        if execution == "single_expert":
            expert = prediction["primary_expert"]
            prediction["confidence"] = min(
                prediction["confidence"], prediction["probabilities"]["primary_expert"][expert]
            )
        return prediction


def escalation_reason(
    variant: str, path: str | None, metadata: dict[str, Any] | None
) -> str | None:
    if path != "system2":
        return None
    if variant != "fine_tuned_laya":
        return "policy_override"
    metadata = metadata or {}
    if str(metadata.get("fallback_reason") or "").startswith("inference_error"):
        return "inference_error"
    if metadata.get("execution_class") == "system2":
        return "predicted_system2"
    if metadata.get("fallback_reason") == "low_confidence":
        return "low_confidence"
    return "policy_override"


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    successes = sum(row["task_success"] for row in rows)
    known_cost = sum(
        row["estimated_cost_usd"] for row in rows if row["estimated_cost_usd"] is not None
    )
    known_tokens = sum((row["input_tokens"] or 0) + (row["output_tokens"] or 0) for row in rows)
    latencies = [row["latency_ms"] for row in rows]
    categories = {
        name: sum(row["failure_category"] == name for row in rows)
        for name in (
            "routing_failure",
            "runtime_failure",
            "provider_token_limit",
            "provider_timeout",
            "task_check_failure",
        )
    }
    causes = {
        name: sum(row["escalation_reason"] == name for row in rows)
        for name in ("predicted_system2", "low_confidence", "inference_error", "policy_override")
    }
    system2_total = sum(row["expected_execution_class"] == "system2" for row in rows)
    simple_total = count - system2_total
    invoked = sum(row["system2_invoked"] for row in rows)
    false_bypass = sum(
        row["expected_execution_class"] == "system2" and not row["system2_invoked"]
        for row in rows
    )
    unnecessary = sum(
        row["expected_execution_class"] != "system2" and row["system2_invoked"]
        for row in rows
    )
    return {
        "task_success": {
            "count": successes,
            "total": count,
            "rate": successes / count if count else None,
        },
        "system2_invocation": {
            "count": invoked,
            "total": count,
            "rate": invoked / count if count else None,
        },
        "cost_per_successful_task_usd_known_usage": known_cost / successes if successes else None,
        "tokens_per_successful_task_known_usage": known_tokens / successes if successes else None,
        "missing_cost_records": sum(row["estimated_cost_usd"] is None for row in rows),
        "missing_token_records": sum(
            row["input_tokens"] is None or row["output_tokens"] is None for row in rows
        ),
        "latency_ms_p50": _percentile(latencies, 0.5) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95) if latencies else None,
        "false_bypass": {
            "count": false_bypass,
            "total": system2_total,
            "rate": false_bypass / system2_total if system2_total else None,
        },
        "unnecessary_escalation": {
            "count": unnecessary,
            "total": simple_total,
            "rate": unnecessary / simple_total if simple_total else None,
        },
        "failure_categories": categories,
        "escalation_causes": causes,
        "unnecessary_escalation_by_reason": {
            name: sum(
                row["escalation_reason"] == name and row["expected_execution_class"] != "system2"
                for row in rows
            )
            for name in causes
        },
    }


async def run(
    routing_path: Path, output: Path, summary_path: Path, timeout_seconds: int = 180
) -> dict[str, Any]:
    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    checkpoint = next(
        (
            name for name in routing["models"]
            if name not in (
                "RulePolicy", "convaiinnovations/laya", "convaiinnovations/laya-typed-decisions"
            )
        ),
        (
            "convaiinnovations/laya-typed-decisions"
            if "convaiinnovations/laya-typed-decisions" in routing["models"] else None
        ),
    )
    if checkpoint is None:
        raise ValueError("Routing results contain no Laya checkpoint")
    fitted = routing["models"][checkpoint]
    threshold = fitted["threshold_selection"]
    if threshold["selection_split"] != "validation":
        raise ValueError("Threshold must be selected on validation")
    if "final_test" in routing["split_ids"]:
        candidate_path = Path("research/data/v04/candidates.jsonl")
        candidates = {candidate.id: candidate for candidate in load_candidates(candidate_path)}
        final_ids = routing["split_ids"]["final_test"]
        predictions = {
            row["id"]: row for row in fitted["predictions"] if row["split"] == "final_test"
        }
        if set(final_ids) != set(predictions) or any(
            candidates[case_id].split != "final_test"
            or candidates[case_id].query != predictions[case_id]["query"]
            for case_id in final_ids
        ):
            raise ValueError("Routing final-test cases differ from frozen candidates")
        checks = {case_id: candidates[case_id].task_check_regex for case_id in final_ids}
        records = {
            case_id: SimpleNamespace(
                query=predictions[case_id]["query"],
                expected_execution_class=predictions[case_id]["expected_execution_class"],
            )
            for case_id in final_ids
        }
    else:
        records = {record.id: record for record in load_adaptive_routing_dataset()}
        checks = CHECKS
        if any(records[case_id].split != "test" for case_id in checks):
            raise ValueError("Downstream task checks include a non-test case")
    config = MoEConfig()
    settings = provider_settings(config, timeout_seconds)
    if "final_test" in routing["split_ids"]:
        frozen_config = json.loads(
            Path("research/data/v04/config.json").read_text(encoding="utf-8")
        )
        if settings != frozen_config["provider_settings"]:
            raise ValueError("Provider settings differ from frozen v0.4 configuration")
    policies = {
        "always_system2": AlwaysSystem2Policy(),
        "rule_policy": RulePolicy(),
        "fine_tuned_laya": CalibratedLayaPolicy(
            checkpoint, fitted["calibration_temperatures"], threshold["threshold"]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        {
            (row["variant"], row["id"])
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        }
        if output.exists()
        else set()
    )
    with fixed_provider_model():
        for variant, policy in policies.items():
            variant_cfg = deepcopy(config)
            variant_cfg.registry_db_path = str(output.with_name(f"{output.stem}_{variant}.sqlite"))
            graph = MoEGraphBuilder(
                variant_cfg,
                policy_engine=policy,
                trace_file=str(output.with_name(f"{output.stem}_{variant}.traces.jsonl")),
            ).build()
            for case_id, pattern in checks.items():
                if (variant, case_id) in completed:
                    continue
                record = records[case_id]
                if isinstance(policy, CalibratedLayaPolicy):
                    policy.last_record = None
                reset_token_tracker()
                start = time.perf_counter()
                try:
                    state = await asyncio.wait_for(
                        graph.ainvoke(create_initial_state(record.query)), timeout_seconds
                    )
                    answer = str(state.get("final_answer") or "")
                    error = str(state.get("code_execution_error") or "") or None
                    metadata = state.get("metadata") or {}
                    usage = state.get("token_usage") or {}
                    path = metadata.get("execution_path")
                    laya_metadata = metadata.get("laya_metadata") or getattr(
                        policy, "last_record", None
                    )
                    passed = (
                        not error
                        and path is not None
                        and bool(re.search(pattern, answer, re.IGNORECASE | re.DOTALL))
                    )
                    invoked = (
                        bool(metadata.get("system2_invoked"))
                        or path == "system2"
                        or str(error or "").startswith("Orchestrator:")
                    )
                    if invoked and path is None:
                        path = "system2"
                except Exception as exc:
                    answer, usage, path, invoked, passed = "", {}, None, False, False
                    laya_metadata = getattr(policy, "last_record", None)
                    error = f"{type(exc).__name__}: {exc}"
                if path is None:
                    if variant == "always_system2":
                        path = "system2"
                    elif variant == "rule_policy":
                        path = RulePolicy().evaluate(record.query).execution_class
                    elif laya_metadata:
                        path = laya_metadata.get("execution_class")
                        if laya_metadata.get("fallback_reason") == "low_confidence":
                            path = "system2"
                        elif str(laya_metadata.get("fallback_reason") or "").startswith(
                            "inference_error"
                        ):
                            path = RulePolicy().evaluate(record.query).execution_class
                    invoked = path == "system2"
                reason = escalation_reason(variant, path, laya_metadata) if invoked else None
                result = {
                    "variant": variant,
                    "id": case_id,
                    "query": record.query,
                    "checkpoint": checkpoint if variant == "fine_tuned_laya" else variant,
                    "provider_settings": settings,
                    "expected_execution_class": record.expected_execution_class,
                    "task_check_regex": pattern,
                    "task_success": passed,
                    "answer": answer,
                    "error": error,
                    "execution_path": path,
                    "system2_invoked": invoked,
                    "escalation_reason": reason,
                    "failure_category": failure_category(error, passed, not error and path is None),
                    "input_tokens": usage.get("total_input_tokens"),
                    "output_tokens": usage.get("total_output_tokens"),
                    "estimated_cost_usd": usage.get("estimated_cost_usd"),
                    "latency_ms": (time.perf_counter() - start) * 1000,
                    "laya_metadata": laya_metadata,
                }
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n")
                print(variant, case_id, passed, result["failure_category"], flush=True)
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    summary = {
        "provider_settings": settings,
        "threshold": threshold,
        "subset_ids": list(checks),
        "variants": {
            variant: summarize([row for row in rows if row["variant"] == variant])
            for variant in policies
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing", type=Path, default=Path("research/v03_routing_results.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    prefix = "v03_" if args.routing.name == "v03_routing_results.json" else ""
    output = args.output or args.routing.with_name(f"{prefix}downstream_results.jsonl")
    summary = args.summary or args.routing.with_name(f"{prefix}downstream_summary.json")
    print(
        json.dumps(
            asyncio.run(run(args.routing, output, summary, args.timeout)), indent=2
        )
    )


if __name__ == "__main__":
    main()
