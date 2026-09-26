"""Run a fixed held-out subset through three real provider-backed policies."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.dataset import load_adaptive_routing_dataset
from src.aveniq.policy import AlwaysSystem2Policy, LayaPolicy, RulePolicy
from src.core.config import MoEConfig
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder
from src.utils.metrics import reset_token_tracker

CONFIG = json.loads(Path(__file__).with_name("v021_config.json").read_text(encoding="utf-8"))

# Case IDs were fixed before execution. Revision 2 accepts Unicode spacing
# and valid acknowledgments.
CHECKS = {
    "route_008": r"\b(bye|goodbye|farewell|see you)\b",
    "route_012": r"\b(yes|available|here|ready|assist)\b",
    "route_013": r"\b(yes|hear|read|message|here|ready)\b",
    "route_018": r"^\s*ok[.!]?\s*$",
    "route_026": r"\b(yes|working|here|ready)\b",
    "route_027": r"\b(received|acknowledged|got it|message)\b",
    "route_030": r"\b(welcome|glad|happy)\b",
    "route_031": r"\b(welcome|glad|happy|thank)\b",
    "route_034": r"\b(bonjour|hello|yes|here)\b",
    "route_039": r"\b(bye|goodbye|here|okay|welcome|got it)\b",
    "route_043": r"\b(query|queries|lookup|search|retriev|select)\w*",
    "route_060": r"\b(shared|persist|reus|mutat|same)\w*",
    "route_061": r"\b(sorted|sort)\b.*\b(key|lambda)\b",
    "route_063": r"\b(lifo|last.in.first.out)\b.*\b(fifo|first.in.first.out)\b",
    "route_064": r"\bemail\s*\?\s*:",
    "route_065": r"\btry\s*:.*\bexcept\b",
    "route_070": (
        r"(?:95\s*%|95 percent|ninety.five percent).*"
        r"\b(interval|repeat|cover|procedure|sample)\w*"
    ),
    "route_071": r"\bprecision\b.*\brecall\b|\brecall\b.*\bprecision\b",
    "route_074": r"\b(true positive|false positive|tp|fp)\b",
    "route_078": r"\b12\b",
    "route_079": r"\b(bias|nonrandom|not random|systematic)\w*",
    "route_084": r"\babsolute\b.*\brelative\b|\brelative\b.*\babsolute\b",
    "route_086": r"\brate\b.*\bcount\b|\bcount\b.*\brate\b",
    "route_088": (
        r"\b(randomized|randomised)\b.*\bobservational\b|"
        r"\bobservational\b.*\b(randomized|randomised)\b"
    ),
    "route_097": r"\b(spring|blossom|flower|bloom)\w*",
    "route_098": r"\b(tea|steep|sip)\w*",
    "route_127": r"\b(sun|moon|earth)\b.*\b(block|shadow|align|obscur)\w*",
    "route_128": r"\bportuguese\b",
    "route_132": r"\b(star|planet)\b.*\b(fusion|orbit|light)\w*",
    "route_140": r"\b(book|information|knowledge|read|resource)\w*",
}


class RecordingLayaPolicy(LayaPolicy):
    """Keep the decision when a later graph node returns an error state."""

    last_record: dict[str, Any] | None = None

    def evaluate(self, query: str, context: dict[str, Any] | None = None):
        decision = super().evaluate(query, context)
        self.last_record = decision.metadata.get("laya_metadata")
        return decision


async def run(
    output: Path,
    summary_path: Path,
    routing_path: Path,
    retry_missing_laya_metadata: bool = False,
) -> dict[str, Any]:
    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    threshold_selection = routing["threshold_selection"]
    if threshold_selection["selection_split"] != "validation":
        raise ValueError("Laya threshold was not selected on validation")
    records = {record.id: record for record in load_adaptive_routing_dataset()}
    if len(CHECKS) < 30 or any(records[case_id].split != "test" for case_id in CHECKS):
        raise ValueError("Downstream subset must contain at least 30 held-out cases")
    cfg = MoEConfig()
    provider = cfg.get_provider_type()
    model = cfg.orchestrator_config.model_name
    if provider != CONFIG["provider"] or model != CONFIG["provider_model"]:
        raise ValueError("Provider and model differ from research/v021_config.json")
    policies = {
        "always_system2": AlwaysSystem2Policy(),
        "rule_policy": RulePolicy(),
        "laya_control": RecordingLayaPolicy(
            mode="control", confidence_threshold=threshold_selection["threshold"], device="cuda"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line:
                item = json.loads(line)
                if not (
                    retry_missing_laya_metadata
                    and item["variant"] == "laya_control"
                    and item["laya_metadata"] is None
                ):
                    completed.add((item["variant"], item["id"]))
    for variant, policy in policies.items():
        pending = [
            (case_id, pattern)
            for case_id, pattern in CHECKS.items()
            if (variant, case_id) not in completed
        ]
        if not pending:
            continue
        variant_cfg = deepcopy(cfg)
        variant_cfg.registry_db_path = str(output.with_name(f"downstream_{variant}.sqlite"))
        builder = MoEGraphBuilder(
            variant_cfg,
            policy_engine=policy,
            trace_file=str(output.with_name(f"downstream_{variant}.traces.jsonl")),
        )
        graph = builder.build()
        for case_id, pattern in pending:
            record = records[case_id]
            if isinstance(policy, RecordingLayaPolicy):
                policy.last_record = None
            reset_token_tracker()
            start = time.perf_counter()
            try:
                state = await asyncio.wait_for(
                    graph.ainvoke(create_initial_state(record.query)), timeout=180
                )
                answer = str(state.get("final_answer") or "")
                metadata = state.get("metadata") or {}
                usage = state.get("token_usage") or {}
                execution_success = not bool(state.get("code_execution_error"))
                task_success = (
                    bool(re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL))
                    if execution_success
                    else False
                )
                result = {
                    "variant": variant,
                    "id": case_id,
                    "query": record.query,
                    "provider_kind": "real",
                    "expected_execution_class": record.expected_execution_class,
                    "execution_success": execution_success,
                    "task_success": task_success,
                    "task_check_regex": pattern,
                    "task_check_revision": CONFIG["task_check_revision"],
                    "answer": answer,
                    "provider_input_tokens": usage.get("total_input_tokens"),
                    "provider_output_tokens": usage.get("total_output_tokens"),
                    "provider_cost_usd_list_price": usage.get("estimated_cost_usd"),
                    "system2_invoked": bool(metadata.get("system2_invoked"))
                    or str(state.get("code_execution_error") or "").startswith("Orchestrator:"),
                    "execution_path": metadata.get("execution_path"),
                    "laya_metadata": metadata.get("laya_metadata")
                    or getattr(policy, "last_record", None),
                    "error": state.get("code_execution_error") or None,
                }
            except Exception as exc:
                result = {
                    "variant": variant,
                    "id": case_id,
                    "query": record.query,
                    "provider_kind": "real",
                    "expected_execution_class": record.expected_execution_class,
                    "execution_success": False,
                    "task_success": False,
                    "task_check_regex": pattern,
                    "task_check_revision": CONFIG["task_check_revision"],
                    "answer": "",
                    "provider_input_tokens": None,
                    "provider_output_tokens": None,
                    "provider_cost_usd_list_price": None,
                    "system2_invoked": str(exc).startswith("Orchestrator:"),
                    "execution_path": None,
                    "laya_metadata": getattr(policy, "last_record", None),
                    "error": str(exc),
                }
            result["end_to_end_latency_ms"] = (time.perf_counter() - start) * 1000
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"{variant} {case_id} execution={result['execution_success']} "
                f"task={result['task_success']}",
                flush=True,
            )
    rows_by_key = {}
    all_lines = output.read_text(encoding="utf-8").splitlines()
    for line in all_lines:
        if line:
            row = json.loads(line)
            rows_by_key[(row["variant"], row["id"])] = row
    rows = list(rows_by_key.values())
    superseded_path = output.with_name("downstream_superseded_attempts.jsonl")
    superseded_count = (
        sum(bool(line) for line in superseded_path.read_text(encoding="utf-8").splitlines())
        if superseded_path.exists()
        else 0
    )
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "provider_kind": "real",
        "same_configured_provider_model_across_variants": True,
        "pricing_source": CONFIG["pricing_source"],
        "pricing_usd_per_million": CONFIG["pricing_usd_per_million"],
        "cost_note": (
            "List-price estimate from reported tokens; cached input discounts and billed "
            "amounts are not exposed by the tracker."
        ),
        "threshold": threshold_selection,
        "task_check_revision": CONFIG["task_check_revision"],
        "subset_ids": list(CHECKS),
        "request_attempts_recorded": len([line for line in all_lines if line]),
        "latest_request_records_used": len(rows),
        "superseded_request_attempts": superseded_count,
        "variants": {},
    }
    for variant in policies:
        subset = [row for row in rows if row["variant"] == variant and row["id"] in CHECKS]
        if len(subset) != len(CHECKS):
            raise RuntimeError(f"{variant} has only {len(subset)} results")
        summary["variants"][variant] = {
            "execution_success": {
                "count": sum(row["execution_success"] for row in subset),
                "total": len(subset),
            },
            "task_success": {
                "count": sum(row["task_success"] for row in subset),
                "total": len(subset),
            },
            "provider_input_tokens": sum(row["provider_input_tokens"] or 0 for row in subset),
            "provider_output_tokens": sum(row["provider_output_tokens"] or 0 for row in subset),
            "provider_cost_usd_list_price": sum(
                row["provider_cost_usd_list_price"] or 0 for row in subset
            ),
            "end_to_end_latency_ms_mean": statistics.mean(
                row["end_to_end_latency_ms"] for row in subset
            ),
            "system2_invocation": {
                "count": sum(row["system2_invoked"] is True for row in subset),
                "total": len(subset),
            },
        }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("research/downstream_results.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("research/downstream_summary.json"))
    parser.add_argument("--routing", type=Path, default=Path("research/routing_results.json"))
    parser.add_argument("--retry-missing-laya-metadata", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                run(args.output, args.summary, args.routing, args.retry_missing_laya_metadata)
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
