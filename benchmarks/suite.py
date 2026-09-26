"""Benchmark harness for the MoE pipeline.

Runs a set of standardised queries, records timing / token usage, and
produces a structured report.  Designed to be executed from the CLI::

    python -m benchmarks.run            # run all benchmarks
    python -m benchmarks.run --filter routing   # substring filter

Or imported programmatically::

    from benchmarks.suite import BenchmarkSuite, BenchmarkCase
    suite = BenchmarkSuite()
    suite.add(BenchmarkCase(query="…", expected_experts=["technical"]))
    report = await suite.run_all(graph)
"""

from __future__ import annotations

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.metrics import get_token_tracker, reset_token_tracker


# ======================================================================
# Data model
# ======================================================================

@dataclass
class BenchmarkCase:
    """A single benchmark scenario."""

    name: str
    query: str
    expected_experts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    family: str = ""
    expected_execution_class: Optional[str] = None
    expected_primary_expert: Optional[str] = None
    deterministic_validation: Optional[Dict[str, Any]] = None

    def matches_filter(self, pattern: str) -> bool:
        pattern_lower = pattern.lower()
        return (
            pattern_lower in self.name.lower()
            or pattern_lower in self.query.lower()
            or pattern_lower in self.family.lower()
            or any(pattern_lower in t.lower() for t in self.tags)
        )


@dataclass
class BenchmarkResult:
    """Result of running a single :class:`BenchmarkCase`."""

    case: BenchmarkCase
    success: bool
    elapsed_seconds: float
    token_summary: Dict[str, Any] = field(default_factory=dict)
    experts_used: List[str] = field(default_factory=list)
    answer_snippet: str = ""
    error: str = ""
    repeat_index: int = 1
    retry_count: int = 0
    retrieval_metrics: Dict[str, Any] = field(default_factory=dict)
    neighborhood_reuse_rate: float = 0.0
    system2_invoked: bool = False
    laya_confidence: Optional[float] = None
    laya_shadow_prediction: Dict[str, Any] = field(default_factory=dict)
    laya_metadata: Optional[Dict[str, Any]] = None
    laya_latency_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    execution_path: str = ""
    execution_class: str = ""
    primary_expert: Optional[str] = None
    is_mock_provider: bool = False
    actual_provider_input_tokens: Optional[int] = None
    actual_provider_output_tokens: Optional[int] = None
    controlling_policy: str = ""


@dataclass
class BenchmarkReport:
    """Aggregated report across all benchmark cases."""

    results: List[BenchmarkResult] = field(default_factory=list)
    total_elapsed: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    # ---- Derived stats -----------------------------------------------

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def task_success_rate_pct(self) -> float:
        if not self.results:
            return 0.0
        return (self.passed / len(self.results)) * 100.0

    @property
    def success_rate_pct(self) -> float:
        return self.task_success_rate_pct

    @property
    def execution_class_accuracy(self) -> float:
        """% of cases where actual execution class matched gold expectation."""
        labeled = [r for r in self.results if r.case.expected_execution_class is not None]
        if not labeled:
            return 0.0
        correct = sum(
            1 for r in labeled
            if (r.execution_class or ("system2" if r.system2_invoked else r.execution_path)) == r.case.expected_execution_class
        )
        return (correct / len(labeled)) * 100.0

    @property
    def expert_routing_accuracy(self) -> float:
        """% of cases with gold expert labels where actual expert matched expectation."""
        labeled_primary = [r for r in self.results if r.case.expected_primary_expert is not None]
        if labeled_primary:
            correct = 0
            for r in labeled_primary:
                expected = r.case.expected_primary_expert
                actual = r.primary_expert or (r.experts_used[0] if r.experts_used else None)
                if actual == expected or (r.experts_used and expected in r.experts_used):
                    correct += 1
            return (correct / len(labeled_primary)) * 100.0

        labeled_experts = [r for r in self.results if r.case.expected_experts]
        if not labeled_experts:
            return 0.0
        correct = sum(
            1 for r in labeled_experts
            if set(r.case.expected_experts) <= set(r.experts_used)
        )
        return (correct / len(labeled_experts)) * 100.0

    @property
    def expert_accuracy(self) -> float:
        """Legacy alias: % of cases where actual experts matched expectations."""
        return self.expert_routing_accuracy

    @property
    def false_bypass_count(self) -> int:
        """Number of requests whose expected class is system2 but controlling policy sent to direct or single_expert."""
        sys2_cases = [r for r in self.results if r.case.expected_execution_class == "system2"]
        return sum(
            1 for r in sys2_cases
            if (r.execution_class or ("system2" if r.system2_invoked else r.execution_path)) in {"direct", "single_expert"}
        )

    @property
    def false_bypass_rate(self) -> float:
        """% of expected system2 requests that were falsely bypassed."""
        sys2_cases = [r for r in self.results if r.case.expected_execution_class == "system2"]
        if not sys2_cases:
            return 0.0
        return (self.false_bypass_count / len(sys2_cases)) * 100.0

    @property
    def unnecessary_escalation_count(self) -> int:
        """Number of requests whose expected class is direct or single_expert but policy sent to system2."""
        non_sys2_cases = [
            r for r in self.results
            if r.case.expected_execution_class in {"direct", "single_expert"}
        ]
        return sum(
            1 for r in non_sys2_cases
            if (r.execution_class or ("system2" if r.system2_invoked else r.execution_path)) == "system2"
        )

    @property
    def unnecessary_escalation_rate(self) -> float:
        """% of simple/direct requests that were unnecessarily escalated to System-2."""
        non_sys2_cases = [
            r for r in self.results
            if r.case.expected_execution_class in {"direct", "single_expert"}
        ]
        if not non_sys2_cases:
            return 0.0
        return (self.unnecessary_escalation_count / len(non_sys2_cases)) * 100.0

    @property
    def system2_invocation_rate_pct(self) -> float:
        if not self.results:
            return 0.0
        return sum(
            1 for r in self.results
            if r.system2_invoked or r.execution_class == "system2" or r.execution_path == "system2"
        ) / len(self.results) * 100.0

    @property
    def latency_ms_mean(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.elapsed_seconds * 1000.0 for r in self.results)

    @property
    def latency_ms_p50(self) -> float:
        if not self.results:
            return 0.0
        times = sorted(r.elapsed_seconds * 1000.0 for r in self.results)
        n = len(times)
        mid = n // 2
        return times[mid] if n % 2 == 1 else (times[mid - 1] + times[mid]) / 2.0

    @property
    def latency_ms_p95(self) -> float:
        if not self.results:
            return 0.0
        times = sorted(r.elapsed_seconds * 1000.0 for r in self.results)
        idx = max(0, min(int(len(times) * 0.95), len(times) - 1))
        return times[idx]

    @property
    def laya_latency_ms_mean(self) -> Optional[float]:
        laya_times = [
            r.laya_latency_ms for r in self.results
            if r.laya_latency_ms is not None
        ]
        return statistics.mean(laya_times) if laya_times else None

    @property
    def actual_provider_input_tokens(self) -> Optional[int]:
        valid = [
            r.actual_provider_input_tokens for r in self.results
            if not r.is_mock_provider and r.actual_provider_input_tokens is not None
        ]
        return sum(valid) if valid else None

    @property
    def actual_provider_output_tokens(self) -> Optional[int]:
        valid = [
            r.actual_provider_output_tokens for r in self.results
            if not r.is_mock_provider and r.actual_provider_output_tokens is not None
        ]
        return sum(valid) if valid else None

    @property
    def estimated_provider_cost(self) -> Optional[float]:
        valid = [
            r.estimated_cost_usd for r in self.results
            if not r.is_mock_provider and r.estimated_cost_usd is not None
        ]
        return sum(valid) if valid else None

    @property
    def brier_score(self) -> Optional[float]:
        """Compute multi-class Brier score over execution class probability distributions."""
        scores = []
        classes = ["direct", "single_expert", "system2"]
        for r in self.results:
            if not r.case.expected_execution_class:
                continue
            meta = r.laya_metadata or r.laya_shadow_prediction
            if not meta or not isinstance(meta, dict):
                continue
            probs = meta.get("probabilities")
            if not isinstance(probs, dict):
                continue
            class_probs = probs.get("execution_class")
            if not isinstance(class_probs, dict):
                continue

            item_score = 0.0
            for c in classes:
                p_c = float(class_probs.get(c, 0.0))
                y_c = 1.0 if r.case.expected_execution_class == c else 0.0
                item_score += (p_c - y_c) ** 2
            scores.append(item_score)

        return round(statistics.mean(scores), 4) if scores else None

    @property
    def expected_calibration_error(self) -> Optional[float]:
        """Compute Expected Calibration Error (ECE) with M=10 equal-width bins."""
        samples = []
        for r in self.results:
            if not r.case.expected_execution_class:
                continue
            meta = r.laya_metadata or r.laya_shadow_prediction
            if not meta or not isinstance(meta, dict):
                continue
            pred_class = meta.get("execution_class")
            conf = meta.get("confidence")
            if pred_class is None or conf is None:
                continue
            correct = 1.0 if pred_class == r.case.expected_execution_class else 0.0
            samples.append((float(conf), correct))

        if not samples:
            return None

        n = len(samples)
        num_bins = 10
        ece = 0.0
        for b in range(num_bins):
            low = b / num_bins
            high = (b + 1) / num_bins
            bin_items = [
                (conf, corr) for (conf, corr) in samples
                if (low <= conf < high if b < num_bins - 1 else low <= conf <= high)
            ]
            if bin_items:
                bin_size = len(bin_items)
                acc = sum(c for _, c in bin_items) / bin_size
                avg_conf = sum(cf for cf, _ in bin_items) / bin_size
                ece += (bin_size / n) * abs(acc - avg_conf)

        return round(ece, 4)

    # Legacy properties for backward compatibility
    @property
    def mean_tokens(self) -> float:
        tokens = [
            int(r.token_summary.get("total_tokens", 0))
            for r in self.results
            if r.token_summary
        ]
        return statistics.mean(tokens) if tokens else 0.0

    @property
    def mean_retries(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.retry_count for r in self.results)

    @property
    def mean_elapsed_seconds(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.elapsed_seconds for r in self.results)

    @property
    def latency_p50_seconds(self) -> float:
        if not self.results:
            return 0.0
        return self.latency_ms_p50 / 1000.0

    @property
    def latency_p95_seconds(self) -> float:
        if not self.results:
            return 0.0
        return self.latency_ms_p95 / 1000.0

    @property
    def mean_cost_usd(self) -> float:
        costs = [
            float(r.token_summary.get("estimated_cost_usd", r.estimated_cost_usd or 0.0))
            for r in self.results
        ]
        return statistics.mean(costs) if costs else 0.0

    @property
    def mean_laya_confidence(self) -> float:
        confs = [r.laya_confidence for r in self.results if r.laya_confidence is not None and r.laya_confidence > 0.0]
        return statistics.mean(confs) if confs else 0.0

    @property
    def mean_neighborhood_reuse_rate(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.neighborhood_reuse_rate for r in self.results)

    @property
    def recovery_rate_pct(self) -> float:
        retried = [r for r in self.results if r.retry_count > 0]
        if not retried:
            return 100.0 if self.passed > 0 else 0.0
        recovered = sum(1 for r in retried if r.success)
        return (recovered / len(retried)) * 100.0

    def summary(self) -> Dict[str, Any]:
        return {
            "total_cases": len(self.results),
            "passed": self.passed,
            "failed": self.failed,
            "task_success_rate_pct": round(self.task_success_rate_pct, 1),
            "success_rate_pct": round(self.success_rate_pct, 1),
            "execution_class_accuracy_pct": round(self.execution_class_accuracy, 1),
            "expert_routing_accuracy_pct": round(self.expert_routing_accuracy, 1),
            "expert_accuracy_pct": round(self.expert_accuracy, 1),
            "false_bypass_rate_pct": round(self.false_bypass_rate, 1),
            "false_bypass_count": self.false_bypass_count,
            "unnecessary_escalation_rate_pct": round(self.unnecessary_escalation_rate, 1),
            "unnecessary_escalation_count": self.unnecessary_escalation_count,
            "system2_invocation_rate_pct": round(self.system2_invocation_rate_pct, 1),
            "latency_ms_mean": round(self.latency_ms_mean, 2),
            "latency_ms_p50": round(self.latency_ms_p50, 2),
            "latency_ms_p95": round(self.latency_ms_p95, 2),
            "laya_latency_ms_mean": round(self.laya_latency_ms_mean, 2) if self.laya_latency_ms_mean is not None else None,
            "actual_provider_input_tokens": self.actual_provider_input_tokens,
            "actual_provider_output_tokens": self.actual_provider_output_tokens,
            "estimated_provider_cost_usd": round(self.estimated_provider_cost, 4) if self.estimated_provider_cost is not None else None,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            # Legacy fields for backward compatibility
            "elapsed_mean_seconds": round(self.mean_elapsed_seconds, 3),
            "latency_p50_seconds": round(self.latency_p50_seconds, 3),
            "latency_p95_seconds": round(self.latency_p95_seconds, 3),
            "retries_mean": round(self.mean_retries, 3),
            "tokens_mean": round(self.mean_tokens, 1),
            "cost_mean_usd": round(self.mean_cost_usd, 5),
            "laya_confidence_mean": round(self.mean_laya_confidence, 3),
            "neighborhood_reuse_rate_mean": round(self.mean_neighborhood_reuse_rate, 3),
            "total_elapsed_seconds": round(self.total_elapsed, 2),
            "recovery_rate_pct": round(self.recovery_rate_pct, 1),
            "by_case": self.case_aggregates(),
            "by_family": self.family_aggregates(),
            "per_case": [
                {
                    "name": r.case.name,
                    "family": r.case.family,
                    "repeat_index": r.repeat_index,
                    "success": r.success,
                    "elapsed_ms": round(r.elapsed_seconds * 1000.0, 2),
                    "execution_class": r.execution_class,
                    "expected_execution_class": r.case.expected_execution_class,
                    "primary_expert": r.primary_expert,
                    "expected_primary_expert": r.case.expected_primary_expert,
                    "experts_used": r.experts_used,
                    "system2_invoked": r.system2_invoked,
                    "laya_latency_ms": r.laya_latency_ms,
                    "laya_confidence": r.laya_confidence,
                    "is_mock_provider": r.is_mock_provider,
                    "tokens": r.token_summary.get("total_tokens", 0),
                    "retry_count": r.retry_count,
                    "neighborhood_reuse_rate": round(r.neighborhood_reuse_rate, 3),
                    "error": r.error[:200] if r.error else "",
                }
                for r in self.results
            ],
        }

    def case_aggregates(self) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            grouped.setdefault(result.case.name, []).append(result)

        rows: List[Dict[str, Any]] = []
        for name, items in grouped.items():
            elapsed_ms = [r.elapsed_seconds * 1000.0 for r in items]
            tokens = [
                int(r.token_summary.get("total_tokens", 0))
                for r in items
                if r.token_summary
            ]
            pass_rate = sum(1 for r in items if r.success) / len(items) * 100
            rows.append({
                "name": name,
                "runs": len(items),
                "pass_rate_pct": round(pass_rate, 1),
                "elapsed_mean_ms": round(statistics.mean(elapsed_ms), 2),
                "elapsed_mean": round(statistics.mean(r.elapsed_seconds for r in items), 3),
                "elapsed_stdev": round(statistics.pstdev(r.elapsed_seconds for r in items), 3) if len(items) > 1 else 0.0,
                "retries_mean": round(statistics.mean(r.retry_count for r in items), 3),
                "tokens_mean": round(statistics.mean(tokens), 1) if tokens else 0.0,
                "neighborhood_reuse_rate_mean": round(statistics.mean(r.neighborhood_reuse_rate for r in items), 3),
            })

        rows.sort(key=lambda r: r["name"])
        return rows

    def family_aggregates(self) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            if result.case.family:
                grouped.setdefault(result.case.family, []).append(result)

        rows: List[Dict[str, Any]] = []
        for family, items in grouped.items():
            elapsed = [r.elapsed_seconds for r in items]
            pass_rate = sum(1 for r in items if r.success) / len(items) * 100
            rows.append({
                "family": family,
                "runs": len(items),
                "pass_rate_pct": round(pass_rate, 1),
                "elapsed_mean": round(statistics.mean(elapsed), 3),
                "retries_mean": round(statistics.mean(r.retry_count for r in items), 3),
                "tokens_mean": round(statistics.mean(int(r.token_summary.get("total_tokens", 0)) for r in items), 1),
                "neighborhood_reuse_rate_mean": round(statistics.mean(r.neighborhood_reuse_rate for r in items), 3),
            })

        rows.sort(key=lambda r: r["family"])
        return rows

    def pretty_print(self) -> str:
        lines = [
            "=" * 72,
            " AVENIQ BENCHMARK REPORT",
            "=" * 72,
            f"  Total cases           : {len(self.results)}",
            f"  Passed / Failed       : {self.passed} / {self.failed} (Success: {self.task_success_rate_pct:.1f}%)",
            f"  Execution-class acc.  : {self.execution_class_accuracy:.1f}%",
            f"  Expert-routing acc.   : {self.expert_routing_accuracy:.1f}%",
            f"  False bypass rate     : {self.false_bypass_rate:.1f}% ({self.false_bypass_count} cases)",
            f"  Unnecessary escalation: {self.unnecessary_escalation_rate:.1f}% ({self.unnecessary_escalation_count} cases)",
            f"  System-2 invocation   : {self.system2_invocation_rate_pct:.1f}%",
            f"  Latency (mean/p50/p95): {self.latency_ms_mean:.1f}ms / {self.latency_ms_p50:.1f}ms / {self.latency_ms_p95:.1f}ms",
            f"  Laya inference latency: {f'{self.laya_latency_ms_mean:.2f}ms' if self.laya_latency_ms_mean is not None else 'null (not run)'}",
            f"  Actual provider tokens: in={self.actual_provider_input_tokens if self.actual_provider_input_tokens is not None else 'null (mock/n/a)'}, out={self.actual_provider_output_tokens if self.actual_provider_output_tokens is not None else 'null (mock/n/a)'}",
            f"  Estimated provider cost: {f'${self.estimated_provider_cost:.4f}' if self.estimated_provider_cost is not None else 'null (mock/n/a)'}",
            f"  Brier score / ECE     : {self.brier_score if self.brier_score is not None else 'null'} / {self.expected_calibration_error if self.expected_calibration_error is not None else 'null'}",
            f"  Total elapsed time    : {self.total_elapsed:.2f}s",
            "-" * 72,
        ]
        for r in self.results:
            mark = "PASS" if r.success else "FAIL"
            cls_info = f"class={r.execution_class}" if r.execution_class else ""
            lines.append(
                f"  [{mark}] {r.case.name:26s}  {r.elapsed_seconds * 1000.0:6.1f}ms  "
                f"{cls_info:18s} sys2={str(r.system2_invoked):5s} experts={r.experts_used}"
            )
            if r.error:
                lines.append(f"         ERROR: {r.error[:120]}")

        if self.results and max(r.repeat_index for r in self.results) > 1:
            lines.append("-" * 72)
            lines.append(" AGGREGATES (BY CASE)")
            for row in self.case_aggregates():
                lines.append(
                    "  {name:30s} runs={runs:<2d} pass={pass_rate_pct:5.1f}% "
                    "elapsed={elapsed_mean:.2f}s±{elapsed_stdev:.2f} retries={retries_mean:.2f} tokens={tokens_mean:.1f} reuse={neighborhood_reuse_rate_mean:.2f}".format(**row)
                )

        family_rows = self.family_aggregates()
        if family_rows:
            lines.append("-" * 72)
            lines.append(" FAMILY AGGREGATES")
            for row in family_rows:
                lines.append(
                    "  {family:24s} runs={runs:<2d} pass={pass_rate_pct:5.1f}% elapsed={elapsed_mean:.2f}s retries={retries_mean:.2f} tokens={tokens_mean:.1f} reuse={neighborhood_reuse_rate_mean:.2f}".format(**row)
                )

        lines.append("=" * 72)
        return "\n".join(lines)


@dataclass
class BenchmarkVariantRun:
    name: str
    report: BenchmarkReport


@dataclass
class BenchmarkComparisonReport:
    variants: List[BenchmarkVariantRun] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        if not self.variants:
            return {"variants": [], "deltas": []}

        baseline = self.variants[0]
        baseline_extra = getattr(baseline.report, "extra_metrics", {}) or {}
        baseline_metrics = {
            "task_success_rate_pct": baseline.report.task_success_rate_pct,
            "success_rate_pct": baseline.report.success_rate_pct,
            "execution_class_accuracy_pct": baseline.report.execution_class_accuracy,
            "expert_routing_accuracy_pct": baseline.report.expert_routing_accuracy,
            "false_bypass_rate_pct": baseline.report.false_bypass_rate,
            "unnecessary_escalation_rate_pct": baseline.report.unnecessary_escalation_rate,
            "system2_invocation_rate_pct": baseline.report.system2_invocation_rate_pct,
            "latency_ms_mean": baseline.report.latency_ms_mean,
            "latency_ms_p50": baseline.report.latency_ms_p50,
            "latency_ms_p95": baseline.report.latency_ms_p95,
            "laya_latency_ms_mean": baseline.report.laya_latency_ms_mean,
            "actual_provider_input_tokens": baseline.report.actual_provider_input_tokens,
            "actual_provider_output_tokens": baseline.report.actual_provider_output_tokens,
            "estimated_provider_cost_usd": baseline.report.estimated_provider_cost,
            "brier_score": baseline.report.brier_score,
            "expected_calibration_error": baseline.report.expected_calibration_error,
            # Legacy metrics
            "elapsed_mean_seconds": baseline.report.mean_elapsed_seconds,
            "latency_p50_seconds": baseline.report.latency_p50_seconds,
            "latency_p95_seconds": baseline.report.latency_p95_seconds,
            "tokens_mean": baseline.report.mean_tokens,
            "cost_mean_usd": baseline.report.mean_cost_usd,
            "expert_accuracy_pct": baseline.report.expert_accuracy,
            "laya_confidence_mean": baseline.report.mean_laya_confidence,
            "retries_mean": baseline.report.mean_retries,
            "neighborhood_reuse_rate_mean": baseline.report.mean_neighborhood_reuse_rate,
            **baseline_extra,
        }

        variants = []
        deltas = []
        for variant in self.variants:
            extra = getattr(variant.report, "extra_metrics", {}) or {}
            metrics = {
                "task_success_rate_pct": round(variant.report.task_success_rate_pct, 1),
                "success_rate_pct": round(variant.report.success_rate_pct, 1),
                "execution_class_accuracy_pct": round(variant.report.execution_class_accuracy, 1),
                "expert_routing_accuracy_pct": round(variant.report.expert_routing_accuracy, 1),
                "false_bypass_rate_pct": round(variant.report.false_bypass_rate, 1),
                "false_bypass_count": variant.report.false_bypass_count,
                "unnecessary_escalation_rate_pct": round(variant.report.unnecessary_escalation_rate, 1),
                "unnecessary_escalation_count": variant.report.unnecessary_escalation_count,
                "system2_invocation_rate_pct": round(variant.report.system2_invocation_rate_pct, 1),
                "latency_ms_mean": round(variant.report.latency_ms_mean, 2),
                "latency_ms_p50": round(variant.report.latency_ms_p50, 2),
                "latency_ms_p95": round(variant.report.latency_ms_p95, 2),
                "laya_latency_ms_mean": round(variant.report.laya_latency_ms_mean, 2) if variant.report.laya_latency_ms_mean is not None else None,
                "actual_provider_input_tokens": variant.report.actual_provider_input_tokens,
                "actual_provider_output_tokens": variant.report.actual_provider_output_tokens,
                "estimated_provider_cost_usd": round(variant.report.estimated_provider_cost, 4) if variant.report.estimated_provider_cost is not None else None,
                "brier_score": variant.report.brier_score,
                "expected_calibration_error": variant.report.expected_calibration_error,
                # Legacy metrics
                "elapsed_mean_seconds": round(variant.report.mean_elapsed_seconds, 3),
                "latency_p50_seconds": round(variant.report.latency_p50_seconds, 3),
                "latency_p95_seconds": round(variant.report.latency_p95_seconds, 3),
                "tokens_mean": round(variant.report.mean_tokens, 1),
                "cost_mean_usd": round(variant.report.mean_cost_usd, 5),
                "expert_accuracy_pct": round(variant.report.expert_accuracy, 1),
                "laya_confidence_mean": round(variant.report.mean_laya_confidence, 3),
                "retries_mean": round(variant.report.mean_retries, 3),
                "neighborhood_reuse_rate_mean": round(variant.report.mean_neighborhood_reuse_rate, 3),
                "failed": variant.report.failed,
                **extra,
            }
            variants.append({"name": variant.name, "metrics": metrics})
            if variant is baseline:
                continue

            delta_dict = {
                "name": variant.name,
                "delta_task_success_rate_pct": round(variant.report.task_success_rate_pct - baseline_metrics["task_success_rate_pct"], 1),
                "delta_execution_class_accuracy_pct": round(variant.report.execution_class_accuracy - baseline_metrics["execution_class_accuracy_pct"], 1),
                "delta_expert_routing_accuracy_pct": round(variant.report.expert_routing_accuracy - baseline_metrics["expert_routing_accuracy_pct"], 1),
                "delta_false_bypass_rate_pct": round(variant.report.false_bypass_rate - baseline_metrics["false_bypass_rate_pct"], 1),
                "delta_unnecessary_escalation_rate_pct": round(variant.report.unnecessary_escalation_rate - baseline_metrics["unnecessary_escalation_rate_pct"], 1),
                "delta_system2_invocation_rate_pct": round(variant.report.system2_invocation_rate_pct - baseline_metrics["system2_invocation_rate_pct"], 1),
                "delta_latency_ms_mean": round(variant.report.latency_ms_mean - baseline_metrics["latency_ms_mean"], 2),
                "delta_latency_ms_p50": round(variant.report.latency_ms_p50 - baseline_metrics["latency_ms_p50"], 2),
                "delta_latency_ms_p95": round(variant.report.latency_ms_p95 - baseline_metrics["latency_ms_p95"], 2),
                # Legacy deltas
                "delta_success_rate_pct": round(variant.report.success_rate_pct - baseline_metrics["success_rate_pct"], 1),
                "delta_elapsed_mean_seconds": round(variant.report.mean_elapsed_seconds - baseline_metrics["elapsed_mean_seconds"], 3),
                "delta_latency_p50_seconds": round(variant.report.latency_p50_seconds - baseline_metrics["latency_p50_seconds"], 3),
                "delta_latency_p95_seconds": round(variant.report.latency_p95_seconds - baseline_metrics["latency_p95_seconds"], 3),
                "delta_tokens_mean": round(variant.report.mean_tokens - baseline_metrics["tokens_mean"], 1),
                "delta_cost_mean_usd": round(variant.report.mean_cost_usd - baseline_metrics["cost_mean_usd"], 5),
                "delta_expert_accuracy_pct": round(variant.report.expert_accuracy - baseline_metrics["expert_accuracy_pct"], 1),
                "delta_laya_confidence_mean": round(variant.report.mean_laya_confidence - baseline_metrics["laya_confidence_mean"], 3),
                "delta_retries_mean": round(variant.report.mean_retries - baseline_metrics["retries_mean"], 3),
                "delta_neighborhood_reuse_rate_mean": round(variant.report.mean_neighborhood_reuse_rate - baseline_metrics["neighborhood_reuse_rate_mean"], 3),
            }
            for k, v in extra.items():
                if k in baseline_metrics and baseline_metrics[k] is not None:
                    delta_dict[f"delta_{k}"] = round(float(v) - float(baseline_metrics[k]), 3)
            deltas.append(delta_dict)

        return {"variants": variants, "deltas": deltas}

    def pretty_print(self) -> str:
        lines = [
            "=" * 98,
            " AVENIQ BENCHMARK SLICE REPORT",
            "=" * 98,
        ]
        summary = self.summary()
        for item in summary["variants"]:
            m = item["metrics"]
            laya_lat_str = f"{m['laya_latency_ms_mean']:.1f}ms" if m.get("laya_latency_ms_mean") is not None else "null"
            ece_str = f"{m['expected_calibration_error']:.3f}" if m.get("expected_calibration_error") is not None else "null"
            brier_str = f"{m['brier_score']:.3f}" if m.get("brier_score") is not None else "null"
            lines.append(
                f"  {item['name']:24s} | success={m['task_success_rate_pct']:5.1f}% | "
                f"lat(mean/p50/p95)={m['latency_ms_mean']:5.1f}/{m['latency_ms_p50']:5.1f}/{m['latency_ms_p95']:5.1f}ms | "
                f"sys2_rate={m['system2_invocation_rate_pct']:5.1f}% | "
                f"class_acc={m['execution_class_accuracy_pct']:5.1f}% | "
                f"false_bypass={m['false_bypass_rate_pct']:5.1f}% | "
                f"unnec_esc={m['unnecessary_escalation_rate_pct']:5.1f}% | "
                f"laya_lat={laya_lat_str} | ece={ece_str}"
            )

        if summary["deltas"]:
            lines.append("-" * 98)
            lines.append(" DELTAS VS BASELINE")
            lines.append("-" * 98)
            for d in summary["deltas"]:
                lines.append(
                    f"  {d['name']:24s} | "
                    f"d_success={d.get('delta_task_success_rate_pct', 0.0):+5.1f}% | "
                    f"d_lat_mean={d.get('delta_latency_ms_mean', 0.0):+6.1f}ms | "
                    f"d_sys2_rate={d.get('delta_system2_invocation_rate_pct', 0.0):+5.1f}% | "
                    f"d_class_acc={d.get('delta_execution_class_accuracy_pct', 0.0):+5.1f}% | "
                    f"d_false_bypass={d.get('delta_false_bypass_rate_pct', 0.0):+5.1f}% | "
                    f"d_unnec_esc={d.get('delta_unnecessary_escalation_rate_pct', 0.0):+5.1f}%"
                )

        lines.append("=" * 98)
        return "\n".join(lines)


# ======================================================================
# Suite runner
# ======================================================================

class BenchmarkSuite:
    """Collection of :class:`BenchmarkCase` objects with a runner."""

    def __init__(self) -> None:
        self._cases: List[BenchmarkCase] = []

    def add(self, case: BenchmarkCase) -> None:
        self._cases.append(case)

    def add_many(self, cases: List[BenchmarkCase]) -> None:
        self._cases.extend(cases)

    @property
    def cases(self) -> List[BenchmarkCase]:
        return list(self._cases)

    async def run_all(
        self,
        graph: Any,
        *,
        filter_pattern: str = "",
        repeats: int = 1,
        is_mock_provider: bool = False,
    ) -> BenchmarkReport:
        """Execute every matching case against the compiled *graph*.

        Parameters
        ----------
        graph:
            A compiled LangGraph (the return of ``MoEGraphBuilder.build()``).
        filter_pattern:
            When non-empty, only cases matching this substring are run.
        repeats:
            How many times to repeat each case.
        is_mock_provider:
            Whether synthetic mock LLM is used (nulls out provider token metrics).
        """
        from src.core.state import create_initial_state

        cases = (
            [c for c in self._cases if c.matches_filter(filter_pattern)]
            if filter_pattern
            else self._cases
        )

        repeats = max(int(repeats), 1)

        report = BenchmarkReport()
        suite_start = time.time()

        for case in cases:
            for repeat_index in range(1, repeats + 1):
                reset_token_tracker()
                state = create_initial_state(case.query)
                t0 = time.time()
                try:
                    result_state = await graph.ainvoke(state)
                    elapsed = time.time() - t0
                    meta = result_state.get("metadata") or {}
                    retrieval_metrics = meta.get("retrieval") or {}
                    token_summary = result_state.get("token_usage", {}) or {}
                    policy_decision_dict = meta.get("policy_decision") or {}
                    laya_meta = (
                        meta.get("laya_metadata")
                        or meta.get("laya_shadow")
                        or policy_decision_dict.get("metadata", {}).get("laya_metadata")
                    )

                    system2_invoked = bool(meta.get("system2_invoked", False))
                    execution_path = str(meta.get("execution_path", ""))

                    # Real controlling execution class
                    if system2_invoked or execution_path == "system2":
                        controlling_class = "system2"
                    elif execution_path in {"direct", "single_expert"}:
                        controlling_class = execution_path
                    elif policy_decision_dict.get("execution_class") in {"direct", "single_expert", "system2"}:
                        controlling_class = policy_decision_dict["execution_class"]
                    else:
                        controlling_class = "system2"

                    primary_expert = policy_decision_dict.get("primary_expert")
                    if not primary_expert and result_state.get("selected_experts"):
                        primary_expert = result_state["selected_experts"][0]

                    laya_conf = None
                    laya_latency_ms = None
                    if laya_meta and isinstance(laya_meta, dict):
                        if laya_meta.get("confidence") is not None:
                            laya_conf = float(laya_meta["confidence"])
                        if laya_meta.get("latency_ms") is not None:
                            laya_latency_ms = float(laya_meta["latency_ms"])

                    if is_mock_provider:
                        actual_inp_toks = None
                        actual_out_toks = None
                        cost = None
                    else:
                        actual_inp_toks = token_summary.get("prompt_tokens")
                        actual_out_toks = token_summary.get("completion_tokens")
                        cost = float(token_summary.get("estimated_cost_usd", 0.0) or 0.0)

                    report.results.append(BenchmarkResult(
                        case=case,
                        success=True,
                        elapsed_seconds=elapsed,
                        token_summary=token_summary,
                        experts_used=result_state.get("selected_experts", []),
                        answer_snippet=(result_state.get("final_answer", "")[:200]),
                        repeat_index=repeat_index,
                        retry_count=max(int(result_state.get("code_execution_iterations", 0)) - 1, 0),
                        retrieval_metrics=retrieval_metrics,
                        neighborhood_reuse_rate=float(retrieval_metrics.get("neighborhood_reuse_rate", 0.0) or 0.0),
                        system2_invoked=system2_invoked,
                        laya_confidence=laya_conf,
                        laya_shadow_prediction=laya_meta or {},
                        laya_metadata=laya_meta,
                        laya_latency_ms=laya_latency_ms,
                        estimated_cost_usd=cost,
                        execution_path=execution_path,
                        execution_class=controlling_class,
                        primary_expert=primary_expert,
                        is_mock_provider=is_mock_provider,
                        actual_provider_input_tokens=actual_inp_toks,
                        actual_provider_output_tokens=actual_out_toks,
                    ))
                except asyncio.CancelledError as exc:
                    elapsed = time.time() - t0
                    report.results.append(BenchmarkResult(
                        case=case,
                        success=False,
                        elapsed_seconds=elapsed,
                        error=f"cancelled: {exc}",
                        repeat_index=repeat_index,
                        is_mock_provider=is_mock_provider,
                    ))
                except Exception as exc:
                    elapsed = time.time() - t0
                    report.results.append(BenchmarkResult(
                        case=case,
                        success=False,
                        elapsed_seconds=elapsed,
                        error=str(exc),
                        repeat_index=repeat_index,
                        is_mock_provider=is_mock_provider,
                    ))

        report.total_elapsed = time.time() - suite_start
        return report

    async def run_variant_slice(
        self,
        graphs: Dict[str, Any],
        *,
        filter_pattern: str = "",
        repeats: int = 1,
        is_mock_provider: bool = False,
    ) -> BenchmarkComparisonReport:
        variants: List[BenchmarkVariantRun] = []
        for name, graph in graphs.items():
            report = await self.run_all(
                graph,
                filter_pattern=filter_pattern,
                repeats=repeats,
                is_mock_provider=is_mock_provider,
            )
            variants.append(BenchmarkVariantRun(name=name, report=report))
        return BenchmarkComparisonReport(variants=variants)



# ======================================================================
# Standard benchmark cases (curated set)
# ======================================================================

STANDARD_CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        name="single_technical",
        query="Explain how Python's GIL works and its implications for multi-threading.",
        expected_experts=["technical"],
        tags=["single", "routing"],
    ),
    BenchmarkCase(
        name="single_creative",
        query="Write a short poem about the beauty of mathematics.",
        expected_experts=["creative"],
        tags=["single", "routing"],
    ),
    BenchmarkCase(
        name="single_analytical",
        query="Compare REST and GraphQL APIs: pros, cons, and when to use each.",
        expected_experts=["analytical"],
        tags=["single", "routing"],
    ),
    BenchmarkCase(
        name="multi_tech_analytical",
        query="Explain the technical architecture of a recommendation engine and analytically compare collaborative filtering vs content-based filtering.",
        expected_experts=["technical", "analytical"],
        tags=["multi", "routing", "parallel"],
    ),
    BenchmarkCase(
        name="multi_creative_general",
        query="Write a creative product pitch for a new AI coding assistant and provide general context about the AI tools market.",
        expected_experts=["creative", "general"],
        tags=["multi", "routing"],
    ),
    BenchmarkCase(
        name="all_experts",
        query="For a startup building an AI-powered education platform: provide technical architecture, creative branding ideas, analytical market comparison, and a general overview of the EdTech landscape.",
        expected_experts=["technical", "creative", "analytical", "general"],
        tags=["multi", "routing", "comprehensive"],
    ),
    BenchmarkCase(
        name="sequential_reasoning",
        query="Explain what a transformer architecture is, then analytically compare it to RNNs.",
        expected_experts=["technical", "analytical"],
        tags=["sequential", "reasoning"],
    ),
    BenchmarkCase(
        name="ambiguous_query",
        query="Tell me about Python.",
        expected_experts=["general"],
        tags=["ambiguous", "routing"],
    ),
    BenchmarkCase(
        name="warm_binary_search_intro",
        query="Explain how binary search works and why it is faster than linear scan on sorted arrays.",
        expected_experts=["technical"],
        tags=["warm", "graph", "retrieval"],
        family="binary_search",
    ),
    BenchmarkCase(
        name="warm_binary_search_prerequisites",
        query="Why does binary search require sorted input, and what breaks if the data is unsorted?",
        expected_experts=["technical", "analytical"],
        tags=["warm", "graph", "retrieval"],
        family="binary_search",
    ),
    BenchmarkCase(
        name="warm_binary_search_tradeoffs",
        query="Compare binary search with linear search and explain when each strategy is preferable.",
        expected_experts=["technical", "analytical"],
        tags=["warm", "graph", "retrieval"],
        family="binary_search",
    ),
    BenchmarkCase(
        name="warm_transformer_intro",
        query="Explain the role of self-attention inside a transformer architecture.",
        expected_experts=["technical"],
        tags=["warm", "graph", "retrieval"],
        family="transformers",
    ),
    BenchmarkCase(
        name="warm_transformer_scaling",
        query="Why does self-attention scale quadratically with sequence length, and what practical tradeoffs follow?",
        expected_experts=["technical", "analytical"],
        tags=["warm", "graph", "retrieval"],
        family="transformers",
    ),
    BenchmarkCase(
        name="warm_transformer_comparison",
        query="Compare transformer self-attention with recurrent sequence models for long-context reasoning.",
        expected_experts=["technical", "analytical"],
        tags=["warm", "graph", "retrieval"],
        family="transformers",
    ),
]


def create_standard_suite() -> BenchmarkSuite:
    """Return a :class:`BenchmarkSuite` pre-loaded with standard cases."""
    suite = BenchmarkSuite()
    suite.add_many(STANDARD_CASES)
    return suite


def create_research_suite(dataset_path: Optional[str | Path] = None) -> BenchmarkSuite:
    """Return a :class:`BenchmarkSuite` loaded with gold-labeled adaptive routing scenarios."""
    from benchmarks.dataset import load_adaptive_routing_dataset

    records = load_adaptive_routing_dataset(dataset_path)
    suite = BenchmarkSuite()
    for rec in records:
        suite.add(
            BenchmarkCase(
                name=rec.id,
                query=rec.query,
                expected_execution_class=rec.expected_execution_class,
                expected_primary_expert=rec.expected_primary_expert,
                expected_experts=[rec.expected_primary_expert] if rec.expected_primary_expert else [],
                tags=list(rec.tags),
                family=rec.tags[0] if rec.tags else "adaptive",
                deterministic_validation=rec.deterministic_validation,
            )
        )
    return suite

