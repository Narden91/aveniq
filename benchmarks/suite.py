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
    laya_confidence: float = 0.0
    laya_shadow_prediction: Dict[str, Any] = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    execution_path: str = ""


@dataclass
class BenchmarkReport:
    """Aggregated report across all benchmark cases."""

    results: List[BenchmarkResult] = field(default_factory=list)
    total_elapsed: float = 0.0

    # ---- Derived stats -----------------------------------------------

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def expert_accuracy(self) -> float:
        """% of cases where the actual experts matched expectations."""
        if not self.results:
            return 0.0
        correct = sum(
            1 for r in self.results
            if set(r.case.expected_experts) <= set(r.experts_used)
        )
        return correct / len(self.results) * 100

    @property
    def success_rate_pct(self) -> float:
        if not self.results:
            return 0.0
        return self.passed / len(self.results) * 100

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
        times = sorted(r.elapsed_seconds for r in self.results)
        n = len(times)
        mid = n // 2
        if n % 2 == 1:
            return times[mid]
        return (times[mid - 1] + times[mid]) / 2.0

    @property
    def latency_p95_seconds(self) -> float:
        if not self.results:
            return 0.0
        times = sorted(r.elapsed_seconds for r in self.results)
        idx = max(0, min(int(len(times) * 0.95), len(times) - 1))
        return times[idx]

    @property
    def system2_invocation_rate_pct(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.system2_invoked) / len(self.results) * 100.0

    @property
    def mean_cost_usd(self) -> float:
        costs = [
            float(r.token_summary.get("estimated_cost_usd", r.estimated_cost_usd or 0.0))
            for r in self.results
        ]
        return statistics.mean(costs) if costs else 0.0

    @property
    def mean_laya_confidence(self) -> float:
        confs = [r.laya_confidence for r in self.results if r.laya_confidence > 0.0]
        return statistics.mean(confs) if confs else 0.0

    @property
    def mean_neighborhood_reuse_rate(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.neighborhood_reuse_rate for r in self.results)

    @property
    def recovery_rate_pct(self) -> float:
        """Percentage of cases requiring retries that ultimately succeeded."""
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
            "expert_accuracy_pct": round(self.expert_accuracy, 1),
            "success_rate_pct": round(self.success_rate_pct, 1),
            "recovery_rate_pct": round(self.recovery_rate_pct, 1),
            "elapsed_mean_seconds": round(self.mean_elapsed_seconds, 3),
            "latency_p50_seconds": round(self.latency_p50_seconds, 3),
            "latency_p95_seconds": round(self.latency_p95_seconds, 3),
            "retries_mean": round(self.mean_retries, 3),
            "tokens_mean": round(self.mean_tokens, 1),
            "cost_mean_usd": round(self.mean_cost_usd, 5),
            "system2_invocation_rate_pct": round(self.system2_invocation_rate_pct, 1),
            "laya_confidence_mean": round(self.mean_laya_confidence, 3),
            "neighborhood_reuse_rate_mean": round(self.mean_neighborhood_reuse_rate, 3),
            "total_elapsed_seconds": round(self.total_elapsed, 2),
            "by_case": self.case_aggregates(),
            "by_family": self.family_aggregates(),
            "per_case": [
                {
                    "name": r.case.name,
                    "family": r.case.family,
                    "repeat_index": r.repeat_index,
                    "success": r.success,
                    "elapsed": round(r.elapsed_seconds, 2),
                    "experts_used": r.experts_used,
                    "expected_experts": r.case.expected_experts,
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
            elapsed = [r.elapsed_seconds for r in items]
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
                "elapsed_mean": round(statistics.mean(elapsed), 3),
                "elapsed_stdev": round(statistics.pstdev(elapsed), 3) if len(elapsed) > 1 else 0.0,
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
            "=" * 60,
            " BENCHMARK REPORT",
            "=" * 60,
            f"  Total cases   : {len(self.results)}",
            f"  Passed        : {self.passed}",
            f"  Failed        : {self.failed}",
            f"  Success rate  : {self.success_rate_pct:.1f}%",
            f"  Expert acc.   : {self.expert_accuracy:.1f}%",
            f"  Mean elapsed  : {self.mean_elapsed_seconds:.2f}s",
            f"  Mean retries  : {self.mean_retries:.2f}",
            f"  Mean tokens   : {self.mean_tokens:.1f}",
            f"  Mean reuse    : {self.mean_neighborhood_reuse_rate:.2f}",
            f"  Total time    : {self.total_elapsed:.2f}s",
            "-" * 60,
        ]
        for r in self.results:
            mark = "PASS" if r.success else "FAIL"
            lines.append(
                f"  [{mark}] {r.case.name:30s}  "
                f"{r.elapsed_seconds:.2f}s  experts={r.experts_used}"
            )
            if r.error:
                lines.append(f"         ERROR: {r.error[:120]}")

        if self.results and max(r.repeat_index for r in self.results) > 1:
            lines.append("-" * 60)
            lines.append(" AGGREGATES (BY CASE)")
            for row in self.case_aggregates():
                lines.append(
                    "  {name:30s} runs={runs:<2d} pass={pass_rate_pct:5.1f}% "
                    "elapsed={elapsed_mean:.2f}s±{elapsed_stdev:.2f} retries={retries_mean:.2f} tokens={tokens_mean:.1f} reuse={neighborhood_reuse_rate_mean:.2f}".format(**row)
                )

        family_rows = self.family_aggregates()
        if family_rows:
            lines.append("-" * 60)
            lines.append(" FAMILY AGGREGATES")
            for row in family_rows:
                lines.append(
                    "  {family:24s} runs={runs:<2d} pass={pass_rate_pct:5.1f}% elapsed={elapsed_mean:.2f}s retries={retries_mean:.2f} tokens={tokens_mean:.1f} reuse={neighborhood_reuse_rate_mean:.2f}".format(**row)
                )

        lines.append("=" * 60)
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
            "success_rate_pct": baseline.report.success_rate_pct,
            "elapsed_mean_seconds": baseline.report.mean_elapsed_seconds,
            "latency_p50_seconds": baseline.report.latency_p50_seconds,
            "latency_p95_seconds": baseline.report.latency_p95_seconds,
            "tokens_mean": baseline.report.mean_tokens,
            "cost_mean_usd": baseline.report.mean_cost_usd,
            "system2_invocation_rate_pct": baseline.report.system2_invocation_rate_pct,
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
                "success_rate_pct": round(variant.report.success_rate_pct, 1),
                "elapsed_mean_seconds": round(variant.report.mean_elapsed_seconds, 3),
                "latency_p50_seconds": round(variant.report.latency_p50_seconds, 3),
                "latency_p95_seconds": round(variant.report.latency_p95_seconds, 3),
                "tokens_mean": round(variant.report.mean_tokens, 1),
                "cost_mean_usd": round(variant.report.mean_cost_usd, 5),
                "system2_invocation_rate_pct": round(variant.report.system2_invocation_rate_pct, 1),
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
                "delta_success_rate_pct": round(variant.report.success_rate_pct - baseline_metrics["success_rate_pct"], 1),
                "delta_elapsed_mean_seconds": round(variant.report.mean_elapsed_seconds - baseline_metrics["elapsed_mean_seconds"], 3),
                "delta_latency_p50_seconds": round(variant.report.latency_p50_seconds - baseline_metrics["latency_p50_seconds"], 3),
                "delta_latency_p95_seconds": round(variant.report.latency_p95_seconds - baseline_metrics["latency_p95_seconds"], 3),
                "delta_tokens_mean": round(variant.report.mean_tokens - baseline_metrics["tokens_mean"], 1),
                "delta_cost_mean_usd": round(variant.report.mean_cost_usd - baseline_metrics["cost_mean_usd"], 5),
                "delta_system2_invocation_rate_pct": round(variant.report.system2_invocation_rate_pct - baseline_metrics["system2_invocation_rate_pct"], 1),
                "delta_expert_accuracy_pct": round(variant.report.expert_accuracy - baseline_metrics["expert_accuracy_pct"], 1),
                "delta_laya_confidence_mean": round(variant.report.mean_laya_confidence - baseline_metrics["laya_confidence_mean"], 3),
                "delta_retries_mean": round(variant.report.mean_retries - baseline_metrics["retries_mean"], 3),
                "delta_neighborhood_reuse_rate_mean": round(variant.report.mean_neighborhood_reuse_rate - baseline_metrics["neighborhood_reuse_rate_mean"], 3),
            }
            for k, v in extra.items():
                if k in baseline_metrics:
                    delta_dict[f"delta_{k}"] = round(float(v) - float(baseline_metrics[k]), 3)
            deltas.append(delta_dict)

        return {"variants": variants, "deltas": deltas}

    def pretty_print(self) -> str:
        lines = [
            "=" * 78,
            " AVENIQ BENCHMARK SLICE REPORT",
            "=" * 78,
        ]
        summary = self.summary()
        for item in summary["variants"]:
            m = item["metrics"]
            lines.append(
                f"  {item['name']:26s} | success={m['success_rate_pct']:5.1f}% | "
                f"lat(mean/p50/p95)={m['elapsed_mean_seconds']:.2f}/{m['latency_p50_seconds']:.2f}/{m['latency_p95_seconds']:.2f}s | "
                f"tokens={m['tokens_mean']:5.0f} | cost=${m['cost_mean_usd']:.4f} | "
                f"sys2_rate={m['system2_invocation_rate_pct']:5.1f}% | "
                f"routing_acc={m['expert_accuracy_pct']:5.1f}% | "
                f"laya_conf={m['laya_confidence_mean']:.2f}"
            )

        if summary["deltas"]:
            lines.append("-" * 78)
            lines.append(" DELTAS VS BASELINE")
            lines.append("-" * 78)
            for d in summary["deltas"]:
                lines.append(
                    f"  {d['name']:26s} | "
                    f"d_success={d['delta_success_rate_pct']:+5.1f}% | "
                    f"d_lat(mean/p50/p95)={d['delta_elapsed_mean_seconds']:+5.2f}/{d['delta_latency_p50_seconds']:+5.2f}/{d['delta_latency_p95_seconds']:+5.2f}s | "
                    f"d_tokens={d['delta_tokens_mean']:+5.0f} | "
                    f"d_sys2_rate={d['delta_system2_invocation_rate_pct']:+5.1f}% | "
                    f"d_acc={d['delta_expert_accuracy_pct']:+5.1f}%"
                )

        lines.append("=" * 78)
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
    ) -> BenchmarkReport:
        """Execute every matching case against the compiled *graph*.

        Parameters
        ----------
        graph:
            A compiled LangGraph (the return of ``MoEGraphBuilder.build()``).
        filter_pattern:
            When non-empty, only cases matching this substring are run.
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
                    laya_shadow = meta.get("laya_shadow") or {}
                    laya_conf = float(laya_shadow.get("confidence", 0.0) or 0.0)
                    cost = float(token_summary.get("estimated_cost_usd", 0.0) or 0.0)
                    system2_invoked = bool(meta.get("system2_invoked", False))
                    execution_path = str(meta.get("execution_path", ""))

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
                        laya_shadow_prediction=laya_shadow,
                        estimated_cost_usd=cost,
                        execution_path=execution_path,
                    ))
                except asyncio.CancelledError as exc:
                    elapsed = time.time() - t0
                    report.results.append(BenchmarkResult(
                        case=case,
                        success=False,
                        elapsed_seconds=elapsed,
                        error=f"cancelled: {exc}",
                        repeat_index=repeat_index,
                    ))
                except Exception as exc:
                    elapsed = time.time() - t0
                    report.results.append(BenchmarkResult(
                        case=case,
                        success=False,
                        elapsed_seconds=elapsed,
                        error=str(exc),
                        repeat_index=repeat_index,
                    ))

        report.total_elapsed = time.time() - suite_start
        return report

    async def run_variant_slice(
        self,
        graphs: Dict[str, Any],
        *,
        filter_pattern: str = "",
        repeats: int = 1,
    ) -> BenchmarkComparisonReport:
        variants: List[BenchmarkVariantRun] = []
        for name, graph in graphs.items():
            report = await self.run_all(graph, filter_pattern=filter_pattern, repeats=repeats)
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
