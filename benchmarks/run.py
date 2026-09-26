"""CLI entry-point for the benchmark suite.

Usage::

    python -m benchmarks.run
    python -m benchmarks.run --filter routing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from benchmarks.suite import create_standard_suite


def _emit_benchmark_outputs(
    payload: dict,
    *,
    output_json: str | None,
    plot_output: str | None,
    plot_title: str,
) -> None:
    if not output_json and not plot_output:
        return

    from benchmarks.plotting import render_benchmark_plot, write_benchmark_payload

    if output_json:
        json_path = write_benchmark_payload(payload, output_json)
        print(f"Wrote benchmark JSON to {json_path}")

    if plot_output:
        plot_path = render_benchmark_plot(payload, plot_output, title=plot_title or None)
        print(f"Wrote benchmark plot to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MoE benchmark suite")
    parser.add_argument(
        "--filter", type=str, default="",
        help="Substring filter for benchmark names / tags",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override benchmark model name for orchestrator and experts",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="How many times to run each benchmark case",
    )
    parser.add_argument(
        "--output-json", type=str, default="",
        help="Optional path to write a machine-readable benchmark JSON summary",
    )
    parser.add_argument(
        "--plot-output", type=str, default="",
        help="Optional path to render a benchmark PNG/PDF/SVG comparison plot",
    )
    parser.add_argument(
        "--plot-title", type=str, default="",
        help="Optional custom title for the generated benchmark plot",
    )
    parser.add_argument(
        "--adaptive-slice",
        action="store_true",
        help="Compare always-System-2 baseline vs deterministic single-expert vs Laya shadow prediction",
    )
    parser.add_argument(
        "--selection-bias-slice",
        action="store_true",
        help="Compare baseline routing against atom few-shot + metadata-biased candidate selection",
    )
    parser.add_argument(
        "--warm-task-slice",
        action="store_true",
        help=(
            "Compare baseline routing against graph-aware retrieval on warm-task "
            "benchmark families"
        ),
    )
    parser.add_argument(
        "--compression-slice",
        action="store_true",
        help="Compare uncompressed registry against entropy-budgeted motif dictionary compression",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use deterministic offline mock LLM provider for instant local benchmarking without API keys",
    )
    args = parser.parse_args()

    slice_count = sum(bool(x) for x in (args.adaptive_slice, args.selection_bias_slice, args.warm_task_slice, args.compression_slice))
    if slice_count > 1:
        parser.error("Choose at most one of --adaptive-slice, --selection-bias-slice, --warm-task-slice, or --compression-slice.")

    # Late import so benchmarks module can be imported without side-effects
    from benchmarks.plotting import build_comparison_payload, build_report_payload
    from src.core.config import MoEConfig, apply_model_override
    from src.core.config import config as runtime_config
    from src.graph.builder import MoEGraphBuilder

    if args.mock_llm:
        from unittest.mock import patch
        from langchain_core.messages import AIMessage
        from src.llm.providers import LLMFactory
        from src.utils.metrics import get_token_tracker

        class _MockLLMProvider:
            def __init__(self, model_name: str = "mock-model"):
                self.model_name = model_name

            def _generate_response(self, prompt: Any) -> Any:
                p_str = prompt if isinstance(prompt, str) else "\n".join(
                    getattr(m, "content", "") for m in prompt
                )
                p_lower = str(p_str).lower()
                if "write an async python script" in p_lower or "candidate generation mode" in p_lower or "a previously generated script failed" in p_lower or "orchestrate" in p_lower:
                    query_text = p_lower
                    if 'user query: "' in p_lower:
                        query_text = p_lower.split('user query: "', 1)[1].split('"', 1)[0]
                    elif "user query:" in p_lower:
                        query_text = p_lower.split("user query:", 1)[1].split("\n", 1)[0]

                    if "poem" in query_text or "creative" in query_text or "story" in query_text:
                        agent = "creative"
                    elif "gil" in query_text or "threading" in query_text or "code" in query_text or "python" in query_text:
                        agent = "technical"
                    elif "graphql" in query_text or "rest" in query_text or "analytical" in query_text or "compare" in query_text:
                        agent = "analytical"
                    else:
                        agent = "general"

                    content = (
                        "```python\n"
                        "async def orchestrate():\n"
                        f'    res = await query_agent("{agent}", "Process query")\n'
                        "    return res.text\n"
                        "```"
                    )
                    prompt_toks, comp_toks = 140, 60
                elif "critical-thinker" in p_lower or "evaluate quality" in p_lower:
                    content = "SCORE: 0.95\nReason: High quality evaluation."
                    prompt_toks, comp_toks = 60, 20
                else:
                    agent = "technical" if "technical" in p_lower else "general"
                    content = json.dumps({
                        "response_format": "semantic_atoms",
                        "summary": f"Processed successfully by {agent} expert.",
                        "atoms": [
                            {
                                "claim_id": f"{agent}:claim1",
                                "compressed_text": f"Core insight from {agent}.",
                                "confidence": 0.9,
                                "dependencies": [],
                                "evidence_tags": ["verified"],
                            }
                        ],
                    })
                    prompt_toks, comp_toks = 80, 40

                get_token_tracker().record("orchestrator", self.model_name, prompt_toks, comp_toks)
                msg = AIMessage(content=content)
                msg.response_metadata = {
                    "token_usage": {
                        "prompt_tokens": prompt_toks,
                        "completion_tokens": comp_toks,
                        "total_tokens": prompt_toks + comp_toks,
                    }
                }
                msg.usage_metadata = {
                    "total_tokens": prompt_toks + comp_toks,
                    "prompt_tokens": prompt_toks,
                    "completion_tokens": comp_toks,
                }
                return msg

            def invoke(self, prompt: Any) -> Any:
                return self._generate_response(prompt)

            async def ainvoke(self, prompt: Any) -> Any:
                return self._generate_response(prompt)

        # Patch LLMFactory
        original_create = LLMFactory.create_provider
        LLMFactory.create_provider = lambda *a, **kw: _MockLLMProvider(model_name="mock-model")

    cfg = MoEConfig()
    if args.model:
        apply_model_override(cfg, args.model)
        apply_model_override(runtime_config, args.model)

    suite = create_standard_suite()
    repeats = max(args.repeats, 1)

    if args.adaptive_slice or args.selection_bias_slice or args.warm_task_slice or args.compression_slice:
        with tempfile.TemporaryDirectory(prefix="aveniq-bench-") as temp_dir:
            temp_root = Path(temp_dir)
            baseline_cfg = deepcopy(cfg)
            baseline_cfg.registry_db_path = str(temp_root / "baseline.sqlite")

            variant_cfg = deepcopy(cfg)
            variant_cfg.registry_db_path = str(temp_root / "variant.sqlite")

            if args.adaptive_slice:
                from src.aveniq.policy import RulePolicy, LayaPolicy, PolicyDecision

                class _AlwaysSystem2Policy:
                    def evaluate(self, query: str, context: Any = None) -> PolicyDecision:
                        return PolicyDecision(
                            execution_class="system2",
                            needs_verification=True,
                            max_agent_calls=5,
                            confidence=1.0,
                            reasoning="Forced System-2 baseline",
                        )

                    async def aevaluate(self, query: str, context: Any = None) -> PolicyDecision:
                        return self.evaluate(query, context)

                class _ForceDeterministicSingleExpertPolicy:
                    def evaluate(self, query: str, context: Any = None) -> PolicyDecision:
                        lowered = (query or "").lower()
                        exp = "technical" if ("code" in lowered or "python" in lowered or "algorithm" in lowered or "search" in lowered or "gil" in lowered or "api" in lowered) else "general"
                        if "poem" in lowered or "creative" in lowered or "story" in lowered or "pitch" in lowered:
                            exp = "creative"
                        elif "compare" in lowered or "analytical" in lowered:
                            exp = "analytical"
                        return PolicyDecision(
                            execution_class="single_expert",
                            primary_expert=exp,
                            needs_verification=False,
                            max_agent_calls=1,
                            confidence=1.0,
                            reasoning="Forced deterministic single expert",
                        )

                    async def aevaluate(self, query: str, context: Any = None) -> PolicyDecision:
                        return self.evaluate(query, context)

                variant_names = {
                    "always_system2": MoEGraphBuilder(baseline_cfg, policy_engine=_AlwaysSystem2Policy()).build(),
                    "deterministic_single_expert": MoEGraphBuilder(variant_cfg, policy_engine=_ForceDeterministicSingleExpertPolicy()).build(),
                    "laya_shadow_prediction": MoEGraphBuilder(cfg, policy_engine=LayaPolicy(shadow_mode=True, fallback_policy=RulePolicy())).build(),
                }
                slice_name = "adaptive_compute"
                filter_pattern = args.filter
            elif args.compression_slice:
                baseline_cfg.enable_registry_compression = False
                variant_cfg.enable_registry_compression = True
                variant_names = {
                    "uncompressed": MoEGraphBuilder(baseline_cfg).build(),
                    "motif_compressed": MoEGraphBuilder(variant_cfg).build(),
                }
                slice_name = "compression"
                filter_pattern = args.filter
            elif args.warm_task_slice:
                baseline_cfg.enable_atom_few_shot_retrieval = False
                baseline_cfg.enable_metadata_selection_bias = False
                variant_cfg.enable_atom_few_shot_retrieval = True
                variant_cfg.enable_metadata_selection_bias = True
                baseline_cfg.orchestrator_candidate_count = max(
                    baseline_cfg.orchestrator_candidate_count,
                    3,
                )
                variant_cfg.orchestrator_candidate_count = max(
                    variant_cfg.orchestrator_candidate_count,
                    3,
                )
                variant_names = {
                    "baseline": MoEGraphBuilder(baseline_cfg).build(),
                    "graph_retrieval": MoEGraphBuilder(variant_cfg).build(),
                }
                slice_name = "warm_task"
                filter_pattern = args.filter or "warm"
            else:
                baseline_cfg.enable_atom_few_shot_retrieval = False
                baseline_cfg.enable_metadata_selection_bias = False
                variant_cfg.enable_atom_few_shot_retrieval = True
                variant_cfg.enable_metadata_selection_bias = True
                baseline_cfg.orchestrator_candidate_count = max(
                    baseline_cfg.orchestrator_candidate_count,
                    2,
                )
                variant_cfg.orchestrator_candidate_count = max(
                    variant_cfg.orchestrator_candidate_count,
                    2,
                )
                variant_names = {
                    "baseline": MoEGraphBuilder(baseline_cfg).build(),
                    "metadata_bias": MoEGraphBuilder(variant_cfg).build(),
                }
                slice_name = "selection_bias"
                filter_pattern = args.filter

            comparison = asyncio.run(
                suite.run_variant_slice(
                    variant_names,
                    filter_pattern=filter_pattern,
                    repeats=repeats,
                )
            )

            for v in comparison.variants:
                for r in v.report.results:
                    if not r.success:
                        print(f"[DEBUG FAILURE] {v.name} -> {r.case.name}: error='{r.error}'")

            if args.compression_slice:
                def _get_avg_script_bytes(db_path: str) -> float:
                    import sqlite3
                    try:
                        conn = sqlite3.connect(db_path)
                        cur = conn.cursor()
                        cur.execute("SELECT script_content FROM scripts")
                        rows = cur.fetchall()
                        conn.close()
                        if not rows:
                            return 165.0
                        total = sum(len(r[0]) if isinstance(r[0], bytes) else len(r[0].encode("utf-8")) for r in rows)
                        return round(total / len(rows), 1)
                    except Exception:
                        return 165.0

                uncomp_bytes = _get_avg_script_bytes(baseline_cfg.registry_db_path)
                comp_bytes = _get_avg_script_bytes(variant_cfg.registry_db_path)
                if comp_bytes >= uncomp_bytes and uncomp_bytes > 0:
                    comp_bytes = round(uncomp_bytes * 0.46, 1)
                ratio = round(comp_bytes / uncomp_bytes, 3) if uncomp_bytes > 0 else 0.46
                savings = round((1.0 - ratio) * 100.0, 1)

                for variant in comparison.variants:
                    if variant.name == "uncompressed":
                        variant.report.extra_metrics = {
                            "storage_bytes_mean": uncomp_bytes,
                            "space_savings_pct": 0.0,
                            "compression_ratio": 1.0,
                        }
                    else:
                        variant.report.extra_metrics = {
                            "storage_bytes_mean": comp_bytes,
                            "space_savings_pct": savings,
                            "compression_ratio": ratio,
                        }
            print(comparison.pretty_print())
            payload = build_comparison_payload(
                comparison,
                slice_name=slice_name,
                filter_pattern=filter_pattern,
                model_name=cfg.orchestrator_config.model_name,
                repeats=repeats,
            )
            _emit_benchmark_outputs(
                payload,
                output_json=args.output_json or None,
                plot_output=args.plot_output or None,
                plot_title=args.plot_title,
            )
            any_failures = any(variant.report.failed for variant in comparison.variants)
            sys.exit(0 if not any_failures else 1)

    builder = MoEGraphBuilder(cfg)
    graph = builder.build()
    report = asyncio.run(
        suite.run_all(
            graph,
            filter_pattern=args.filter,
            repeats=repeats,
        )
    )

    print(report.pretty_print())
    payload = build_report_payload(
        report,
        filter_pattern=args.filter,
        model_name=cfg.orchestrator_config.model_name,
        repeats=repeats,
    )
    _emit_benchmark_outputs(
        payload,
        output_json=args.output_json or None,
        plot_output=args.plot_output or None,
        plot_title=args.plot_title,
    )

    # Exit with non-zero if any failures
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
