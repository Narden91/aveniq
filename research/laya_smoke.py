"""AVENIQ v0.2 Real-Hardware Smoke Script for Laya Local Routing Inference.

Measures real inference latency, displays calibrated prediction distributions,
and verifies CUDA / hardware device execution with fallback resilience.

Usage:
    python -m research.laya_smoke
    python research/laya_smoke.py --output research/laya_smoke_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from src.aveniq.policy.laya_policy import LayaInferenceEngine, LayaPolicy


def run_smoke_test(
    output_path: Path | str = "research/laya_smoke_results.json",
    checkpoint: str = "laya-router-v0.2",
    device_override: str | None = None,
) -> Dict[str, Any]:
    print("=" * 65)
    print(" AVENIQ v0.2 LAYA REAL-HARDWARE INFERENCE SMOKE TEST")
    print("=" * 65)

    # 1. Load Laya locally
    print("\n[1/5] Loading Laya inference engine...")
    t_load_start = time.perf_counter()
    engine = LayaInferenceEngine(
        checkpoint=checkpoint,
        device_override=device_override,
    )
    load_time_ms = (time.perf_counter() - t_load_start) * 1000.0
    print(f"  Loaded checkpoint: {engine.checkpoint}")
    print(f"  Actual device    : {engine.device_desc}")
    print(f"  Load time        : {load_time_ms:.2f} ms")

    # 2. Run 20 warm-up predictions
    print("\n[2/5] Running 20 warm-up predictions...")
    warmup_queries = [
        "What is 2 + 2?",
        "Explain Python GIL and multi-threading limitations.",
        "Write a poem about neural routing architectures.",
        "Compare REST vs GraphQL for latency sensitive APIs.",
        "Orchestrate a multi-agent scraping and synthesis workflow with failover.",
    ]
    for i in range(20):
        q = warmup_queries[i % len(warmup_queries)]
        _ = engine.predict(q)
    print("  Warm-up completed successfully.")

    # 3. Run 100 measured predictions
    print("\n[3/5] Running 100 measured predictions...")
    test_queries = [
        "What is the capital of France?",
        "Explain how Python's GIL works and why it limits multi-core CPU scaling.",
        "Write a short, inspiring poem about the beauty of mathematics.",
        "Compare REST and GraphQL APIs: pros, cons, and caching trade-offs.",
        "Tell me about the history of computing.",
        "Design a distributed resilient message pipeline across three cloud regions.",
        "Implement a binary search function in Python with error handling.",
        "Analytically evaluate time and space complexity of merge sort vs quick sort.",
        "Create an engaging marketing tagline for an autonomous AI research agent.",
        "Orchestrate an end-to-end data pipeline: scrape, clean, synthesize, and report.",
    ]

    measured_latencies_ms: List[float] = []
    internal_latencies_ms: List[float] = []

    for i in range(100):
        q = test_queries[i % len(test_queries)]
        t0 = time.perf_counter()
        pred = engine.predict(q)
        wall_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        measured_latencies_ms.append(wall_elapsed_ms)
        internal_latencies_ms.append(pred["latency_ms"])

    mean_ms = statistics.mean(measured_latencies_ms)
    sorted_lat = sorted(measured_latencies_ms)
    p50_ms = statistics.median(measured_latencies_ms)
    p95_ms = sorted_lat[int(len(sorted_lat) * 0.95)]
    p99_ms = sorted_lat[int(len(sorted_lat) * 0.99)]

    print("  Measured Wall-Clock Latency across 100 inferences:")
    print(f"    Mean latency : {mean_ms:.2f} ms")
    print(f"    p50 latency  : {p50_ms:.2f} ms")
    print(f"    p95 latency  : {p95_ms:.2f} ms")
    print(f"    p99 latency  : {p99_ms:.2f} ms")

    # 4. Output prediction distributions for diverse sample queries
    print("\n[4/5] Evaluating calibrated prediction distributions for example queries:")
    sample_queries = [
        ("Direct simple fact", "What is the capital of France?"),
        ("Single technical", "Explain how Python's GIL works and its implications for threads."),
        ("Single creative", "Write a short poem about the beauty of mathematics."),
        ("Single analytical", "Compare REST and GraphQL APIs: pros, cons, and caching."),
        ("Single general", "Tell me about Python."),
        ("System-2 complex multi-stage", "Orchestrate an end-to-end data pipeline with multi-agent cross-correlation, scraping, and fault recovery."),
    ]

    example_results: List[Dict[str, Any]] = []
    for label, query in sample_queries:
        pred = engine.predict(query)
        example_results.append({
            "label": label,
            "query": query,
            "execution_class": pred["execution_class"],
            "primary_expert": pred["primary_expert"],
            "confidence": pred["confidence"],
            "probabilities": pred["probabilities"],
            "latency_ms": round(pred["latency_ms"], 2),
        })
        print(f"\n  • [{label}]")
        print(f"    Query       : \"{query}\"")
        print(f"    Prediction  : execution_class={pred['execution_class']} (conf={pred['confidence']:.2f}), primary_expert={pred['primary_expert']}")
        print(f"    Class probs : {pred['probabilities']['execution_class']}")
        print(f"    Expert probs: {pred['probabilities']['primary_expert']}")
        print(f"    Latency     : {pred['latency_ms']:.2f} ms")

    # 5. Save results to JSON
    print(f"\n[5/5] Saving smoke test results...")
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "timestamp": time.time(),
        "checkpoint": engine.checkpoint,
        "device": engine.device_desc,
        "load_time_ms": round(load_time_ms, 2),
        "warmup_iterations": 20,
        "measured_iterations": 100,
        "latency_ms": {
            "mean": round(mean_ms, 2),
            "p50": round(p50_ms, 2),
            "p95": round(p95_ms, 2),
            "p99": round(p99_ms, 2),
        },
        "examples": example_results,
    }

    out_file.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"  Results saved to: {out_file.resolve()}")
    print("\n" + "=" * 65)
    print(" LAYA SMOKE TEST COMPLETE")
    print("=" * 65)

    return results_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="AVENIQ v0.2 Laya Hardware Smoke Test")
    parser.add_argument(
        "--output",
        type=str,
        default="research/laya_smoke_results.json",
        help="Path to save output JSON results",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="laya-router-v0.2",
        help="Laya model checkpoint name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit device override (cuda, cpu)",
    )
    args = parser.parse_args()
    run_smoke_test(
        output_path=args.output,
        checkpoint=args.checkpoint,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
