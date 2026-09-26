"""Run a real Laya typed decision and save device and latency measurements."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.aveniq.policy.laya_policy import LayaInferenceEngine

CONFIG = json.loads(Path(__file__).with_name("v021_config.json").read_text(encoding="utf-8"))

QUERIES = (
    "Hello.",
    "Explain how Python's GIL affects CPU-bound threads.",
    "Write a short poem about winter.",
    "Compare the costs of two database designs.",
    "Plan a data migration, test it, and explain how to roll it back.",
)


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
    }


def run_smoke_test(
    output_path: Path | str = "research/laya_smoke_results.json",
    checkpoint: str = CONFIG["checkpoint"],
    device_override: str | None = None,
) -> dict[str, Any]:
    if CONFIG["warmup_iterations"] < 20 or CONFIG["measured_iterations"] < 100:
        raise ValueError("Smoke test requires at least 20 warm-ups and 100 measurements")
    load_start = time.perf_counter()
    engine = LayaInferenceEngine(checkpoint, device_override)
    model = engine.model_metadata()
    if model["checkpoint_revision"] != CONFIG["checkpoint_revision"]:
        raise RuntimeError("Loaded Laya checkpoint revision differs from research/v021_config.json")
    load_ms = (time.perf_counter() - load_start) * 1000
    if model["device"] != model["parameter_device"]:
        raise RuntimeError("Reported device differs from Laya model parameter placement")
    if (
        device_override
        and torch.device(device_override).type == "cuda"
        and model["device"] == "cpu"
    ):
        raise RuntimeError("CUDA was requested, but Laya model parameters remain on CPU")

    cuda = model["device"].startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    typed_decision = engine.predict(QUERIES[0])
    if typed_decision["backend"] != "laya.Agent.system_one":
        raise RuntimeError("Typed decision did not use the loaded Laya Agent")

    for i in range(CONFIG["warmup_iterations"]):
        engine.predict(QUERIES[i % len(QUERIES)])

    samples: list[dict[str, Any]] = []
    for i in range(CONFIG["measured_iterations"]):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = engine.predict(QUERIES[i % len(QUERIES)])
        if cuda:
            torch.cuda.synchronize()
        samples.append(
            {
                "query_index": i % len(QUERIES),
                "preprocessing_latency_ms": prediction["preprocessing_latency_ms"],
                "forward_latency_ms": prediction["forward_latency_ms"],
                "total_policy_latency_ms": prediction["total_policy_latency_ms"],
                "wall_latency_ms": (time.perf_counter() - start) * 1000,
                "execution_class": prediction["execution_class"],
                "primary_expert": prediction["primary_expert"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
            }
        )

    fields = (
        "preprocessing_latency_ms",
        "forward_latency_ms",
        "total_policy_latency_ms",
        "wall_latency_ms",
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "config": CONFIG,
        "torch_version": torch.__version__,
        "load_latency_ms": load_ms,
        "warmup_iterations": CONFIG["warmup_iterations"],
        "measured_iterations": len(samples),
        "typed_decision": typed_decision,
        "latency_ms": {field: _stats([float(s[field]) for s in samples]) for field in fields},
        "cuda_memory_after_measurement": {
            "allocated_bytes": torch.cuda.memory_allocated() if cuda else None,
            "reserved_bytes": torch.cuda.memory_reserved() if cuda else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        },
        "samples": samples,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(destination.resolve()),
                "model": model,
                "latency_ms": payload["latency_ms"],
            },
            indent=2,
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a loaded Laya checkpoint")
    parser.add_argument("--output", default="research/laya_smoke_results.json")
    parser.add_argument("--checkpoint", default=CONFIG["checkpoint"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_smoke_test(args.output, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
