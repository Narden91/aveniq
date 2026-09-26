"""Measure Laya routing on fixed validation and held-out test cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.dataset import load_adaptive_routing_dataset
from src.aveniq.policy.laya_policy import LayaInferenceEngine

CLASSES = ("direct", "single_expert", "system2")
CONFIG = json.loads(Path(__file__).with_name("v021_config.json").read_text(encoding="utf-8"))


def controlled_class(prediction: dict[str, Any], threshold: float) -> str:
    return "system2" if prediction["confidence"] < threshold else prediction["execution_class"]


def choose_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use validation only; a false bypass costs four other routing errors."""
    candidates = sorted({0.0, 1.0, *(float(row["prediction"]["confidence"]) for row in rows)})
    scores = []
    for threshold in candidates:
        errors = sum(
            controlled_class(row["prediction"], threshold) != row["expected_execution_class"]
            for row in rows
        )
        bypasses = sum(
            row["expected_execution_class"] == "system2"
            and controlled_class(row["prediction"], threshold) != "system2"
            for row in rows
        )
        scores.append((errors + CONFIG["false_bypass_penalty"] * bypasses, errors, threshold))
    objective, errors, threshold = min(scores)
    return {
        "threshold": threshold,
        "objective": objective,
        "class_errors": errors,
        "false_bypass_penalty": CONFIG["false_bypass_penalty"],
        "selection_split": "validation",
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "count": numerator,
        "total": denominator,
        "percent": 100 * numerator / denominator if denominator else None,
    }


def routing_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    matrix = {gold: {predicted: 0 for predicted in CLASSES} for gold in CLASSES}
    for row in rows:
        matrix[row["expected_execution_class"]][controlled_class(row["prediction"], threshold)] += 1
    total = len(rows)
    correct = sum(matrix[name][name] for name in CLASSES)
    gold_single = [row for row in rows if row["expected_execution_class"] == "single_expert"]
    expert_correct = sum(
        controlled_class(row["prediction"], threshold) == "single_expert"
        and row["prediction"]["primary_expert"] == row["expected_primary_expert"]
        for row in gold_single
    )
    raw_expert_correct = sum(
        max(
            row["prediction"]["probabilities"]["primary_expert"],
            key=row["prediction"]["probabilities"]["primary_expert"].get,
        )
        == row["expected_primary_expert"]
        for row in gold_single
    )
    system2_gold = [row for row in rows if row["expected_execution_class"] == "system2"]
    non_system2_gold = [row for row in rows if row["expected_execution_class"] != "system2"]
    false_bypass = sum(
        controlled_class(row["prediction"], threshold) != "system2" for row in system2_gold
    )
    unnecessary = sum(
        controlled_class(row["prediction"], threshold) == "system2" for row in non_system2_gold
    )
    by_class = {}
    for name in CLASSES:
        tp = matrix[name][name]
        predicted_count = sum(matrix[gold][name] for gold in CLASSES)
        actual_count = sum(matrix[name].values())
        by_class[name] = {
            "precision": _rate(tp, predicted_count),
            "recall": _rate(tp, actual_count),
        }

    brier = (
        sum(
            sum(
                (
                    float(row["prediction"]["probabilities"]["execution_class"][name])
                    - int(row["expected_execution_class"] == name)
                )
                ** 2
                for name in CLASSES
            )
            for row in rows
        )
        / total
    )
    bins: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    for row in rows:
        probabilities = row["prediction"]["probabilities"]["execution_class"]
        raw_choice = row["prediction"]["execution_class"]
        confidence = float(probabilities[raw_choice])
        bins[min(int(confidence * 10), 9)].append(
            (confidence, int(raw_choice == row["expected_execution_class"]))
        )
    ece = sum(
        len(bucket)
        / total
        * abs(
            sum(correct for _, correct in bucket) / len(bucket)
            - sum(confidence for confidence, _ in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )
    coverage = []
    for cutoff in [i / 20 for i in range(21)]:
        accepted = [row for row in rows if row["prediction"]["confidence"] >= cutoff]
        accepted_correct = sum(
            row["prediction"]["execution_class"] == row["expected_execution_class"]
            for row in accepted
        )
        coverage.append(
            {
                "threshold": cutoff,
                "coverage": _rate(len(accepted), total),
                "raw_class_accuracy": _rate(accepted_correct, len(accepted)),
            }
        )
    return {
        "sample_count": total,
        "execution_class_accuracy": _rate(correct, total),
        "primary_expert_accuracy": _rate(raw_expert_correct, len(gold_single)),
        "routed_primary_expert_accuracy": _rate(expert_correct, len(gold_single)),
        "false_bypass": _rate(false_bypass, len(system2_gold)),
        "unnecessary_escalation": _rate(unnecessary, len(non_system2_gold)),
        "confusion_matrix": matrix,
        "per_class": by_class,
        "multiclass_brier_score_raw_laya": brier,
        "ece_raw_laya": ece,
        "coverage_by_confidence": coverage,
        "system2_invocation_rate": _rate(sum(matrix[gold]["system2"] for gold in CLASSES), total),
    }


def run(output: Path, checkpoint: str, device: str | None) -> dict[str, Any]:
    records = load_adaptive_routing_dataset()
    counts = Counter(record.split for record in records)
    if len(records) < 200 or counts["validation"] == 0 or counts["test"] < 30:
        raise ValueError(f"Corpus or fixed splits are too small: {counts}")
    engine = LayaInferenceEngine(checkpoint, device)
    model = engine.model_metadata()
    if model["checkpoint_revision"] != CONFIG["checkpoint_revision"]:
        raise RuntimeError("Loaded Laya checkpoint revision differs from research/v021_config.json")
    rows = []
    for record in records:
        if record.split == "development":
            continue
        prediction = engine.predict(record.query)
        rows.append(
            {
                "id": record.id,
                "query": record.query,
                "split": record.split,
                "expected_execution_class": record.expected_execution_class,
                "expected_primary_expert": record.expected_primary_expert,
                "prediction": prediction,
            }
        )
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    selection = choose_threshold(validation)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "config": CONFIG,
        "corpus_counts": dict(counts),
        "threshold_selection": selection,
        "validation_metrics": routing_metrics(validation, selection["threshold"]),
        "held_out_test_metrics": routing_metrics(test, selection["threshold"]),
        "predictions": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("research/routing_results.json"))
    parser.add_argument("--checkpoint", default=CONFIG["checkpoint"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    payload = run(args.output, args.checkpoint, args.device)
    print(
        json.dumps(
            {
                "threshold_selection": payload["threshold_selection"],
                "held_out_test_metrics": payload["held_out_test_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
