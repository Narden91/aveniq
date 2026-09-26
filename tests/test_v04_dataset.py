"""Checks for the frozen v0.4 task pool and split boundaries."""

import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.dataset import load_adaptive_routing_dataset
from research.counterfactual import load_candidates

ROOT = Path("research/data/v04")


def test_frozen_candidate_pool_and_splits():
    candidates = load_candidates(ROOT / "candidates.jsonl")
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert len(candidates) == 500
    assert Counter(candidate.split for candidate in candidates) == {
        "training": 356, "calibration": 48, "validation": 48, "final_test": 48,
    }
    assert sum(part["count"] for part in manifest["splits"].values()) == 500
    old_test = [record for record in load_adaptive_routing_dataset() if record.split == "test"]
    assert not {candidate.id for candidate in candidates} & {record.id for record in old_test}
    assert not {candidate.query.casefold() for candidate in candidates} & {
        record.query.casefold() for record in old_test
    }


def test_frozen_hash_catches_modified_candidate(tmp_path: Path):
    for name in ("candidates.jsonl", "manifest.json", "config.json"):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    candidate_file = tmp_path / "candidates.jsonl"
    candidate_file.write_bytes(candidate_file.read_bytes().replace(b"TASK000", b"TASK999", 1))
    with pytest.raises(ValueError, match="hash differs"):
        load_candidates(candidate_file)
