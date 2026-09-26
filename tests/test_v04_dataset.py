"""Checks for the frozen v0.4 task pool and split boundaries."""

import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.dataset import load_adaptive_routing_dataset
from research.check_counterfactual import check
from research.counterfactual import load_candidates

ROOT = Path("research/data/v04")
DIRECT_ROOT = Path("research/data/v04_direct")


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


def test_direct_feasible_pool_is_frozen_and_separate():
    hard = load_candidates(ROOT / "candidates.jsonl")
    direct = load_candidates(DIRECT_ROOT / "candidates.jsonl")
    assert Counter(candidate.split for candidate in direct) == {
        "training": 24, "calibration": 4, "validation": 4, "final_test": 4,
    }
    assert not {case.id for case in hard} & {case.id for case in direct}
    assert not {case.query.casefold() for case in hard} & {
        case.query.casefold() for case in direct
    }
    hard_config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    direct_config = json.loads((DIRECT_ROOT / "config.json").read_text(encoding="utf-8"))
    assert direct_config["provider_settings"] == hard_config["provider_settings"]


def test_frozen_hash_catches_modified_candidate(tmp_path: Path):
    for name in ("candidates.jsonl", "manifest.json", "config.json"):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    candidate_file = tmp_path / "candidates.jsonl"
    candidate_file.write_bytes(candidate_file.read_bytes().replace(b"TASK000", b"TASK999", 1))
    with pytest.raises(ValueError, match="hash differs"):
        load_candidates(candidate_file)


def test_checker_replay_rejects_changed_result(tmp_path: Path):
    for name in ("candidates.jsonl", "manifest.json", "config.json"):
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    candidate = next(
        row for row in load_candidates(tmp_path / "candidates.jsonl")
        if row.id == "v04_structured_extraction_000"
    )
    settings = json.loads((tmp_path / "config.json").read_text())["provider_settings"]
    row = {
        "id": candidate.id,
        "query": candidate.query,
        "split": candidate.split,
        "route": "single_expert",
        "task_check_regex": candidate.task_check_regex,
        "provider_settings": settings,
        "answer": "owner000",
        "checker_passed": False,
        "error": None,
    }
    (tmp_path / "route_outcomes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checker result changed"):
        check(tmp_path)
