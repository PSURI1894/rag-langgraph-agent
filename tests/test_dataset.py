"""The eval dataset is part of the contract — validate its shape."""

import json
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "evals" / "dataset.jsonl"


def load_items():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dataset_parses_and_has_items():
    items = load_items()
    assert len(items) >= 10


def test_items_have_required_fields():
    for item in load_items():
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["question"], str) and item["question"].endswith("?")
        assert isinstance(item["reference_answer"], str) and len(item["reference_answer"]) > 40
        assert isinstance(item["must_retrieve"], list) and item["must_retrieve"]


def test_ids_are_unique():
    ids = [item["id"] for item in load_items()]
    assert len(ids) == len(set(ids))
