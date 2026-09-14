"""
Unit tests for CheckpointManager:
- Valid key loading and crash recovery
- Corrupted/malformed line resilience
- Key canonicalization consistency
- Atomic save and flush
"""

import os
import json
import pytest
from src.pipeline.checkpoint import CheckpointManager, _get_key, _format_k


def test_checkpoint_load_existing_keys(tmp_path):
    checkpoint_file = str(tmp_path / "test_results.jsonl")

    records = [
        {"dataset": "colon", "fold": 0, "method": "None", "classifier": "rf", "seed": 42, "K": None, "status": "SUCCESS"},
        {"dataset": "colon", "fold": 0, "method": "ANOVA", "classifier": "svm", "seed": 42, "K": 10, "status": "SUCCESS"},
        {"dataset": "colon", "fold": 1, "method": "Lasso", "classifier": "knn", "seed": 42, "K": 25, "status": "ERROR: fail"},
    ]

    with open(checkpoint_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cm = CheckpointManager(filepath=checkpoint_file)

    # SUCCESS records must be loaded
    assert cm.is_completed("colon", 0, "None", "rf", 42, None) is True
    assert cm.is_completed("colon", 0, "ANOVA", "svm", 42, 10) is True

    # ERROR records must NOT be marked as completed (allows retry)
    assert cm.is_completed("colon", 1, "Lasso", "knn", 42, 25) is False

    # Non-existent records must return False
    assert cm.is_completed("colon", 2, "None", "rf", 42, None) is False


def test_checkpoint_corrupted_line_resilience(tmp_path):
    checkpoint_file = str(tmp_path / "corrupt_results.jsonl")

    with open(checkpoint_file, "w", encoding="utf-8") as f:
        f.write('{"dataset": "colon", "fold": 0, "method": "None", "classifier": "rf", "seed": 42, "K": null, "status": "SUCCESS"}\n')
        f.write('{"broken json line without closing bracket\n')
        f.write('   \n')
        f.write('{"dataset": "colon", "fold": 0, "method": "ANOVA", "classifier": "knn", "seed": 42, "K": 20, "status": "SUCCESS"}\n')

    cm = CheckpointManager(filepath=checkpoint_file)

    assert len(cm.existing_keys) == 2
    assert cm.is_completed("colon", 0, "None", "rf", 42, None) is True
    assert cm.is_completed("colon", 0, "ANOVA", "knn", 42, 20) is True


def test_checkpoint_save_and_immediate_flush(tmp_path):
    checkpoint_file = str(tmp_path / "save_test.jsonl")
    cm = CheckpointManager(filepath=checkpoint_file)

    rec = {"dataset": "prostate", "fold": 2, "method": "QUBO-SB", "classifier": "catboost", "seed": 42, "K": 20, "status": "SUCCESS"}
    cm.save_record(rec)

    assert cm.is_completed("prostate", 2, "QUBO-SB", "catboost", 42, 20) is True

    # Verify physical file contains the line immediately
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["method"] == "QUBO-SB"
