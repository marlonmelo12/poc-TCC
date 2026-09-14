"""
Checkpointing and incremental result persistence module.
Ensures crash recovery, idempotency, and complete audit trail.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Set
import pandas as pd

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "checkpoints")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "results.jsonl")


def _format_k(k: Any) -> str:
    """Normalizes K to string representation, handling None, null, floats and ints."""
    if k is None or pd.isna(k) or str(k).lower() in ("none", "null", "nan"):
        return "None"
    try:
        return str(int(k))
    except (ValueError, TypeError):
        return str(k)


def _make_key(
    dataset: Any,
    fold: Any,
    method: Any,
    classifier: Any,
    seed: Any,
    k: Any,
) -> str:
    """
    Generates a canonical, collision-free unique key for each experiment run.
    Case-insensitive for names, normalized for numeric types.
    """
    ds_str = str(dataset).strip().lower() if dataset is not None else "unknown"
    fold_str = str(int(fold)) if fold is not None else "0"
    method_str = str(method).strip().upper() if method is not None else "UNKNOWN"
    clf_str = str(classifier).strip().lower() if classifier is not None else "unknown"
    seed_str = str(int(seed)) if seed is not None else "42"
    k_str = _format_k(k)
    return f"{ds_str}_{fold_str}_{method_str}_{clf_str}_{seed_str}_{k_str}"


def _get_key(record: Dict[str, Any]) -> str:
    """Extracts the canonical key from a record dictionary."""
    return _make_key(
        dataset=record.get("dataset"),
        fold=record.get("fold"),
        method=record.get("method"),
        classifier=record.get("classifier"),
        seed=record.get("seed"),
        k=record.get("K"),
    )


class CheckpointManager:
    def __init__(self, filepath: str = CHECKPOINT_FILE):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.existing_keys: Set[str] = set()
        self._load_existing_keys()

    def _load_existing_keys(self) -> None:
        if not os.path.exists(self.filepath):
            return

        loaded_count = 0
        corrupt_count = 0

        with open(self.filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    if isinstance(data, dict) and data.get("status") == "SUCCESS":
                        self.existing_keys.add(_get_key(data))
                        loaded_count += 1
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    corrupt_count += 1
                    logger.warning(
                        "Skipping corrupted line %d in checkpoint file %s: %s",
                        line_idx, self.filepath, e
                    )

        if corrupt_count > 0:
            print(f"[CheckpointManager] Warning: skipped {corrupt_count} malformed records in {self.filepath}")
        if loaded_count > 0:
            print(f"[CheckpointManager] Loaded {len(self.existing_keys)} existing completed checkpoints from {self.filepath}")

    def is_completed(
        self,
        dataset: str,
        fold: int,
        method: str,
        classifier: str,
        seed: int,
        K: Optional[int] = None,
    ) -> bool:
        key = _make_key(dataset, fold, method, classifier, seed, K)
        return key in self.existing_keys

    def save_record(self, record: Dict[str, Any]) -> None:
        key = _get_key(record)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # Fallback on filesystems that don't support fsync on append
        if record.get("status") == "SUCCESS":
            self.existing_keys.add(key)

    def load_all_results(self) -> pd.DataFrame:
        if not os.path.exists(self.filepath):
            return pd.DataFrame()
        records = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        records.append(json.loads(line_str))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return pd.DataFrame(records)
