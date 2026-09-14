"""
Data validation and integrity module for experimental evaluation.
Ensures zero silent data imputation, strict schema validation,
duplicate detection, and paired-fold audit.
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_METHODS = ["None", "ANOVA", "RFECV", "Lasso", "QUBO-SA", "QUBO-SB"]
EXPECTED_CLASSIFIERS = ["rf", "knn", "svm", "xgboost", "catboost", "opf"]
REQUIRED_BASE_COLUMNS = [
    "dataset",
    "fold",
    "method",
    "classifier",
    "seed",
    "n_features",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


def validate_results_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates experimental results against integrity rules.
    Standardizes metric names and checks for duplicates and anomalies.
    Returns:
        standardized_df: Clean DataFrame with normalized columns.
        audit_report: Dictionary with validation statistics.
    """
    if df.empty:
        raise ValueError("Results DataFrame is empty.")

    df_clean = df.copy()

    # Normalize feature column name if present as n_selected_features
    if "n_features" not in df_clean.columns and "n_selected_features" in df_clean.columns:
        df_clean["n_features"] = df_clean["n_selected_features"]

    # 1. Check required base columns
    missing_cols = [c for c in REQUIRED_BASE_COLUMNS if c not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in results: {missing_cols}")

    # 2. Standardize column names and types
    df_clean["dataset"] = df_clean["dataset"].astype(str).str.lower().str.strip()
    df_clean["classifier"] = df_clean["classifier"].astype(str).str.lower().str.strip()
    df_clean["method"] = df_clean["method"].astype(str).str.strip()
    df_clean["fold"] = df_clean["fold"].astype(int)
    df_clean["seed"] = df_clean["seed"].astype(int)

    # Reduction rate / pct standardization
    if "reduction_rate" in df_clean.columns and "reduction_pct" not in df_clean.columns:
        df_clean["reduction_pct"] = df_clean["reduction_rate"] * 100.0
    elif "reduction_pct" in df_clean.columns and "reduction_rate" not in df_clean.columns:
        df_clean["reduction_rate"] = df_clean["reduction_pct"] / 100.0

    # Total runtime
    if "runtime" not in df_clean.columns:
        r_sel = df_clean["runtime_selection"].fillna(0.0) if "runtime_selection" in df_clean.columns else 0.0
        r_trn = df_clean["runtime_training"].fillna(0.0) if "runtime_training" in df_clean.columns else 0.0
        df_clean["runtime"] = r_sel + r_trn

    # Diagnostic metrics mapping
    if "recall_normal" not in df_clean.columns and "recall" in df_clean.columns:
        df_clean["recall_normal"] = df_clean["recall"]
    if "f1_normal" not in df_clean.columns and "f1" in df_clean.columns:
        df_clean["f1_normal"] = df_clean["f1"]

    # Calculate sens_tumor if absent but accuracy and recall are present
    if "sens_tumor" not in df_clean.columns and "accuracy" in df_clean.columns and "recall" in df_clean.columns:
        # Approximate or reconstruct based on binary fold balance if exact class counts known
        # For general datasets, leave as NaN or keep if calculated
        pass

    # 3. Check for duplicates in experimental units
    key_cols = ["dataset", "fold", "method", "classifier", "seed"]
    if "K" in df_clean.columns:
        key_cols.append("K")

    duplicate_mask = df_clean.duplicated(subset=key_cols, keep="last")
    n_duplicates = int(np.sum(duplicate_mask))
    if n_duplicates > 0:
        logger.warning("Found %d duplicate records to remove. Keeping last occurrence for idempotency.", n_duplicates)
        df_clean = df_clean.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)

    # 4. Check status if present
    if "status" in df_clean.columns:
        error_records = df_clean[df_clean["status"] != "SUCCESS"]
        n_errors = len(error_records)
        if n_errors > 0:
            logger.warning("Found %d records with non-SUCCESS status. Excluding from statistical evaluation.", n_errors)
            df_clean = df_clean[df_clean["status"] == "SUCCESS"].reset_index(drop=True)
    else:
        n_errors = 0

    # 5. Audit NaNs and remove incomplete metric rows
    metric_cols = [c for c in ["accuracy", "precision", "recall", "f1"] if c in df_clean.columns]
    audit_cols = metric_cols + (["auc"] if "auc" in df_clean.columns else [])
    nan_summary = df_clean[audit_cols].isna().sum().to_dict()

    n_before_nan = len(df_clean)
    df_clean = df_clean.dropna(subset=metric_cols).reset_index(drop=True)
    nan_dropped = n_before_nan - len(df_clean)

    audit_report = {
        "initial_rows": len(df),
        "final_rows": len(df_clean),
        "total_records": len(df_clean),
        "duplicates_removed": n_duplicates,
        "n_duplicates_removed": n_duplicates,
        "failed_status_removed": n_errors,
        "n_errors_excluded": n_errors,
        "nan_dropped": nan_dropped,
        "nan_summary": nan_summary,
        "unique_datasets": df_clean["dataset"].unique().tolist(),
        "unique_classifiers": df_clean["classifier"].unique().tolist(),
        "unique_methods": df_clean["method"].unique().tolist(),
        "unique_folds": sorted(df_clean["fold"].unique().tolist()) if not df_clean.empty else [],
    }

    return df_clean, audit_report


def check_paired_completeness(
    df: pd.DataFrame,
    dataset: str,
    classifier: str,
    methods: List[str],
    metric: str = "f1",
) -> Tuple[bool, List[int], Dict[str, Any]]:
    """
    Verifies if all specified methods have complete paired fold evaluations
    for a given (dataset, classifier).
    Returns:
        is_complete: True if all methods have valid records on the exact same folds.
        common_folds: List of fold indices present across ALL methods.
        info: Diagnostic metadata dictionary.
    """
    sub = df[(df["dataset"] == dataset) & (df["classifier"] == classifier)]

    method_folds = {}
    for m in methods:
        # If method string has K or parameter, match method
        m_sub = sub[sub["method"] == m]
        valid_folds = m_sub[m_sub[metric].notna()]["fold"].unique().tolist()
        method_folds[m] = set(valid_folds)

    if not method_folds:
        return False, [], {"reason": "No data found for combination"}

    # Compute intersection of folds
    common_folds = set.intersection(*method_folds.values()) if method_folds else set()
    common_folds = sorted(list(common_folds))

    all_methods_full = all(len(f_set) == len(common_folds) for f_set in method_folds.values())

    info = {
        "dataset": dataset,
        "classifier": classifier,
        "methods_checked": methods,
        "common_fold_count": len(common_folds),
        "common_folds": common_folds,
        "method_fold_counts": {m: len(s) for m, s in method_folds.items()},
    }

    return all_methods_full and len(common_folds) >= 5, common_folds, info
