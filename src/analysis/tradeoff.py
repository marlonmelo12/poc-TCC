"""
Dimensionality reduction trade-off, runtime profiling, QUBO energy analysis,
and feature selection stability (Jaccard) module.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


def compute_reduction_statistics(
    df: pd.DataFrame,
    dataset_features_map: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Computes dimensionality reduction rate RR = 1 - (n_features / p),
    mean selected features, and runtime profiles by method and K.
    """
    if dataset_features_map is None:
        dataset_features_map = {
            "colon": 2000,
            "prostate": 6033,
            "ovarian": 15154,
        }

    records = []
    datasets = df["dataset"].unique()

    for ds in datasets:
        p_orig = dataset_features_map.get(ds, 2000)
        ds_sub = df[df["dataset"] == ds]

        methods = ds_sub["method"].unique()
        for m in methods:
            m_sub = ds_sub[ds_sub["method"] == m]
            k_vals = m_sub["K"].unique() if "K" in m_sub.columns else [None]

            for k in k_vals:
                if pd.notnull(k):
                    km_sub = m_sub[m_sub["K"] == k]
                    method_label = f"{m} (K={int(k)})"
                else:
                    km_sub = m_sub
                    method_label = m

                n_feat_mean = float(km_sub["n_features"].mean())
                n_feat_std = float(km_sub["n_features"].std(ddof=1)) if len(km_sub) > 1 else 0.0

                red_pct = (1.0 - (n_feat_mean / p_orig)) * 100.0

                r_sel = float(km_sub["runtime_selection"].mean()) if "runtime_selection" in km_sub.columns else 0.0
                r_trn = float(km_sub["runtime_training"].mean()) if "runtime_training" in km_sub.columns else 0.0
                r_tot = float(km_sub["runtime"].mean()) if "runtime" in km_sub.columns else (r_sel + r_trn)

                f1_mean = float(km_sub["f1"].mean())
                acc_mean = float(km_sub["accuracy"].mean())

                records.append({
                    "Dataset": ds,
                    "Method": m,
                    "K": int(k) if pd.notnull(k) else None,
                    "Method_Label": method_label,
                    "Original_p": p_orig,
                    "Mean_Features": n_feat_mean,
                    "Std_Features": n_feat_std,
                    "Reduction_pct": red_pct,
                    "F1_Mean": f1_mean,
                    "Accuracy_Mean": acc_mean,
                    "Runtime_Selection_s": r_sel,
                    "Runtime_Training_s": r_trn,
                    "Runtime_Total_s": r_tot,
                })

    df_red = pd.DataFrame(records)
    return df_red


def compute_qubo_energy_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates correlation between QUBO solution energy and predictive metrics (F1, Accuracy)
    to assess whether lower minimum energy translates into superior generalization.
    """
    qubo_records = df[df["method"].isin(["QUBO-SA", "QUBO-SB"])].copy()
    if qubo_records.empty or "qubo_energy" not in qubo_records.columns:
        return pd.DataFrame()

    results = []
    for (ds, m, k), group in qubo_records.groupby(["dataset", "method", "K"]):
        valid = group.dropna(subset=["qubo_energy", "f1", "accuracy"])
        if len(valid) < 5:
            continue

        corr_f1 = valid["qubo_energy"].corr(valid["f1"])
        corr_acc = valid["qubo_energy"].corr(valid["accuracy"])

        results.append({
            "Dataset": ds,
            "Method": m,
            "K": int(k),
            "N_Evaluations": len(valid),
            "Mean_Energy": float(valid["qubo_energy"].mean()),
            "Std_Energy": float(valid["qubo_energy"].std(ddof=1)) if len(valid) > 1 else 0.0,
            "Corr_Energy_F1": float(corr_f1) if pd.notnull(corr_f1) else 0.0,
            "Corr_Energy_Accuracy": float(corr_acc) if pd.notnull(corr_acc) else 0.0,
        })

    return pd.DataFrame(results)


def compute_jaccard_stability_matrix(feature_sets: List[np.ndarray]) -> Tuple[float, float]:
    """
    Computes average pairwise Jaccard similarity across a list of selected feature index arrays:
        Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    n_sets = len(feature_sets)
    if n_sets < 2:
        return 1.0, 0.0

    similarities = []
    for i in range(n_sets):
        set_i = set(feature_sets[i])
        for j in range(i + 1, n_sets):
            set_j = set(feature_sets[j])
            union_len = len(set_i.union(set_j))
            if union_len == 0:
                sim = 1.0
            else:
                sim = len(set_i.intersection(set_j)) / float(union_len)
            similarities.append(sim)

    return float(np.mean(similarities)), float(np.std(similarities, ddof=1)) if len(similarities) > 1 else 0.0
