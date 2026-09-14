"""
Friedman and Nemenyi non-parametric statistical testing module.
Evaluates global ranking differences across paired folds and performs
post-hoc Critical Difference (CD) analysis (Demšar, 2006).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata
import scikit_posthocs as sp


def build_paired_matrix(
    df: pd.DataFrame,
    dataset: str,
    classifier: Optional[str] = None,
    metric: str = "f1",
    methods: Optional[List[str]] = None,
    k_val: Optional[int] = None,
) -> pd.DataFrame:
    """
    Constructs a paired observations matrix.
    Rows: Experimental blocks (Folds, or Classifier x Fold).
    Columns: Methods.

    If classifier is provided:
        Rows are individual folds for that (dataset, classifier).
    If classifier is None:
        Rows are blocks of (classifier, fold) for global dataset-level comparison.
    """
    if methods is None:
        methods = ["None", "ANOVA", "RFECV", "Lasso", "QUBO-SA", "QUBO-SB"]

    sub = df[df["dataset"] == dataset].copy()
    if classifier is not None:
        sub = sub[sub["classifier"] == classifier]

    # Filter K for methods that have K
    filtered_rows = []
    for _, row in sub.iterrows():
        m = row["method"]
        k = row.get("K")
        if m in ["ANOVA", "QUBO-SA", "QUBO-SB"]:
            if k_val is not None:
                if pd.notnull(k) and int(k) == int(k_val):
                    filtered_rows.append(row)
            else:
                # If no k_val specified, include all
                filtered_rows.append(row)
        else:
            filtered_rows.append(row)

    if not filtered_rows:
        return pd.DataFrame()

    sub_filtered = pd.DataFrame(filtered_rows)

    # Define block identifier
    if classifier is not None:
        sub_filtered["block"] = sub_filtered["fold"].astype(str)
    else:
        sub_filtered["block"] = sub_filtered["classifier"] + "_f" + sub_filtered["fold"].astype(str)

    # Pivot to matrix form
    pivot = sub_filtered.pivot_table(
        index="block",
        columns="method",
        values=metric,
        aggfunc="mean",
    )

    # Keep only requested methods that exist in columns
    available_methods = [m for m in methods if m in pivot.columns]
    pivot = pivot[available_methods]

    # Drop any blocks with missing method evaluations to preserve strict pairing
    pivot_clean = pivot.dropna(how="any")

    return pivot_clean


def compute_method_ranks(
    paired_matrix: pd.DataFrame,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Computes average ranks across paired blocks per Demšar (2006).
    Best performing algorithm receives rank 1.
    """
    if paired_matrix.empty:
        return pd.DataFrame()

    matrix_vals = paired_matrix.values
    if higher_is_better:
        # Negate values so that largest score gets rank 1
        ranks = rankdata(-matrix_vals, axis=1, method="average")
    else:
        ranks = rankdata(matrix_vals, axis=1, method="average")

    mean_ranks = np.mean(ranks, axis=0)
    std_ranks = np.std(ranks, axis=0, ddof=1) if ranks.shape[0] > 1 else np.zeros_like(mean_ranks)

    ranks_df = pd.DataFrame({
        "Method": paired_matrix.columns,
        "Mean_Rank": mean_ranks,
        "Std_Rank": std_ranks,
    }).sort_values("Mean_Rank").reset_index(drop=True)

    return ranks_df


def run_friedman_test(
    paired_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Runs paired Friedman non-parametric omnibus test across columns.
    H0: All methods have identical performance distribution.
    H1: At least one method has a different distribution.
    """
    if paired_matrix.empty or paired_matrix.shape[0] < 3 or paired_matrix.shape[1] < 2:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "N": paired_matrix.shape[0],
            "k": paired_matrix.shape[1],
            "significant": False,
            "alpha": alpha,
            "status": "insufficient_data",
        }

    # Extract arrays for each column
    columns_data = [paired_matrix[col].values for col in paired_matrix.columns]
    res = friedmanchisquare(*columns_data)

    stat = float(res.statistic)
    p_val = float(res.pvalue)

    return {
        "statistic": stat,
        "p_value": p_val,
        "N": int(paired_matrix.shape[0]),
        "k": int(paired_matrix.shape[1]),
        "significant": bool(p_val < alpha),
        "alpha": alpha,
        "status": "SUCCESS",
    }


def compute_critical_difference_value(
    k: int,
    N: int,
    alpha: float = 0.05,
) -> float:
    """
    Computes Critical Difference (CD) threshold for Nemenyi test:
        CD = q_alpha * sqrt(k * (k + 1) / (6 * N))
    Uses standard Studentized range critical values for two-tailed Nemenyi.
    """
    # Standard two-tailed Nemenyi critical values q_alpha / sqrt(2) for alpha=0.05
    # k from 2 to 10 (Demšar, 2006, Table 5)
    q_alpha_0_05 = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }
    q_val = q_alpha_0_05.get(k, 2.850)
    cd = q_val * np.sqrt((k * (k + 1.0)) / (6.0 * N))
    return float(cd)


def run_nemenyi_test(
    paired_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Executes Nemenyi post-hoc test on paired matrix.
    Returns:
        pairwise_results: DataFrame of pairwise comparisons (Method_A, Method_B, p_value, significant).
        p_matrix: Symmetric k x k DataFrame of p-values.
        cd_value: Critical Difference value.
    """
    if paired_matrix.empty or paired_matrix.shape[0] < 3 or paired_matrix.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame(), np.nan

    methods = list(paired_matrix.columns)
    k = len(methods)
    N = paired_matrix.shape[0]

    # scikit-posthocs posthoc_nemenyi_friedman expects observations
    # It takes matrix where rows are blocks and columns are groups
    p_mat = sp.posthoc_nemenyi_friedman(paired_matrix.values)
    p_mat.index = methods
    p_mat.columns = methods

    cd_val = compute_critical_difference_value(k, N, alpha=alpha)

    # Flatten upper triangle into pairwise rows
    pairwise_rows = []
    for i in range(k):
        for j in range(i + 1, k):
            m_a = methods[i]
            m_b = methods[j]
            p_val = float(p_mat.iloc[i, j])
            pairwise_rows.append({
                "Method_A": m_a,
                "Method_B": m_b,
                "p_value": p_val,
                "Significant": bool(p_val < alpha),
            })

    pairwise_df = pd.DataFrame(pairwise_rows)
    return pairwise_df, p_mat, cd_val
