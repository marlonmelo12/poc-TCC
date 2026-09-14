"""
Paired Wilcoxon signed-rank testing, Holm-Bonferroni Family-Wise Error Rate (FWER)
correction, and Rank-Biserial Correlation effect size estimation.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests


PLANNED_COMPARISONS = [
    ("QUBO-SA", "None"),
    ("QUBO-SA", "ANOVA"),
    ("QUBO-SA", "RFECV"),
    ("QUBO-SA", "Lasso"),
    ("QUBO-SA", "QUBO-SB"),
    ("QUBO-SB", "None"),
    ("QUBO-SB", "ANOVA"),
    ("QUBO-SB", "RFECV"),
    ("QUBO-SB", "Lasso"),
]


def compute_rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Computes Rank-Biserial Correlation (r_rb) for paired observations:
        r_rb = (W+ - W-) / (W+ + W-)
    Interpretation rule:
        |r_rb| < 0.10: Negligible
        0.10 <= |r_rb| < 0.30: Small
        0.30 <= |r_rb| < 0.50: Medium
        |r_rb| >= 0.50: Large
    """
    diffs = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    nonzero_diffs = diffs[diffs != 0.0]

    if len(nonzero_diffs) == 0:
        return 0.0, "Negligível"

    abs_diffs = np.abs(nonzero_diffs)
    ranks = rankdata(abs_diffs, method="average")

    w_pos = float(np.sum(ranks[nonzero_diffs > 0.0]))
    w_neg = float(np.sum(ranks[nonzero_diffs < 0.0]))
    w_total = w_pos + w_neg

    if w_total == 0.0:
        return 0.0, "Negligível"

    r_rb = float((w_pos - w_neg) / w_total)
    abs_r = abs(r_rb)

    if abs_r < 0.10:
        interp = "Negligível"
    elif abs_r < 0.30:
        interp = "Pequeno"
    elif abs_r < 0.50:
        interp = "Médio"
    else:
        interp = "Grande"

    return r_rb, interp


def run_planned_wilcoxon_tests(
    df: pd.DataFrame,
    dataset: str,
    classifier: Optional[str] = None,
    metric: str = "f1",
    k_val: Optional[int] = 20,
    alpha: float = 0.05,
    custom_comparisons: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Executes planned directional paired Wilcoxon signed-rank tests
    with Holm-Bonferroni correction and Rank-Biserial Correlation effect size.
    """
    comparisons_to_run = custom_comparisons or PLANNED_COMPARISONS

    sub = df[df["dataset"] == dataset].copy()
    if classifier is not None:
        sub = sub[sub["classifier"] == classifier]

    # Prepare method-keyed score vectors indexed by block (fold or clf_fold)
    method_scores = {}
    for m in sub["method"].unique():
        m_sub = sub[sub["method"] == m]
        if m in ["ANOVA", "QUBO-SA", "QUBO-SB"] and k_val is not None and "K" in m_sub.columns:
            m_sub = m_sub[m_sub["K"] == k_val]

        if classifier is not None:
            m_sub = m_sub.sort_values("fold")
            method_scores[m] = dict(zip(m_sub["fold"], m_sub[metric]))
        else:
            m_sub = m_sub.copy()
            m_sub["block"] = m_sub["classifier"] + "_f" + m_sub["fold"].astype(str)
            method_scores[m] = dict(zip(m_sub["block"], m_sub[metric]))

    records = []
    raw_p_values = []

    for m_a, m_b in comparisons_to_run:
        if m_a not in method_scores or m_b not in method_scores:
            continue

        dict_a = method_scores[m_a]
        dict_b = method_scores[m_b]

        # Common paired blocks
        common_blocks = sorted(list(set(dict_a.keys()).intersection(set(dict_b.keys()))))
        if len(common_blocks) < 5:
            continue

        scores_a = np.array([dict_a[b] for b in common_blocks], dtype=np.float64)
        scores_b = np.array([dict_b[b] for b in common_blocks], dtype=np.float64)

        # Drop NaNs pairwise
        valid_mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
        scores_a = scores_a[valid_mask]
        scores_b = scores_b[valid_mask]

        if len(scores_a) < 5:
            continue

        diffs = scores_a - scores_b
        mean_delta = float(np.mean(diffs))

        if np.all(diffs == 0.0):
            stat = 0.0
            p_val = 1.0
        else:
            try:
                res = wilcoxon(scores_a, scores_b, zero_method="pratt")
                stat = float(res.statistic)
                p_val = float(res.pvalue)
            except Exception:
                stat = np.nan
                p_val = 1.0

        r_rb, effect_interp = compute_rank_biserial_correlation(scores_a, scores_b)

        rec = {
            "Dataset": dataset,
            "Classifier": classifier if classifier is not None else "All_Classifiers",
            "Metric": metric,
            "Method_A": f"{m_a} (K={k_val})" if m_a in ["ANOVA", "QUBO-SA", "QUBO-SB"] and k_val else m_a,
            "Method_B": f"{m_b} (K={k_val})" if m_b in ["ANOVA", "QUBO-SA", "QUBO-SB"] and k_val else m_b,
            "N_Pairs": len(scores_a),
            "Mean_A": float(np.mean(scores_a)),
            "Mean_B": float(np.mean(scores_b)),
            "Delta_Mean": mean_delta,
            "Statistic": stat,
            "p_raw": p_val,
            "Effect_Size_r_rb": r_rb,
            "Effect_Interpretation": effect_interp,
        }
        records.append(rec)
        raw_p_values.append(p_val)

    if not records:
        return pd.DataFrame()

    # Apply Holm-Bonferroni correction
    _, p_holm, _, _ = multipletests(raw_p_values, alpha=alpha, method="holm")

    for i, p_h in enumerate(p_holm):
        records[i]["p_holm"] = float(p_h)
        records[i]["Significant_Holm"] = bool(p_h < alpha)

    results_df = pd.DataFrame(records)
    return results_df
