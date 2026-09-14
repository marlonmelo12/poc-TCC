"""
Unit tests for the Statistical Analysis module.
Covers:
- DataFrame validation and completeness checking (validation.py)
- Paired matrix construction, ranking, Friedman & Nemenyi tests (friedman_nemenyi.py)
- Wilcoxon signed-rank test, Holm-Bonferroni correction, Rank-Biserial correlation (wilcoxon_holm.py)
- Dimensionality reduction, runtime profiling, Jaccard stability (tradeoff.py)
- Full pipeline integration and report generation (stats.py, reporting.py)
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.analysis.validation import validate_results_dataframe, check_paired_completeness
from src.analysis.friedman_nemenyi import (
    build_paired_matrix,
    compute_method_ranks,
    run_friedman_test,
    run_nemenyi_test,
)
from src.analysis.wilcoxon_holm import (
    run_planned_wilcoxon_tests,
    compute_rank_biserial_correlation,
)
from src.analysis.tradeoff import (
    compute_reduction_statistics,
    compute_jaccard_stability_matrix,
)
from src.analysis.stats import run_full_statistical_pipeline


@pytest.fixture
def synthetic_results_df():
    """Generates a synthetic results DataFrame with 5 folds, 3 classifiers, 4 methods."""
    np.random.seed(42)
    records = []
    methods = ["None", "ANOVA", "QUBO-SA", "QUBO-SB"]
    classifiers = ["rf", "svm", "knn"]
    n_folds = 5

    base_scores = {
        "None": 0.75,
        "ANOVA": 0.80,
        "QUBO-SA": 0.85,
        "QUBO-SB": 0.83,
    }

    feature_counts = {
        "None": 2000,
        "ANOVA": 20,
        "QUBO-SA": 18,
        "QUBO-SB": 19,
    }

    for fold in range(n_folds):
        for clf in classifiers:
            for method in methods:
                f1 = base_scores[method] + np.random.normal(0, 0.02)
                f1 = min(max(f1, 0.0), 1.0)
                acc = f1 + np.random.normal(0, 0.01)
                acc = min(max(acc, 0.0), 1.0)
                k_val = 20 if method != "None" else None

                records.append({
                    "dataset": "colon",
                    "fold": fold,
                    "method": method,
                    "classifier": clf,
                    "seed": 42,
                    "K": k_val,
                    "status": "SUCCESS",
                    "f1": f1,
                    "accuracy": acc,
                    "precision": acc,
                    "recall": acc,
                    "auc": acc,
                    "selected_features": list(range(feature_counts[method])),
                    "runtime_selection": 0.5 if "QUBO" not in method else 3.5,
                    "runtime_training": 1.0 if "QUBO" not in method else 4.0,
                    "train_samples": 50,
                    "test_samples": 12,
                    "n_features": feature_counts[method],
                })

    return pd.DataFrame(records)


# =========================================================================
# 1. Validation Tests
# =========================================================================

def test_validation_clean_data(synthetic_results_df):
    df_clean, audit = validate_results_dataframe(synthetic_results_df)
    assert len(df_clean) == len(synthetic_results_df)
    assert audit["initial_rows"] == len(synthetic_results_df)
    assert audit["final_rows"] == len(synthetic_results_df)
    assert audit["duplicates_removed"] == 0
    assert audit["nan_dropped"] == 0


def test_validation_detects_duplicates_and_nan():
    df = pd.DataFrame([
        {"dataset": "colon", "fold": 0, "method": "ANOVA", "classifier": "rf", "seed": 42, "K": 20, "n_features": 20, "status": "SUCCESS", "f1": 0.80, "accuracy": 0.80, "precision": 0.80, "recall": 0.80},
        {"dataset": "colon", "fold": 0, "method": "ANOVA", "classifier": "rf", "seed": 42, "K": 20, "n_features": 20, "status": "SUCCESS", "f1": 0.82, "accuracy": 0.82, "precision": 0.82, "recall": 0.82},
        {"dataset": "colon", "fold": 1, "method": "ANOVA", "classifier": "rf", "seed": 42, "K": 20, "n_features": 20, "status": "SUCCESS", "f1": np.nan, "accuracy": 0.80, "precision": 0.80, "recall": 0.80},
        {"dataset": "colon", "fold": 2, "method": "ANOVA", "classifier": "rf", "seed": 42, "K": 20, "n_features": 20, "status": "ERROR: crash", "f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0},
    ])

    df_clean, audit = validate_results_dataframe(df)
    assert len(df_clean) == 1
    assert audit["duplicates_removed"] == 1
    assert audit["failed_status_removed"] == 1
    assert audit["nan_dropped"] == 1


def test_check_paired_completeness(synthetic_results_df):
    is_complete, common_folds, info = check_paired_completeness(
        synthetic_results_df,
        dataset="colon",
        classifier="rf",
        methods=["None", "ANOVA", "QUBO-SA", "QUBO-SB"],
    )
    assert is_complete is True
    assert len(common_folds) == 5

    df_incomplete = synthetic_results_df[~((synthetic_results_df["method"] == "QUBO-SA") & (synthetic_results_df["fold"] == 0))]
    is_comp_inc, common_inc, _ = check_paired_completeness(
        df_incomplete,
        dataset="colon",
        classifier="rf",
        methods=["None", "ANOVA", "QUBO-SA", "QUBO-SB"],
    )
    assert is_comp_inc is False
    assert 0 not in common_inc


# =========================================================================
# 2. Friedman & Nemenyi Tests
# =========================================================================

def test_paired_matrix_and_ranks(synthetic_results_df):
    matrix = build_paired_matrix(
        synthetic_results_df,
        dataset="colon",
        classifier=None,
        metric="f1",
        methods=["None", "ANOVA", "QUBO-SA", "QUBO-SB"],
        k_val=20,
    )
    assert matrix.shape == (15, 4)

    ranks_df = compute_method_ranks(matrix, higher_is_better=True)
    assert len(ranks_df) == 4
    sa_rank = ranks_df.loc[ranks_df["Method"] == "QUBO-SA", "Mean_Rank"].iloc[0]
    none_rank = ranks_df.loc[ranks_df["Method"] == "None", "Mean_Rank"].iloc[0]
    assert sa_rank < none_rank


def test_friedman_test(synthetic_results_df):
    matrix = build_paired_matrix(
        synthetic_results_df,
        dataset="colon",
        classifier=None,
        metric="f1",
        methods=["None", "ANOVA", "QUBO-SA", "QUBO-SB"],
        k_val=20,
    )
    res = run_friedman_test(matrix, alpha=0.05)
    assert "statistic" in res
    assert "p_value" in res
    assert res["N"] == 15
    assert res["k"] == 4
    assert res["p_value"] < 0.001
    assert res["significant"] is True


def test_nemenyi_test(synthetic_results_df):
    matrix = build_paired_matrix(
        synthetic_results_df,
        dataset="colon",
        classifier=None,
        metric="f1",
        methods=["None", "ANOVA", "QUBO-SA", "QUBO-SB"],
        k_val=20,
    )
    pairwise_df, p_mat, cd_val = run_nemenyi_test(matrix, alpha=0.05)
    assert cd_val > 0
    assert p_mat.shape == (4, 4)
    for m in matrix.columns:
        assert p_mat.loc[m, m] == 1.0 or np.isnan(p_mat.loc[m, m])
    assert np.isclose(p_mat.loc["None", "QUBO-SA"], p_mat.loc["QUBO-SA", "None"])


# =========================================================================
# 3. Wilcoxon + Holm & Effect Size Tests
# =========================================================================

def test_rank_biserial_correlation():
    x_better = np.array([0.9, 0.85, 0.88, 0.92, 0.87])
    y_worse = np.array([0.7, 0.65, 0.68, 0.72, 0.67])
    r_pos, mag_pos = compute_rank_biserial_correlation(x_better, y_worse)
    assert np.isclose(r_pos, 1.0)
    assert mag_pos == "Grande"

    r_neg, mag_neg = compute_rank_biserial_correlation(y_worse, x_better)
    assert np.isclose(r_neg, -1.0)
    assert mag_neg == "Grande"

    x_same = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    r_zero, mag_zero = compute_rank_biserial_correlation(x_same, x_same)
    assert np.isclose(r_zero, 0.0)
    assert mag_zero == "Negligível"


def test_planned_wilcoxon_tests(synthetic_results_df):
    res_df = run_planned_wilcoxon_tests(
        synthetic_results_df,
        dataset="colon",
        classifier=None,
        metric="f1",
        alpha=0.05,
        k_val=20,
    )
    assert not res_df.empty
    expected_cols = [
        "Dataset", "Classifier", "Metric", "Method_A", "Method_B",
        "N_Pairs", "Mean_A", "Mean_B", "Delta_Mean", "Statistic",
        "p_raw", "Effect_Size_r_rb", "Effect_Interpretation",
        "p_holm", "Significant_Holm",
    ]
    for c in expected_cols:
        assert c in res_df.columns

    for _, row in res_df.iterrows():
        assert row["p_holm"] >= row["p_raw"] - 1e-9


# =========================================================================
# 4. Trade-off and Stability Tests
# =========================================================================

def test_reduction_statistics(synthetic_results_df):
    stats_df = compute_reduction_statistics(synthetic_results_df)
    assert not stats_df.empty
    assert "Reduction_pct" in stats_df.columns
    none_row = stats_df[stats_df["Method"] == "None"]
    if not none_row.empty:
        assert np.isclose(none_row["Reduction_pct"].iloc[0], 0.0)
    anova_row = stats_df[stats_df["Method"] == "ANOVA"]
    if not anova_row.empty:
        assert anova_row["Reduction_pct"].iloc[0] > 90.0


def test_jaccard_stability():
    fsets = [
        np.array([0, 1, 2, 3, 4]),
        np.array([0, 1, 2, 5, 6]),
        np.array([0, 1, 7, 8, 9]),
    ]
    mean_j, std_j = compute_jaccard_stability_matrix(fsets)
    assert 0.0 <= mean_j <= 1.0
    assert 0.0 <= std_j <= 1.0


# =========================================================================
# 5. Full Pipeline Integration Test
# =========================================================================

def test_full_pipeline_integration(synthetic_results_df, tmp_path):
    out_dir = str(tmp_path / "test_reports")
    res = run_full_statistical_pipeline(
        synthetic_results_df,
        output_dir=out_dir,
        primary_metric="f1",
        alpha=0.05,
        k_target=20,
    )

    tables = res["tables"]
    assert "tabela1_desempenho" in tables
    assert "tabela2_friedman" in tables
    assert "tabela3_rankings" in tables
    assert "tabela4_nemenyi" in tables
    assert "tabela5_wilcoxon_holm" in tables
    assert "tabela6_eficiencia_reducao" in tables

    for t_name in tables:
        csv_file = os.path.join(out_dir, f"{t_name}.csv")
        assert os.path.exists(csv_file), f"Missing CSV: {csv_file}"

    report_file = res["report_path"]
    assert os.path.exists(report_file)
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "# Relatório de Análise Estatística Experimental" in content
    assert "Teste de Friedman (Omnibus)" in content
    assert "Wilcoxon + Correção de Holm + Tamanho de Efeito" in content
