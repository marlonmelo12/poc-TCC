"""
Unified Statistical Analysis Facade Module.
Integrates validation, Friedman, Nemenyi, Wilcoxon + Holm, effect size,
dimensionality reduction trade-offs, and automated reporting.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd

from .validation import validate_results_dataframe, check_paired_completeness
from .friedman_nemenyi import (
    build_paired_matrix,
    compute_method_ranks,
    run_friedman_test,
    run_nemenyi_test,
)
from .wilcoxon_holm import run_planned_wilcoxon_tests, compute_rank_biserial_correlation
from .tradeoff import compute_reduction_statistics, compute_qubo_energy_correlation, compute_jaccard_stability_matrix
from .reporting import (
    generate_table1_descriptive,
    generate_all_statistical_tables,
    generate_scientific_plots,
    generate_markdown_report,
)


def run_full_statistical_pipeline(
    df: pd.DataFrame,
    output_dir: str = "results/reports",
    primary_metric: str = "f1",
    alpha: float = 0.05,
    k_target: int = 20,
) -> Dict[str, Any]:
    """
    Executes complete end-to-end layered statistical analysis:
    1. Integrity check & validation.
    2. Descriptive metrics (Table 1).
    3. Friedman omnibus tests (Table 2).
    4. Method rankings (Table 3).
    5. Nemenyi post-hoc tests (Table 4).
    6. Wilcoxon + Holm-Bonferroni + Effect Size (Table 5).
    7. Dimensionality reduction & efficiency (Table 6).
    8. Publication-ready scientific plots.
    9. Technical markdown report (statistical_report.md).
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Validation
    df_clean, audit_report = validate_results_dataframe(df)

    # 2. Generate Tables 1 to 6
    tables = generate_all_statistical_tables(
        df_clean,
        primary_metric=primary_metric,
        alpha=alpha,
        k_target=k_target,
    )

    # Save Tables as CSV
    for t_name, t_df in tables.items():
        if not t_df.empty:
            csv_path = os.path.join(output_dir, f"{t_name}.csv")
            t_df.to_csv(csv_path, index=False)

    # 3. Generate Scientific Plots
    plot_paths = generate_scientific_plots(tables, df_clean, output_dir, primary_metric=primary_metric)

    # 4. Generate Markdown Report
    report_path = os.path.join(output_dir, "statistical_report.md")
    report_text = generate_markdown_report(tables, plot_paths, audit_report, report_path, primary_metric=primary_metric)

    return {
        "tables": tables,
        "audit_report": audit_report,
        "plot_paths": plot_paths,
        "report_path": report_path,
    }


# Backwards compatibility wrappers
def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Computes descriptive summary table for backward compatibility."""
    df_clean, _ = validate_results_dataframe(df)
    return generate_table1_descriptive(df_clean)


def run_paired_wilcoxon_tests(
    df: pd.DataFrame,
    baseline_method: str = "None",
    qubo_method: str = "QUBO-SA",
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Backward compatible paired Wilcoxon runner."""
    df_clean, _ = validate_results_dataframe(df)
    custom_comp = [(qubo_method, baseline_method)]
    return run_planned_wilcoxon_tests(
        df_clean,
        dataset=df_clean["dataset"].iloc[0],
        metric=metric,
        custom_comparisons=custom_comp,
    )
