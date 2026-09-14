"""
Analysis and statistical evaluation package.
"""

from .validation import validate_results_dataframe, check_paired_completeness
from .friedman_nemenyi import build_paired_matrix, compute_method_ranks, run_friedman_test, run_nemenyi_test
from .wilcoxon_holm import run_planned_wilcoxon_tests, compute_rank_biserial_correlation
from .tradeoff import compute_reduction_statistics, compute_qubo_energy_correlation, compute_jaccard_stability_matrix
from .reporting import generate_all_statistical_tables, generate_scientific_plots, generate_markdown_report
from .stats import run_full_statistical_pipeline, compute_summary_table, run_paired_wilcoxon_tests

__all__ = [
    "validate_results_dataframe",
    "check_paired_completeness",
    "build_paired_matrix",
    "compute_method_ranks",
    "run_friedman_test",
    "run_nemenyi_test",
    "run_planned_wilcoxon_tests",
    "compute_rank_biserial_correlation",
    "compute_reduction_statistics",
    "compute_qubo_energy_correlation",
    "compute_jaccard_stability_matrix",
    "generate_all_statistical_tables",
    "generate_scientific_plots",
    "generate_markdown_report",
    "run_full_statistical_pipeline",
    "compute_summary_table",
    "run_paired_wilcoxon_tests",
]
