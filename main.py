"""
Main entry point for experiments, testing, data loading, and statistical analysis.
Usage:
    python main.py --mode download
    python main.py --mode test
    python main.py --mode pilot --dataset colon
    python main.py --mode full --dataset colon
    python main.py --mode full --dataset prostate
    python main.py --mode full --dataset ovarian
    python main.py --mode stats
"""

import sys
import argparse
import pytest
import numpy as np

from src.data.loader import load_colon, load_ovarian, load_prostate
from src.pipeline.experiment import run_full_dataset_experiment
from src.pipeline.checkpoint import CheckpointManager
from src.analysis.stats import run_full_statistical_pipeline, compute_summary_table, run_paired_wilcoxon_tests


def handle_download():
    print("=== Downloading and verifying datasets ===")
    print("\n1. Loading Colon Cancer dataset...")
    X_c, y_c, f_c = load_colon()
    print(f"   Colon loaded: {X_c.shape}, classes={dict(zip(*np.unique(y_c, return_counts=True)))}")

    print("\n2. Loading Prostate Tumor dataset...")
    X_p, y_p, f_p = load_prostate()
    print(f"   Prostate loaded: {X_p.shape}, classes={dict(zip(*np.unique(y_p, return_counts=True)))}")

    print("\n3. Loading Ovarian Cancer dataset...")
    X_o, y_o, f_o = load_ovarian()
    print(f"   Ovarian loaded: {X_o.shape}, classes={dict(zip(*np.unique(y_o, return_counts=True)))}")

    print("\nAll datasets downloaded and validated successfully!")


def handle_test():
    print("=== Running unit and validation tests ===")
    ret = pytest.main(["-v", "tests"])
    sys.exit(ret)


def handle_pilot(dataset: str):
    print(f"=== Running controlled PILOT experiment on {dataset.upper()} ===")
    # 1 fold, fast budget
    run_full_dataset_experiment(
        dataset_name=dataset,
        methods=["None", "ANOVA", "RFECV", "Lasso", "QUBO-SA"],
        classifiers=["rf", "knn", "svm"],
        k_grid=[10, 20],
        seeds=[42],
        n_splits=2,  # 2 folds for quick pilot
    )
    print("\nPilot completed successfully!")


def handle_full(dataset: str, methods: list, classifiers: list, k_grid: list, seeds: list, n_splits: int = 10, checkpoint_filepath: str = None):
    datasets_to_run = ["colon", "prostate", "ovarian"] if dataset == "all" else [dataset]

    for ds in datasets_to_run:
        run_full_dataset_experiment(
            dataset_name=ds,
            methods=methods,
            classifiers=classifiers,
            k_grid=k_grid,
            seeds=seeds,
            n_splits=n_splits,
            checkpoint_filepath=checkpoint_filepath,
        )


def handle_stats(checkpoint_filepath: str = None, output_dir: str = "results/reports", primary_metric: str = "f1", k_target: int = 20):
    print("=== Aggregating results and running comprehensive statistical analysis ===")
    mgr = CheckpointManager(checkpoint_filepath) if checkpoint_filepath else CheckpointManager()
    df = mgr.load_all_results()

    if df.empty:
        print("No results found in checkpoints yet.")
        return

    print(f"Loaded {len(df)} experiment records.")
    pipeline_res = run_full_statistical_pipeline(
        df,
        output_dir=output_dir,
        primary_metric=primary_metric,
        alpha=0.05,
        k_target=k_target,
    )

    print(f"\nStatistical analysis completed successfully!")
    print(f"Reports and Tables saved in: {output_dir}")
    print(f"Markdown technical report: {pipeline_res['report_path']}")


def main():
    parser = argparse.ArgumentParser(description="TCC Feature Selection Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["download", "test", "pilot", "full", "stats"], help="Operation mode")
    parser.add_argument("--dataset", type=str, default="colon", choices=["colon", "prostate", "ovarian", "all"], help="Dataset name")
    parser.add_argument("--methods", type=str, default="None,ANOVA,RFECV,Lasso,QUBO-SA,QUBO-SB", help="Comma-separated methods")
    parser.add_argument("--classifiers", type=str, default="rf,knn,svm,xgboost,catboost,opf", help="Comma-separated classifiers")
    parser.add_argument("--k_grid", type=str, default="10,20,50,100", help="Comma-separated K values")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of outer folds (e.g. 5 or 10)")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Custom checkpoint file path")
    parser.add_argument("--primary_metric", type=str, default="f1", help="Primary metric for statistical tests (default: f1)")
    parser.add_argument("--output_dir", type=str, default="results/reports", help="Output directory for reports and plots")
    parser.add_argument("--k_target", type=int, default=20, help="Target K for fixed-K comparison across methods (default: 20)")

    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    k_grid = [int(k.strip()) for k in args.k_grid.split(",") if k.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.mode == "download":
        handle_download()
    elif args.mode == "test":
        handle_test()
    elif args.mode == "pilot":
        handle_pilot(args.dataset)
    elif args.mode == "full":
        handle_full(args.dataset, methods, classifiers, k_grid, seeds, n_splits=args.n_splits, checkpoint_filepath=args.checkpoint_file)
    elif args.mode == "stats":
        handle_stats(
            checkpoint_filepath=args.checkpoint_file,
            output_dir=args.output_dir,
            primary_metric=args.primary_metric,
            k_target=args.k_target,
        )


if __name__ == "__main__":
    main()
