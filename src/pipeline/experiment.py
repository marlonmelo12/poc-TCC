"""
Experiment runner module: Orchestrates Outer Stratified 10-Fold CV, Feature Selection,
Inner 3-Fold HPO, Model Training, and Outer Test Evaluation with strict zero data leakage.
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from ..data.loader import load_dataset
from ..qubo.metrics_mrmr import compute_mrmr_matrices
from ..selectors.none_selector import NoneFeatureSelector
from ..selectors.anova import ANOVAFeatureSelector
from ..selectors.rfecv_selector import RFECVFeatureSelector
from ..selectors.lasso import LassoFeatureSelector
from ..selectors.qubo_selector import QUBOFeatureSelector
from ..models.classifiers import fit_classifier_with_hpo
from .evaluation import evaluate_predictions
from .checkpoint import CheckpointManager


DEFAULT_K_GRID = [10, 20, 50, 100]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_CLASSIFIERS = ["rf", "knn", "svm", "xgboost", "catboost", "opf"]


def run_experiment_fold(
    dataset_name: str,
    fold_idx: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    methods: List[str],
    classifiers: List[str],
    k_grid: List[int],
    seeds: List[int],
    checkpoint_mgr: CheckpointManager,
) -> None:
    """
    Executes one outer fold across all methods and classifiers without test leakage.
    """
    n_features_orig = X_train.shape[1]

    # 1. Normalization: fit scaler strictly on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Precompute mRMR matrices once per fold on X_train_scaled for efficiency
    needs_qubo = any("qubo" in m.lower() for m in methods)
    precomputed_mrmr = None
    if needs_qubo:
        print(f"[{dataset_name} | Fold {fold_idx+1}] Precomputing mRMR matrices on TRAIN...")
        precomputed_mrmr = compute_mrmr_matrices(X_train_scaled, y_train, random_state=42)

    # 2. Loop over methods
    for method in methods:
        method_clean = method.strip().upper()

        if method_clean == "NONE":
            selector = NoneFeatureSelector()
            selector.fit(X_train_scaled, y_train)
            _evaluate_selector_on_classifiers(
                dataset_name, fold_idx, "None", selector, None, 42,
                X_train_scaled, y_train, X_test_scaled, y_test,
                classifiers, checkpoint_mgr
            )

        elif method_clean == "ANOVA":
            # Test with default percentile or for each K in k_grid
            for K in k_grid:
                selector = ANOVAFeatureSelector(k=K)
                selector.fit(X_train_scaled, y_train)
                _evaluate_selector_on_classifiers(
                    dataset_name, fold_idx, "ANOVA", selector, K, 42,
                    X_train_scaled, y_train, X_test_scaled, y_test,
                    classifiers, checkpoint_mgr
                )

        elif method_clean == "RFECV":
            selector = RFECVFeatureSelector(n_estimators=50, step=0.10, cv_splits=3, random_state=42)
            selector.fit(X_train_scaled, y_train)
            _evaluate_selector_on_classifiers(
                dataset_name, fold_idx, "RFECV", selector, selector.n_selected, 42,
                X_train_scaled, y_train, X_test_scaled, y_test,
                classifiers, checkpoint_mgr
            )

        elif method_clean == "LASSO":
            selector = LassoFeatureSelector(cv_splits=3, random_state=42)
            selector.fit(X_train_scaled, y_train)
            _evaluate_selector_on_classifiers(
                dataset_name, fold_idx, "Lasso", selector, selector.n_selected, 42,
                X_train_scaled, y_train, X_test_scaled, y_test,
                classifiers, checkpoint_mgr
            )

        elif method_clean in ["QUBO-SA", "QUBO_SA"]:
            for K in k_grid:
                for seed in seeds:
                    selector = QUBOFeatureSelector(
                        solver_name="QUBO-SA",
                        K=K,
                        seed=seed,
                        precomputed_mrmr=precomputed_mrmr,
                    )
                    selector.fit(X_train_scaled, y_train)
                    _evaluate_selector_on_classifiers(
                        dataset_name, fold_idx, "QUBO-SA", selector, K, seed,
                        X_train_scaled, y_train, X_test_scaled, y_test,
                        classifiers, checkpoint_mgr
                    )

        elif method_clean in ["QUBO-SB", "QUBO_SB"]:
            for K in k_grid:
                for seed in seeds:
                    selector = QUBOFeatureSelector(
                        solver_name="QUBO-SB",
                        K=K,
                        seed=seed,
                        precomputed_mrmr=precomputed_mrmr,
                    )
                    try:
                        selector.fit(X_train_scaled, y_train)
                        _evaluate_selector_on_classifiers(
                            dataset_name, fold_idx, "QUBO-SB", selector, K, seed,
                            X_train_scaled, y_train, X_test_scaled, y_test,
                            classifiers, checkpoint_mgr
                        )
                    except Exception as e:
                        print(f"QUBO-SB failed for K={K}, seed={seed}: {e}")


def _evaluate_selector_on_classifiers(
    dataset_name: str,
    fold_idx: int,
    method_name: str,
    selector: Any,
    K: Optional[int],
    seed: int,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    classifiers: List[str],
    checkpoint_mgr: CheckpointManager,
) -> None:
    mask = selector.get_support()
    n_features = int(np.sum(mask))
    reduction_rate = float(selector.reduction_rate)
    runtime_selection = float(getattr(selector, "runtime_", 0.0))
    qubo_energy = getattr(selector, "qubo_energy_", None)

    X_train_sub = X_train_scaled[:, mask]
    X_test_sub = X_test_scaled[:, mask]

    for clf_name in classifiers:
        if checkpoint_mgr.is_completed(dataset_name, fold_idx, method_name, clf_name, seed, K):
            continue

        start_train = time.perf_counter()
        try:
            model, best_params = fit_classifier_with_hpo(
                classifier_name=clf_name,
                X_train=X_train_sub,
                y_train=y_train,
                cv_splits=3,
                seed=42,
            )
            runtime_training = time.perf_counter() - start_train

            # Predict on outer test
            y_pred = model.predict(X_test_sub)
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test_sub)
                except Exception:
                    y_prob = None

            metrics = evaluate_predictions(y_test, y_pred, y_prob)

            solver_info = getattr(selector, "solver_info_", None) or {}
            raw_k = solver_info.get("raw_k", None)
            card_native = solver_info.get("cardinality_satisfied_natively", None)

            record = {
                "dataset": dataset_name,
                "fold": fold_idx,
                "method": method_name,
                "classifier": clf_name,
                "seed": seed,
                "K": K,
                "n_features": n_features,
                "reduction_rate": reduction_rate,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
                "runtime_selection": runtime_selection,
                "runtime_training": runtime_training,
                "qubo_energy": qubo_energy,
                "raw_k": raw_k,
                "cardinality_satisfied_natively": card_native,
                "best_params": best_params,
                "status": "SUCCESS",
            }
            checkpoint_mgr.save_record(record)
            print(
                f"[{dataset_name} | Fold {fold_idx+1}] {method_name} (K={K}) + {clf_name.upper()} "
                f"-> Acc: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}, Red: {reduction_rate*100:.1f}%"
            )
        except Exception as e:
            print(f"Error fitting {clf_name} for {method_name}: {e}")
            record = {
                "dataset": dataset_name,
                "fold": fold_idx,
                "method": method_name,
                "classifier": clf_name,
                "seed": seed,
                "K": K,
                "n_features": n_features,
                "reduction_rate": reduction_rate,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "auc": None,
                "runtime_selection": runtime_selection,
                "runtime_training": None,
                "qubo_energy": qubo_energy,
                "best_params": None,
                "status": f"ERROR: {e}",
            }
            checkpoint_mgr.save_record(record)


def run_full_dataset_experiment(
    dataset_name: str,
    methods: Optional[List[str]] = None,
    classifiers: Optional[List[str]] = None,
    k_grid: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    n_splits: int = 10,
    checkpoint_filepath: Optional[str] = None,
) -> None:
    """
    Runs full 10-fold CV for a dataset across all specified methods and classifiers.
    """
    if methods is None:
        methods = ["None", "ANOVA", "RFECV", "Lasso", "QUBO-SA"]
    if classifiers is None:
        classifiers = DEFAULT_CLASSIFIERS
    if k_grid is None:
        k_grid = DEFAULT_K_GRID
    if seeds is None:
        seeds = [42]

    print(f"\n=======================================================")
    print(f"Starting experiment for dataset: {dataset_name.upper()}")
    print(f"=======================================================")

    X, y, feature_names = load_dataset(dataset_name)
    print(f"Loaded {dataset_name}: Shape={X.shape}, Classes={np.unique(y, return_counts=True)}")

    checkpoint_mgr = CheckpointManager(checkpoint_filepath) if checkpoint_filepath else CheckpointManager()

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n--- FOLD {fold_idx + 1} / {n_splits} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        run_experiment_fold(
            dataset_name=dataset_name,
            fold_idx=fold_idx,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            methods=methods,
            classifiers=classifiers,
            k_grid=k_grid,
            seeds=seeds,
            checkpoint_mgr=checkpoint_mgr,
        )
