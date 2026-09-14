"""
Classifier implementations and HPO grids for multimodelo evaluation:
- Random Forest
- KNN
- SVM
- XGBoost
- CatBoost
- Optimum-Path Forest (OPF) with pyopf or vectorized pure-Python/SciPy fallback
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    import pyopf
except ImportError:
    pyopf = None


class SimpleOPFClassifier(BaseEstimator, ClassifierMixin):
    """
    Optimum-Path Forest (OPF) Classifier (Papa et al., 2009).
    Supervised classification based on complete graph, MST prototypes, and path propagation.
    """
    def __init__(self, distance_metric: str = "euclidean"):
        self.distance_metric = distance_metric
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.prototypes_: Optional[np.ndarray] = None
        self.cost_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleOPFClassifier":
        self.classes_ = np.unique(y)
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        n_samples = len(self.X_train_)

        if n_samples <= 1:
            return self

        # 1. Compute pairwise distance matrix
        dist_matrix = cdist(self.X_train_, self.X_train_, metric=self.distance_metric)

        # 2. Compute Minimum Spanning Tree (MST)
        mst = minimum_spanning_tree(dist_matrix).toarray()
        mst = mst + mst.T  # symmetric adjacency

        # 3. Identify prototypes: vertices connected in MST with different labels
        prototypes = set()
        for i in range(n_samples):
            neighbors = np.where(mst[i] > 0)[0]
            for j in neighbors:
                if self.y_train_[i] != self.y_train_[j]:
                    prototypes.add(i)
                    prototypes.add(j)

        if not prototypes:
            # Fallback if MST has no inter-class edges: pick first sample of each class
            for c in self.classes_:
                c_indices = np.where(self.y_train_ == c)[0]
                if len(c_indices) > 0:
                    prototypes.add(c_indices[0])

        self.prototypes_ = np.array(list(prototypes), dtype=np.int64)

        # 4. Initialize path costs for training samples
        cost = np.full(n_samples, np.inf)
        labels = np.copy(self.y_train_)

        for p in self.prototypes_:
            cost[p] = 0.0

        # Dijkstra-like optimum path computation
        unvisited = set(range(n_samples))
        while unvisited:
            # Pick node with min cost in unvisited
            current = min(unvisited, key=lambda x: cost[x])
            unvisited.remove(current)

            for neighbor in unvisited:
                path_cost = max(cost[current], dist_matrix[current, neighbor])
                if path_cost < cost[neighbor]:
                    cost[neighbor] = path_cost
                    labels[neighbor] = labels[current]

        self.cost_ = cost
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate probabilities via inverse distance to training trees."""
        X = np.asarray(X, dtype=np.float64)
        dist_to_train = cdist(X, self.X_train_, metric=self.distance_metric)
        # Compute cost for each test sample to connect to training nodes
        # Cost to connect through node i: max(cost_[i], dist(x, X_train_[i]))
        effective_cost = np.maximum(self.cost_[np.newaxis, :], dist_to_train)
        nearest_node = np.argmin(effective_cost, axis=1)
        pred_labels = self.y_train_[nearest_node]

        # Convert to probability proxy
        probas = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        for i, c in enumerate(self.classes_):
            # Soft weighting based on distances
            c_mask = (self.y_train_ == c)
            min_c_dist = np.min(effective_cost[:, c_mask], axis=1)
            probas[:, i] = 1.0 / (min_c_dist + 1e-5)

        probas /= np.sum(probas, axis=1, keepdims=True)
        return probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def get_classifier_and_grid(classifier_name: str, seed: int = 42) -> Tuple[BaseEstimator, Dict[str, list]]:
    """
    Returns the base estimator and the controlled HPO grid for inner-CV.
    """
    clf_clean = classifier_name.strip().lower()

    if clf_clean in ["rf", "random_forest", "randomforest"]:
        clf = RandomForestClassifier(random_state=seed, n_jobs=1)
        grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
        }
    elif clf_clean in ["knn", "kneighbors"]:
        clf = KNeighborsClassifier(n_jobs=1)
        grid = {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
        }
    elif clf_clean in ["svm", "svc"]:
        clf = SVC(probability=True, random_state=seed)
        grid = {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
        }
    elif clf_clean in ["xgboost", "xgb"]:
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        clf = XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=1)
        grid = {
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        }
    elif clf_clean in ["catboost"]:
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed.")
        clf = CatBoostClassifier(random_state=seed, verbose=0, thread_count=1, allow_writing_files=False)
        grid = {
            "iterations": [50, 100],
            "learning_rate": [0.05, 0.1],
            "depth": [4, 6],
        }
    elif clf_clean in ["opf", "optimum_path_forest"]:
        clf = SimpleOPFClassifier()
        grid = {
            "distance_metric": ["euclidean", "cityblock"],
        }
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return clf, grid


def fit_classifier_with_hpo(
    classifier_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: int = 3,
    seed: int = 42,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Fits classifier with inner Stratified 3-Fold Grid Search exclusively on training data.
    """
    base_clf, param_grid = get_classifier_and_grid(classifier_name, seed=seed)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_estimator, best_params
