"""Helper functions for RLT implementations."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def variance_impurity(y):
    """Calculate variance impurity for regression."""
    return np.var(y, ddof=0) if len(y) > 0 else 0.0


def gini_impurity(y):
    """Calculate Gini impurity for classification."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y.astype(int), return_counts=True)
    ps = counts / counts.sum()
    return 1.0 - np.sum(ps ** 2)


def impurity(y, task="classification"):
    """Calculate impurity based on task type."""
    return variance_impurity(y) if task == "regression" else gini_impurity(y)


def impurity_reduction(y_parent, y_left, y_right, task="classification"):
    """Calculate impurity reduction from a split."""
    n = len(y_parent)
    if n == 0:
        return 0.0
    return impurity(y_parent, task) - (
        len(y_left) / n * impurity(y_left, task) +
        len(y_right) / n * impurity(y_right, task)
    )


def pdf_fraction_from_mode(pd_mode):
    """Get muting fraction based on mode."""
    return {"none": 0.0, "moderate": 0.5, "aggressive": 0.8}[pd_mode]


def compute_vi_on_node_oob(X_df, y, task="classification", n_estimators=100,
                            random_state=None, min_samples_leaf=1, max_depth=None):
    """
    Compute variable importance using OOB predictions.
    Implements the embedded model approach from RLT paper.
    """
    from sklearn.tree import ExtraTreeRegressor, ExtraTreeClassifier
    
    rng = np.random.default_rng(random_state)
    X = X_df.to_numpy()
    y = np.asarray(y)
    
    n, p = X.shape
    all_idx = np.arange(n)
    
    vi = np.zeros(p, dtype=float)
    counts = np.zeros(p, dtype=float)
    
    for _ in range(n_estimators):
        # Bootstrap sample
        boot = rng.integers(0, n, size=n)
        boot_unique = np.unique(boot)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[boot_unique] = False
        oob = all_idx[oob_mask]
        
        if oob.size == 0:
            continue
        
        seed = int(rng.integers(0, 2**31 - 1))
        if task == "regression":
            tree = ExtraTreeRegressor(
                splitter="random",
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=seed,
            )
        else:
            tree = ExtraTreeClassifier(
                splitter="random",
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=seed,
            )
        
        tree.fit(X[boot], y[boot])
        
        # Baseline prediction
        base_pred = tree.predict(X[oob])
        if task == "regression":
            base_err = np.mean((y[oob] - base_pred) ** 2)
        else:
            base_err = np.mean(y[oob] != base_pred)
        
        # Permutation importance
        for j in range(p):
            X_p = X[oob].copy()
            rng.shuffle(X_p[:, j])
            pred = tree.predict(X_p)
            if task == "regression":
                err = np.mean((y[oob] - pred) ** 2)
            else:
                err = np.mean(y[oob] != pred)
            
            vi[j] += (err - base_err)
            counts[j] += 1
    
    vi = np.maximum(0.0, vi / np.maximum(counts, 1))
    return pd.DataFrame({"Feature": X_df.columns, "Importance": vi}) \
        .sort_values("Importance", ascending=False) \
        .reset_index(drop=True)
