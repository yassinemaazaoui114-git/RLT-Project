"""
Traditional Reinforcement Learning Trees (RLT) implementation.
Based on Zhu et al. (2015): "Reinforcement Learning Trees"
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from utils.rlt_utils import (
    impurity_reduction, pdf_fraction_from_mode,
    compute_vi_on_node_oob
)


def rlt_root_node(X_train_scaled, y_train, feature_names, M, k, pd_mode,
                  p0, nmin, random_state, task="classification",
                  n_estimators_embedded=100, alpha=0.25):
    """Initialize root node for RLT."""
    X_root = pd.DataFrame(X_train_scaled, columns=feature_names)
    y_root = np.asarray(y_train)
    n, p = X_root.shape
    
    # Compute VI at root
    importance_df = compute_vi_on_node_oob(
        X_root, y_root, task=task,
        n_estimators=n_estimators_embedded,
        random_state=random_state,
    )
    
    # Root protection
    p0 = int(max(0, min(p0 or 0, p)))
    protected_features = importance_df.head(p0)["Feature"].tolist() if p0 > 0 else []
    
    return {
        "importance_df": importance_df,
        "muted_features": [],
        "protected_features": protected_features,
        "remaining_features": feature_names,
        "M": M,
        "k": k,
        "pd_mode": pd_mode,
        "p0": p0,
        "nmin": nmin,
        "n": n,
        "p": p,
        "task": task,
        "n_estimators_embedded": n_estimators_embedded,
        "alpha": alpha,
    }


def rlt_child_node(X, y, feature_names, k, pd_mode, p0, nmin,
                   n_estimators_embedded, random_state, muted_parent,
                   protected_parent, task="classification", alpha=0.25,
                   n_candidates=8):
    """Create child node with muting and splitting."""
    from scipy.stats import pearsonr
    
    rng = np.random.default_rng(random_state)
    X_df = pd.DataFrame(X, columns=feature_names)
    y = np.asarray(y)
    
    if len(y) <= nmin:
        return {"is_leaf": True, "prediction": np.mean(y), "n": len(y)}
    
    muted_parent = list(muted_parent)
    protected_parent = list(dict.fromkeys(protected_parent))
    
    available = [f for f in feature_names if f not in muted_parent or f in protected_parent]
    if len(available) == 0:
        return {"is_leaf": True, "prediction": np.mean(y), "n": len(y)}
    
    # Compute VI
    vi_df = compute_vi_on_node_oob(
        X_df[available], y, task=task,
        n_estimators=n_estimators_embedded,
        random_state=random_state,
    )
    
    # MUTING
    frac = pdf_fraction_from_mode(pd_mode)
    n_mute = int(frac * len(available))
    candidates = vi_df[~vi_df["Feature"].isin(protected_parent)]
    to_mute = candidates.tail(n_mute)["Feature"].tolist()
    muted = list(set(muted_parent + to_mute))
    
    available = [f for f in feature_names if f not in muted or f in protected_parent]
    if len(available) == 0:
        return {"is_leaf": True, "prediction": np.mean(y), "n": len(y)}
    
    vi_df = vi_df[vi_df["Feature"].isin(available)]
    max_vi = vi_df["Importance"].max()
    eligibles = (
        vi_df[vi_df["Importance"] >= alpha * max_vi]["Feature"].tolist()
        if max_vi > 0 else available
    )
    
    split_vars = eligibles[:k] if len(eligibles) >= k else eligibles
    
    # SPLIT SEARCH
    best = None
    best_score = -np.inf
    
    def thresholds(v):
        qs = np.linspace(0.05, 0.95, n_candidates)
        return np.unique(np.quantile(v, qs))
    
    if len(split_vars) == 1:
        v = split_vars[0]
        vals = X_df[v].values
        for t in thresholds(vals):
            mask = vals <= t
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            score = impurity_reduction(y, y[mask], y[~mask], task)
            if score > best_score:
                best_score = score
                best = ("single", v, t)
    else:
        coefs = []
        for v in split_vars:
            if np.std(X_df[v]) == 0:
                sgn = 1.0
            else:
                c = pearsonr(X_df[v], y)[0]
                sgn = np.sign(c) if not np.isnan(c) else 1.0
            vi = vi_df.set_index("Feature").loc[v, "Importance"]
            coefs.append(sgn * np.sqrt(vi))
        
        coefs = np.array(coefs)
        coefs /= (np.linalg.norm(coefs) + 1e-12)
        Z = X_df[split_vars].values @ coefs
        
        for t in thresholds(Z):
            mask = Z <= t
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            score = impurity_reduction(y, y[mask], y[~mask], task)
            if score > best_score:
                best_score = score
                best = ("linear", split_vars, coefs, t)
    
    if best is None:
        return {"is_leaf": True, "prediction": np.mean(y), "n": len(y)}
    
    # APPLY SPLIT
    if best[0] == "single":
        _, v, t = best
        mask = X_df[v].values <= t
        protected = list(dict.fromkeys(protected_parent + [v]))
        info = {"type": "single", "variable": v, "threshold": float(t)}
    else:
        _, vars, coefs, t = best
        Z = X_df[vars].values @ coefs
        mask = Z <= t
        protected = list(dict.fromkeys(protected_parent + vars))
        info = {
            "type": "linear",
            "variables": vars,
            "coefficients": coefs.tolist(),
            "threshold": float(t),
        }
    
    return {
        "is_leaf": False,
        "split_info": info,
        "muted_features": muted,
        "protected_features": protected,
        "X_left": X[mask],
        "y_left": y[mask],
        "X_right": X[~mask],
        "y_right": y[~mask],
    }


def grow_rlt_tree(X, y, feature_names, k, pd_mode, p0, nmin,
                  n_estimators_embedded, random_state, muted_parent,
                  protected_parent, task="classification", alpha=0.25):
    """Recursively grow an RLT tree."""
    node = rlt_child_node(
        X, y, feature_names, k, pd_mode, p0, nmin,
        n_estimators_embedded, random_state, muted_parent,
        protected_parent, task, alpha,
    )
    
    if node["is_leaf"]:
        return node
    
    node["left"] = grow_rlt_tree(
        node["X_left"], node["y_left"], feature_names,
        k, pd_mode, p0, nmin, n_estimators_embedded,
        random_state + 1, node["muted_features"],
        node["protected_features"], task, alpha,
    )
    node["right"] = grow_rlt_tree(
        node["X_right"], node["y_right"], feature_names,
        k, pd_mode, p0, nmin, n_estimators_embedded,
        random_state + 2, node["muted_features"],
        node["protected_features"], task, alpha,
    )
    return node


def fit_rlt_forest(X, y, feature_names, M, k, pd_mode, p0, nmin,
                   n_estimators_embedded, random_state, task="classification",
                   alpha=0.25):
    """Fit RLT forest with M trees."""
    forest = []
    for m in range(M):
        forest.append(
            grow_rlt_tree(
                X, y, feature_names, k, pd_mode, p0, nmin,
                n_estimators_embedded, random_state + 1000 * m,
                [], [], task, alpha,
            )
        )
    return forest


def predict_one_tree(tree, x_row):
    """Predict using a single RLT tree."""
    node = tree
    while not node["is_leaf"]:
        split = node["split_info"]
        
        if split["type"] == "single":
            var = split["variable"]
            threshold = split["threshold"]
            if x_row[var] <= threshold:
                node = node["left"]
            else:
                node = node["right"]
        else:
            vars = split["variables"]
            coefs = np.array(split["coefficients"])
            z = np.dot(x_row[vars], coefs)
            if z <= split["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
    
    return node["prediction"]


def predict_rlt_forest(forest, X_test_df):
    """Predict using RLT forest."""
    preds = np.zeros(len(X_test_df))
    
    for i in range(len(X_test_df)):
        x_row = X_test_df.iloc[i]
        preds[i] = np.mean([predict_one_tree(tree, x_row) for tree in forest])
    
    # Binary classification via regression-style averaging
    return (preds > 0.5).astype(int)


def train_traditional_rlt(X_train, y_train):
    """Train Traditional RLT Forest with BEST hyperparameters from tuning."""
    
    print("\n" + "="*70)
    print(" TRAINING TRADITIONAL RLT FOREST (BEST PARAMS)")
    print("="*70)
    
    # ✅ BEST HYPERPARAMETERS FROM TUNING RESULTS
    forest = fit_rlt_forest(
        X=X_train.values,
        y=y_train.values,
        feature_names=X_train.columns.tolist(),
        M=50,                    # ✅ Best from CV tuning
        k=1,                     # ✅ Best from CV tuning
        pd_mode="none",          # ✅ Best from CV tuning
        p0=3,                    # ✅ Best from CV tuning
        nmin=10,                 # ✅ Best from CV tuning
        n_estimators_embedded=50, # ✅ Best from CV tuning
        random_state=42,
        task="classification",
    )
    
    print("✅ Traditional RLT Forest trained successfully!")
    print("="*70 + "\n")
    
    return forest

