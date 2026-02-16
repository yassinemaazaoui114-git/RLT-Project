"""
Reinforcement Learning Trees (RLT) Implementation
Based on Zhu et al. (2015) paper
"""

import numpy as np
import pandas as pd
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor


def variance_impurity(y):
    """Calculate variance for regression"""
    return np.var(y, ddof=0) if len(y) > 0 else 0.0


def gini_impurity(y):
    """Calculate Gini impurity for classification"""
    if len(y) == 0:
        return 0.0
    values, counts = np.unique(y.astype(int), return_counts=True)
    ps = counts / counts.sum()
    return 1.0 - np.sum(ps ** 2)


def impurity(y, task='regression'):
    """Calculate impurity based on task"""
    return variance_impurity(y) if task == 'regression' else gini_impurity(y)


def impurity_reduction(y_parent, y_left, y_right, task='regression'):
    """Calculate impurity reduction from a split"""
    n = len(y_parent)
    if n == 0:
        return 0.0
    return impurity(y_parent, task) - (
        (len(y_left) / n) * impurity(y_left, task) +
        (len(y_right) / n) * impurity(y_right, task)
    )


def pd_fraction_from_mode(pdmode):
    """Get fraction of variables to mute based on mode"""
    return {'none': 0.0, 'moderate': 0.5, 'aggressive': 0.8}.get(pdmode, 0.0)


def compute_vi_on_node_oob(X_df, y, task='regression', n_estimators=100, random_state=None, min_samples_leaf=1, max_depth=None):
    """
    Compute variable importance using OOB samples
    """
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
        if task == 'regression':
            tree = ExtraTreeRegressor(
                splitter='random',
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=seed,
            )
        else:
            tree = ExtraTreeClassifier(
                splitter='random',
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=seed,
            )
        
        tree.fit(X[boot], y[boot])
        
        # Base prediction on OOB
        base_pred = tree.predict(X[oob])
        if task == 'regression':
            base_err = np.mean((y[oob] - base_pred) ** 2)
        else:
            base_err = np.mean(y[oob] != base_pred)
        
        # Permutation importance
        for j in range(p):
            X_p = X[oob].copy()
            rng.shuffle(X_p[:, j])  # permute feature j on OOB
            pred = tree.predict(X_p)
            if task == 'regression':
                err = np.mean((y[oob] - pred) ** 2)
            else:
                err = np.mean(y[oob] != pred)
            
            vi[j] += (err - base_err)
            counts[j] += 1
    
    vi /= np.maximum(counts, 1)
    return (
        pd.DataFrame({'Feature': X_df.columns, 'Importance': vi})
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )


def rlt_root_node(X_train_scaled, y_train, feature_names, M, k, pdmode, p0, nmin, 
                  random_state, task='regression', n_estimators_embedded=100, alpha=0.25):
    """Initialize root node with variable importance"""
    X_root = pd.DataFrame(X_train_scaled, columns=feature_names)
    y_root = np.asarray(y_train)
    n, p = X_root.shape
    
    # VI at root (no repeats, no embedded model returned)
    importance_df = compute_vi_on_node_oob(
        X_root,
        y_root,
        task=task,
        n_estimators=n_estimators_embedded,
        random_state=random_state,
    )
    
    # Optional root protection (can be empty)
    p0 = int(max(0, min(p0 or 0, p)))
    protected_features = importance_df.head(p0)['Feature'].tolist() if p0 > 0 else []
    
    # NO muting at root (recommended)
    muted_features = []
    
    return {
        'importance_df': importance_df,
        'muted_features': muted_features,
        'protected_features': protected_features,
        'remaining_features': feature_names,
        'M': M,
        'k': k,
        'pdmode': pdmode,
        'p0': p0,
        'nmin': nmin,
        'n': n,
        'p': p,
        'task': task,
        'n_estimators_embedded': n_estimators_embedded,
        'alpha': alpha,
    }


def rlt_child_node(X, y, feature_names, k, pdmode, p0, nmin, n_estimators_embedded,
                   random_state, muted_parent, protected_parent, task='regression',
                   alpha=0.25, n_candidates=8):
    """Build a child node with RLT logic"""
    rng = np.random.default_rng(random_state)
    X_df = pd.DataFrame(X, columns=feature_names)
    y = np.asarray(y)
    
    if len(y) <= nmin:
        return {'is_leaf': True, 'prediction': np.mean(y), 'n': len(y)}
    
    muted_parent = list(muted_parent)
    protected_parent = list(dict.fromkeys(protected_parent))
    
    available = [f for f in feature_names if f not in muted_parent or f in protected_parent]
    if len(available) == 0:
        return {'is_leaf': True, 'prediction': np.mean(y), 'n': len(y)}
    
    # Compute VI on available features
    vi_df = compute_vi_on_node_oob(
        X_df[available],
        y,
        task=task,
        n_estimators=n_estimators_embedded,
        random_state=random_state,
    )
    
    # ------------------ MUTING ------------------
    frac = pd_fraction_from_mode(pdmode)
    n_mute = int(frac * len(available))
    candidates = vi_df[~vi_df['Feature'].isin(protected_parent)]
    to_mute = candidates.tail(n_mute)['Feature'].tolist()
    muted = list(set(muted_parent + to_mute))
    
    available = [f for f in feature_names if f not in muted or f in protected_parent]
    if len(available) == 0:
        return {'is_leaf': True, 'prediction': np.mean(y), 'n': len(y)}
    
    vi_df = vi_df[vi_df['Feature'].isin(available)]
    max_vi = vi_df['Importance'].max()
    eligibles = (
        vi_df[vi_df['Importance'] >= alpha * max_vi]['Feature'].tolist()
        if max_vi > 0
        else available
    )
    
    split_vars = eligibles[:k] if len(eligibles) >= k else eligibles
    
    # ------------------ SPLIT SEARCH ------------------
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
                best = ('single', v, t)
    else:
        from scipy.stats import pearsonr
        coefs = []
        for v in split_vars:
            if np.std(X_df[v]) == 0:
                sgn = 1.0
            else:
                c = pearsonr(X_df[v], y)[0]
                sgn = np.sign(c) if not np.isnan(c) else 1.0
            vi = vi_df.set_index('Feature').loc[v, 'Importance']
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
                best = ('linear', split_vars, coefs, t)
    
    if best is None:
        return {'is_leaf': True, 'prediction': np.mean(y), 'n': len(y)}
    
    # ------------------ APPLY SPLIT ------------------
    if best[0] == 'single':
        _, v, t = best
        mask = X_df[v].values <= t
        protected = list(dict.fromkeys(protected_parent + [v]))
        info = {'type': 'single', 'variable': v, 'threshold': float(t)}
    else:
        _, vars, coefs, t = best
        Z = X_df[vars].values @ coefs
        mask = Z <= t
        protected = list(dict.fromkeys(protected_parent + vars))
        info = {
            'type': 'linear',
            'variables': vars,
            'coefficients': coefs.tolist(),
            'threshold': float(t),
        }
    
    return {
        'is_leaf': False,
        'split_info': info,
        'muted_features': muted,
        'protected_features': protected,
        'X_left': X[mask],
        'y_left': y[mask],
        'X_right': X[~mask],
        'y_right': y[~mask],
    }


def grow_rlt_tree(X, y, feature_names, k, pdmode, p0, nmin, n_estimators_embedded,
                  random_state, muted_parent, protected_parent, task='regression', alpha=0.25):
    """Recursively grow an RLT tree"""
    node = rlt_child_node(
        X, y, feature_names, k, pdmode, p0, nmin, n_estimators_embedded,
        random_state, muted_parent, protected_parent, task, alpha,
    )
    
    if node['is_leaf']:
        return node
    
    node['left'] = grow_rlt_tree(
        node['X_left'], node['y_left'], feature_names,
        k, pdmode, p0, nmin, n_estimators_embedded,
        random_state + 1, node['muted_features'], node['protected_features'], task, alpha,
    )
    
    node['right'] = grow_rlt_tree(
        node['X_right'], node['y_right'], feature_names,
        k, pdmode, p0, nmin, n_estimators_embedded,
        random_state + 2, node['muted_features'], node['protected_features'], task, alpha,
    )
    
    return node


def fit_rlt_forest(X, y, feature_names, M, k, pdmode, p0, nmin, n_estimators_embedded,
                   random_state, task='regression', alpha=0.25):
    """Train an RLT forest"""
    forest = []
    for m in range(M):
        forest.append(
            grow_rlt_tree(
                X, y, feature_names,
                k, pdmode, p0, nmin, n_estimators_embedded,
                random_state + 1000 + m, [], [],
                task, alpha,
            )
        )
    return forest


def predict_one_tree(tree, xrow):
    """Predict using one tree"""
    node = tree
    while not node['is_leaf']:
        split = node['split_info']
        
        if split['type'] == 'single':
            var = split['variable']
            threshold = split['threshold']
            if xrow[var] <= threshold:
                node = node['left']
            else:
                node = node['right']
        else:
            vars = split['variables']
            coefs = np.array(split['coefficients'])
            z = np.dot(xrow[vars], coefs)
            if z <= split['threshold']:
                node = node['left']
            else:
                node = node['right']
    
    return node['prediction']


def predict_rlt_forest(forest, X_test_df):
    """Predict using RLT forest (returns class labels)"""
    preds = np.zeros(len(X_test_df))
    
    for i in range(len(X_test_df)):
        xrow = X_test_df.iloc[i]
        preds[i] = np.mean([predict_one_tree(tree, xrow) for tree in forest])
    
    # Binary classification via regression-style averaging
    return (preds >= 0.5).astype(int)


def predict_rlt_forest_scores(forest, X_test_df):
    """Predict using RLT forest (returns probability scores)"""
    scores = np.zeros(len(X_test_df), dtype=float)
    
    for i in range(len(X_test_df)):
        xrow = X_test_df.iloc[i]
        scores[i] = np.mean([predict_one_tree(tree, xrow) for tree in forest])
    
    return scores
