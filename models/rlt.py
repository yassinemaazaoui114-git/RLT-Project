"""
Reinforcement Learning Trees (RLT) for classification.

Implementation of the method proposed in:
    Zhu, R., Zeng, D. & Kosorok, M. R. (2015).
    "Reinforcement Learning Trees". Journal of the American Statistical
    Association, 110(512), 1770-1784. https://doi.org/10.1080/01621459.2015.1036994

At every internal node of every tree:
  1. Embedded model -- an Extremely Randomized Trees model is fitted on the
     node's data and an out-of-bag permutation variable importance (VI) is
     computed for every variable that has not been muted.
  2. Split variable selection -- the node is split on the variable with the
     largest VI ("looking ahead" instead of a marginal, greedy search).
  3. Linear combination split -- optionally, up to `combination_size`
     variables with VI >= alpha * max(VI) are combined as
     z = sum_j sign(rho_j) * sqrt(VI_j) * x_j and the node is split on z.
  4. Variable muting -- a fraction `muting_rate` of the remaining, unprotected
     variables with the lowest VI are muted: they are never considered again
     in the descendant nodes. Variables used for splitting along the path are
     protected and can never be muted.

Trees are grown on bootstrap samples and their leaf class distributions are
averaged (`predict_proba`), as in a random forest.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils.validation import check_is_fitted

MUTING_MODES = {"none": 0.0, "moderate": 0.5, "aggressive": 0.8}


class _Node:
    """Node of an RLT tree. Leaves store the class distribution of their samples."""

    __slots__ = ("features", "coefs", "threshold", "left", "right", "value")

    def __init__(self, value=None):
        self.features = None   # indices of the split variable(s)
        self.coefs = None      # linear combination coefficients (1.0 for a single variable)
        self.threshold = None
        self.left = None
        self.right = None
        self.value = value     # class proportions (leaves only)

    @property
    def is_leaf(self):
        return self.features is None


def _gini(counts):
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return 1.0 - np.sum(p ** 2)


def _best_cut(z, y, n_classes, min_samples_leaf):
    """Exhaustive search of the cut point on z that maximises the Gini reduction."""
    order = np.argsort(z, kind="mergesort")
    zs, ys = z[order], y[order]
    n = len(zs)
    onehot = np.eye(n_classes)[ys]
    left = np.cumsum(onehot, axis=0)[:-1]
    total = left[-1] + onehot[-1]
    right = total - left
    n_left = np.arange(1, n)[:, None]
    n_right = n - n_left
    gini_left = 1.0 - np.sum((left / n_left) ** 2, axis=1)
    gini_right = 1.0 - np.sum((right / n_right) ** 2, axis=1)
    weighted = (n_left[:, 0] * gini_left + n_right[:, 0] * gini_right) / n

    valid = (zs[1:] > zs[:-1]) & (n_left[:, 0] >= min_samples_leaf) & (n_right[:, 0] >= min_samples_leaf)
    if not valid.any():
        return None, 0.0
    weighted[~valid] = np.inf
    i = int(np.argmin(weighted))
    return (zs[i] + zs[i + 1]) / 2.0, _gini(total) - weighted[i]


def _embedded_importance(X, y, n_classes, n_estimators, max_features, rng):
    """
    Embedded model of RLT: Extremely Randomized Trees fitted on the node data.

    Returns the out-of-bag permutation importance of each column of X (increase
    of the OOB Brier score when the column is permuted). If no variable has a
    positive permutation importance (e.g. very small nodes), the impurity-based
    importance of the same embedded model is returned instead.
    """
    n, p = X.shape
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    onehot = np.eye(n_classes)[y]
    perm_vi = np.zeros(p)
    mdi_vi = np.zeros(p)
    n_used = 0

    for _ in range(n_estimators):
        boot = rng.integers(0, n, size=n)
        oob = np.setdiff1d(np.arange(n), boot, assume_unique=False)
        tree = ExtraTreeClassifier(max_features=max_features,
                                   random_state=int(rng.integers(2**31 - 1)))
        tree.fit(X32[boot], y[boot])
        mdi_vi += tree.feature_importances_
        if len(oob) < 2:
            continue

        cols = tree.classes_.astype(int)

        def brier(Xo):
            proba = np.zeros((len(Xo), n_classes))
            # low-level Tree.predict skips input validation (much faster on tiny nodes)
            values = tree.tree_.predict(Xo).reshape(len(Xo), -1)[:, :len(cols)]
            proba[:, cols] = values / values.sum(axis=1, keepdims=True)
            return np.mean(np.sum((proba - onehot[oob]) ** 2, axis=1))

        X_oob = X32[oob]
        base = brier(X_oob)
        for j in range(p):
            X_perm = X_oob.copy()
            X_perm[:, j] = X_oob[rng.permutation(len(oob)), j]
            perm_vi[j] += brier(X_perm) - base
        n_used += 1

    if n_used > 0 and perm_vi.max() > 0:
        return np.maximum(perm_vi / n_used, 0.0)
    return mdi_vi / n_estimators


def _grow_tree(X, y, params, seed):
    """Grow one RLT tree on a bootstrap sample. Returns (root, split importances)."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    if params["bootstrap"]:
        idx = rng.integers(0, n, size=n)
        X, y = X[idx], y[idx]
    importances = np.zeros(p)
    root = _grow_node(X, y, np.arange(p), [], 0, params, rng, importances)
    return root, importances


def _grow_node(X, y, active, protected, depth, params, rng, importances):
    n_classes = params["n_classes"]
    counts = np.bincount(y, minlength=n_classes).astype(float)
    leaf = _Node(value=counts / counts.sum())
    if (len(y) < 2 * params["min_samples_leaf"] or len(y) < params["min_samples_split"]
            or depth >= params["max_depth"] or counts.max() == len(y)):
        return leaf

    # 1. Embedded model: variable importance of the non-muted variables
    vi = _embedded_importance(X[:, active], y, n_classes, params["embedded_n_estimators"],
                              params["embedded_max_features"], rng)
    order = np.argsort(-vi, kind="mergesort")

    # 2./3. Split variable(s): the best one, or a linear combination of the top ones
    k = params["combination_size"]
    top = [i for i in order[:k] if vi[i] >= params["alpha"] * vi[order[0]]] or [order[0]]
    features = active[top]
    if len(features) == 1:
        coefs = np.array([1.0])
    else:
        signs = []
        for f in features:
            x = X[:, f]
            rho = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 0.0
            signs.append(np.sign(rho) if np.isfinite(rho) and rho != 0 else 1.0)
        coefs = np.array(signs) * np.sqrt(vi[top])
        coefs /= np.linalg.norm(coefs) + 1e-12

    z = X[:, features] @ coefs
    threshold, gain = _best_cut(z, y, n_classes, params["min_samples_leaf"])
    if threshold is None or gain <= 0:
        return leaf

    # 4. Variable muting: drop the least important unprotected variables
    protected = list(dict.fromkeys(protected + features.tolist()))
    n_unprotected = sum(1 for f in active if f not in protected)
    n_mute = min(int(params["muting_rate"] * n_unprotected),
                 max(0, len(active) - params["min_features"]))
    muted = [active[i] for i in order[::-1] if active[i] not in protected][:n_mute]
    child_active = np.array([f for f in active if f not in muted])

    importances[features] += gain * len(y) * np.abs(coefs)

    node = _Node()
    node.features, node.coefs, node.threshold = features, coefs, threshold
    mask = z <= threshold
    node.left = _grow_node(X[mask], y[mask], child_active, protected, depth + 1, params, rng, importances)
    node.right = _grow_node(X[~mask], y[~mask], child_active, protected, depth + 1, params, rng, importances)
    return node


def _predict_tree(node, X, out, rows):
    """Route the rows of X down the tree and write leaf distributions into out."""
    if node.is_leaf:
        out[rows] += node.value
        return
    go_left = X[rows][:, node.features] @ node.coefs <= node.threshold
    _predict_tree(node.left, X, out, rows[go_left])
    _predict_tree(node.right, X, out, rows[~go_left])


def _count_leaves(node):
    return 1 if node.is_leaf else _count_leaves(node.left) + _count_leaves(node.right)


class RLTClassifier(ClassifierMixin, BaseEstimator):
    """
    Reinforcement Learning Trees forest classifier (scikit-learn compatible).

    Parameters
    ----------
    n_estimators : int, default=100
        Number of RLT trees (M in the paper).
    muting_rate : float or {"none", "moderate", "aggressive"}, default=0.5
        Fraction of the remaining unprotected variables muted at each node
        (p_d in the paper; "moderate" = 0.5, "aggressive" = 0.8).
    min_features : int, default=3
        Muting never reduces the number of active variables below this value.
    combination_size : int, default=1
        Maximum number of variables in a linear combination split (k in the
        paper). 1 means single-variable splits.
    alpha : float, default=0.25
        A variable enters the linear combination only if its VI is at least
        alpha * max(VI).
    embedded_n_estimators : int, default=25
        Number of extremely randomized trees in each node's embedded model.
    embedded_max_features : int, float or "sqrt", default="sqrt"
        Candidate variables per split inside the embedded trees.
    min_samples_leaf : int, default=3
        Minimum number of samples in a terminal node.
    min_samples_split : int, default=10
        Nodes with fewer samples are not split (n_min in the paper).
    max_depth : int, default=20
        Maximum depth of each tree.
    bootstrap : bool, default=True
        Grow each tree on a bootstrap sample of the training data.
    n_jobs : int, default=-1
        Number of trees grown in parallel.
    random_state : int or None, default=None
    """

    def __init__(self, n_estimators=100, muting_rate=0.5, min_features=3, combination_size=1,
                 alpha=0.25, embedded_n_estimators=25, embedded_max_features="sqrt",
                 min_samples_leaf=3, min_samples_split=10, max_depth=20, bootstrap=True,
                 n_jobs=-1, random_state=None):
        self.n_estimators = n_estimators
        self.muting_rate = muting_rate
        self.min_features = min_features
        self.combination_size = combination_size
        self.alpha = alpha
        self.embedded_n_estimators = embedded_n_estimators
        self.embedded_max_features = embedded_max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        X, y = self._check_input(X, y, reset=True)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        muting_rate = MUTING_MODES.get(self.muting_rate, self.muting_rate)
        params = {
            "n_classes": len(self.classes_),
            "muting_rate": float(muting_rate),
            "min_features": max(1, self.min_features),
            "combination_size": max(1, self.combination_size),
            "alpha": self.alpha,
            "embedded_n_estimators": self.embedded_n_estimators,
            "embedded_max_features": self.embedded_max_features,
            "min_samples_leaf": max(1, self.min_samples_leaf),
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
            "bootstrap": self.bootstrap,
        }
        seeds = np.random.default_rng(self.random_state).integers(2**31 - 1, size=self.n_estimators)
        grown = Parallel(n_jobs=self.n_jobs)(
            delayed(_grow_tree)(X, y_enc, params, int(s)) for s in seeds
        )
        self.trees_ = [tree for tree, _ in grown]
        importances = np.sum([imp for _, imp in grown], axis=0)
        self.feature_importances_ = importances / importances.sum() if importances.sum() > 0 else importances
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "trees_")
        X, _ = self._check_input(X, reset=False)
        out = np.zeros((X.shape[0], len(self.classes_)))
        rows = np.arange(X.shape[0])
        for tree in self.trees_:
            _predict_tree(tree, X, out, rows)
        return out / len(self.trees_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _check_input(self, X, y=None, reset=True):
        """Convert inputs to float arrays and record/check the feature names."""
        if hasattr(X, "columns"):
            names = np.asarray(X.columns, dtype=object)
            if reset:
                self.feature_names_in_ = names
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {X.shape}")
        if reset:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but RLTClassifier was fitted with "
                             f"{self.n_features_in_} features")
        return X, (None if y is None else np.asarray(y).ravel())

    @property
    def n_leaves_(self):
        check_is_fitted(self, "trees_")
        return [_count_leaves(tree) for tree in self.trees_]
