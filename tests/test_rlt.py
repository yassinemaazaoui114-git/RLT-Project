"""Unit tests of the RLT classifier."""
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from models.rlt import RLTClassifier


def _data(n_informative=3, n_features=20):
    return make_classification(n_samples=300, n_features=n_features, n_informative=n_informative,
                               n_redundant=0, shuffle=False, random_state=0)


def test_fit_predict_shapes_and_probabilities():
    X, y = _data()
    model = RLTClassifier(n_estimators=10, random_state=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(model.predict(X)) == {0, 1}
    assert model.score(X, y) > 0.9


def test_reproducible_with_random_state():
    X, y = _data()
    a = RLTClassifier(n_estimators=5, random_state=1).fit(X, y).predict_proba(X)
    b = RLTClassifier(n_estimators=5, random_state=1).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_importance_concentrates_on_informative_variables():
    # with shuffle=False the informative variables are the first columns
    X, y = _data(n_informative=3, n_features=30)
    model = RLTClassifier(n_estimators=20, muting_rate="moderate", random_state=0).fit(X, y)
    assert model.feature_importances_[:3].sum() > 0.5


def test_linear_combination_splits_and_string_labels():
    X, y = _data()
    labels = np.where(y == 1, "M", "B")
    model = RLTClassifier(n_estimators=5, combination_size=3, random_state=0).fit(X, labels)
    assert set(model.classes_) == {"B", "M"}
    assert any(len(t.features) > 1 for t in model.trees_ if not t.is_leaf)


def test_sklearn_clone():
    model = clone(RLTClassifier(n_estimators=3, muting_rate=0.8))
    assert model.get_params()["muting_rate"] == 0.8
