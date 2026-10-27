"""Baseline models compared against RLT."""
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from config import CV_FOLDS, PARAM_GRIDS, RANDOM_STATE


def get_baseline_models():
    """Baseline classifiers with the fixed hyperparameters used for the benchmark and the API."""
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    }


def tune_baseline(name, X_train, y_train):
    """Optional: tune one baseline with GridSearchCV over config.PARAM_GRIDS (F1, 5-fold CV)."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", get_baseline_models()[name])])
    grid = GridSearchCV(pipeline, PARAM_GRIDS[name], cv=CV_FOLDS, scoring=make_scorer(f1_score), n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"✓ {name} best params: {grid.best_params_} (CV F1 = {grid.best_score_:.4f})")
    return grid.best_estimator_
