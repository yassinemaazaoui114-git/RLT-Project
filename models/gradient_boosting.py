"""Gradient Boosting model pipeline."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from config import PARAM_GRIDS, CV_FOLDS, RANDOM_STATE


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting with GridSearchCV."""
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
    
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRIDS["GradientBoosting"],
        cv=CV_FOLDS,
        scoring=make_scorer(f1_score),
        n_jobs=-1
    )
    
    print("🔄 Training Gradient Boosting...")
    grid.fit(X_train, y_train)
    
    print(f"✓ Best params: {grid.best_params_}")
    print(f"✓ Best CV F1: {grid.best_score_:.4f}")
    
    return grid.best_estimator_
