"""XGBoost model pipeline."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from config import PARAM_GRIDS, CV_FOLDS, RANDOM_STATE


def train_xgboost(X_train, y_train):
    """Train XGBoost with GridSearchCV."""
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(eval_metric='logloss', 
                             n_jobs=-1, random_state=RANDOM_STATE))
    ])
    
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRIDS["XGBoost"],
        cv=CV_FOLDS,
        scoring=make_scorer(f1_score),
        n_jobs=-1
    )
    
    print("🔄 Training XGBoost...")
    grid.fit(X_train, y_train)
    
    print(f"✓ Best params: {grid.best_params_}")
    print(f"✓ Best CV F1: {grid.best_score_:.4f}")
    
    return grid.best_estimator_
