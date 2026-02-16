"""Configuration file for the breast cancer classification project."""

# Data paths
DATA_PATH = "data/wdbc.data"
MODELS_DIR = "trained_models"

# Feature names
FEATURE_COLS = [
    "id", "diagnosis",
    # Mean features
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    # SE features  
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    # Worst features
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Hyperparameter grids
PARAM_GRIDS = {
    "DecisionTree": {
        "clf__max_depth": [None, 3, 5],
        "clf__min_samples_leaf": [2, 10]
    },
    "RandomForest": {
        "clf__n_estimators": [500, 1000],
        "clf__max_features": ["sqrt", "log2", None],
        "clf__min_samples_leaf": [2, 10],
        "clf__bootstrap": [True, False]
    },
    "ExtraTrees": {
        "clf__n_estimators": [500, 1000],
        "clf__max_features": ["sqrt", "log2", None],
        "clf__min_samples_leaf": [2, 10]
    },
    "GradientBoosting": {
        "clf__n_estimators": [200, 500, 1000],
        "clf__learning_rate": [0.001, 0.01, 0.1],
        "clf__max_depth": [1, 3, 5],
        "clf__min_samples_leaf": [2, 10]
    },
    "XGBoost": {
        "clf__n_estimators": [200, 500, 1000],
        "clf__learning_rate": [0.001, 0.01, 0.1],
        "clf__max_depth": [1, 3, 5],
        "clf__min_child_weight": [2, 10]
    }
}
