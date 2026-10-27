# ============================================
# train_and_save.py
# Train the API models once and save them to trained_models/
# ============================================
import json
import time
import joblib
from config import MODELS_DIR
from models.baselines import get_baseline_models
from models.rlt import RLTClassifier
from utils.data_loader import load_split
from utils.evaluation import evaluate_model

# Hyperparameters of the RLT forest served by the API (chosen by 5-fold CV on the training set)
RLT_PARAMS = dict(
    n_estimators=100,
    muting_rate=0.5,
    combination_size=2,
    embedded_n_estimators=25,
    min_samples_leaf=3,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
)

# Neural RLT: our own addition (not in the paper) -- bagged neural networks on random feature subsets
NEURAL_RLT_PARAMS = dict(
    n_estimators=10,
    max_features="sqrt",
    hidden_layers=(64, 32, 16),
    dropout_rate=0.3,
    learning_rate=0.001,
    epochs=50,
    batch_size=32,
    bootstrap=True,
    random_state=42,
)


def main():
    from models.neural_rlt import NeuralRLTForest  # imports TensorFlow, so only when training

    print("🚀 Starting training process...")

    # 1. Data: stratified 80/20 split, standardised features, 1 = Malignant
    X_train, X_test, y_train, y_test, scaler = load_split()
    print(f"✅ Data ready: {len(X_train)} training samples, {len(X_test)} test samples")

    # 2. Train all models (5 baselines + RLT + our Neural RLT)
    models = get_baseline_models()
    models["RLT"] = RLTClassifier(**RLT_PARAMS)
    models["NeuralRLT"] = NeuralRLTForest(**NEURAL_RLT_PARAMS)

    metrics = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train.values, y_train.values)
        train_time = time.time() - start
        y_pred = model.predict(X_test.values)
        y_proba = model.predict_proba(X_test.values)[:, 1]
        metrics[name] = evaluate_model(name, y_test, y_pred, y_proba, verbose=False)
        metrics[name]["Train time (s)"] = round(train_time, 2)
        print(f"  {name:<17} accuracy={metrics[name]['Accuracy']:.4f}  "
              f"ROC-AUC={metrics[name]['ROC-AUC']:.4f}  ({train_time:.1f}s)")

    # 3. Save models, scaler, feature names and test metrics
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{name.lower()}.joblib")
    joblib.dump({"feature_names": X_train.columns.tolist(), "n_features": X_train.shape[1]},
                MODELS_DIR / "feature_info.joblib")
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"split": "stratified, test_size=0.2, random_state=42", "n_test": len(y_test),
                   "rlt_params": RLT_PARAMS, "neural_rlt_params": NEURAL_RLT_PARAMS, "metrics": metrics}, f, indent=2)

    print(f"\n💾 Saved {len(models)} models, scaler, feature info and metrics.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
