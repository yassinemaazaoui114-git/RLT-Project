"""
Breast Cancer Classification - Full Benchmark Pipeline
Trains the baselines, the RLT forest and our Neural RLT (not part of the paper) on the same
split, prints the test metrics, and saves the results table and figures to docs/.

Usage:
    python main.py              # all models
    python main.py --no-neural  # skip the TensorFlow Neural RLT
"""
import argparse
import time
import warnings
import pandas as pd
from config import BASE_DIR
from models.baselines import get_baseline_models
from models.rlt import RLTClassifier
from train_and_save import NEURAL_RLT_PARAMS, RLT_PARAMS
from utils.data_loader import load_split
from utils.evaluation import evaluate_model
from utils.visualizations import generate_all_visualizations

warnings.filterwarnings("ignore")
OUT_DIR = BASE_DIR / "docs"


def build_models(include_neural=True):
    models = get_baseline_models()
    models["RLT"] = RLTClassifier(**RLT_PARAMS)
    if include_neural:
        from models.neural_rlt import NeuralRLTForest
        models["NeuralRLT"] = NeuralRLTForest(**NEURAL_RLT_PARAMS)
    return models


def main(include_neural=True):
    print("\n" + "=" * 70)
    print("🏥 BREAST CANCER ML - COMPLETE TRAINING PIPELINE")
    print("=" * 70)

    X_train, X_test, y_train, y_test, _ = load_split()
    print(f"✓ Train set: {len(X_train)} samples | Test set: {len(X_test)} samples | "
          f"{X_train.shape[1]} features")

    results, predictions, probas = [], {}, {}
    for name, model in build_models(include_neural).items():
        print(f"\n🤖 Training {name}...")
        start = time.time()
        model.fit(X_train.values, y_train.values)
        train_time = time.time() - start
        predictions[name] = model.predict(X_test.values)
        probas[name] = model.predict_proba(X_test.values)[:, 1]
        metrics = evaluate_model(name, y_test, predictions[name], probas[name])
        metrics["Train time (s)"] = round(train_time, 2)
        results.append(metrics)
        if name == "RLT":
            rlt_model = model

    results_df = pd.DataFrame(results).sort_values(["Accuracy", "ROC-AUC"], ascending=False)

    print("\n" + "=" * 70)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 70)
    for path in generate_all_visualizations(results_df, predictions, probas, y_test, rlt_model,
                                            X_train.columns.tolist(), OUT_DIR / "figures"):
        print(f"✓ {path.relative_to(BASE_DIR)}")

    results_df.to_csv(OUT_DIR / "benchmark_results.csv", index=False, float_format="%.4f")

    print("\n" + "=" * 70)
    print(" FINAL RESULTS SUMMARY")
    print("=" * 70 + "\n")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n✓ Best model: {results_df.iloc[0]['Model']}")
    print(f"✓ Results saved to {(OUT_DIR / 'benchmark_results.csv').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-neural", action="store_true", help="skip the TensorFlow Neural RLT")
    main(include_neural=not parser.parse_args().no_neural)
