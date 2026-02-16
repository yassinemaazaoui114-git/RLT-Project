"""Model evaluation utilities."""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(name, y_true, y_pred, y_proba):
    """Evaluate model performance."""
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }
    
    print(f"\n{'='*60}")
    print(f"  {name} Performance")
    print(f"{'='*60}")
    for key, value in list(metrics.items())[1:]:
        print(f"  {key:12s}: {value:.4f}")
    
    return metrics
