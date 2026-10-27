"""Figures for the model benchmark (saved as PNG files, used in the README)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

HIGHLIGHT = "#2a78d6"   # RLT
SECOND = "#eb6834"      # Random Forest (best baseline reference)
MUTED = "#b4b2a9"       # other models
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _color(name):
    if "RLT" in name and "Neural" not in name:
        return HIGHLIGHT
    return SECOND if name == "RandomForest" else MUTED


def plot_metric_comparison(results_df, out_dir, metrics=("Accuracy", "F1-Score", "ROC-AUC")):
    """One horizontal bar panel per metric, models sorted by value, RLT highlighted."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 0.45 * len(results_df) + 1.4))
    for ax, metric in zip(axes, metrics):
        df = results_df.sort_values(metric)
        lo = max(0.0, np.floor((df[metric].min() - 0.03) * 20) / 20)
        ax.barh(df["Model"], df[metric], color=[_color(m) for m in df["Model"]], height=0.6)
        for y, v in enumerate(df[metric]):
            ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=9, color=TEXT)
        ax.set_xlim(lo, 1.02)
        ax.set_title(metric, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Test-set performance (114 samples, stratified 80/20 split)",
                 x=0.01, ha="left", color=TEXT_SECONDARY, fontsize=10)
    fig.tight_layout()
    path = Path(out_dir) / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curves(probas, y_test, out_dir):
    """ROC curves of all models; RLT and Random Forest emphasised."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    order = sorted(probas, key=lambda m: _color(m) != MUTED)  # draw emphasised curves last
    others = [m for m in order if _color(m) == MUTED]
    for name in order:
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        emphasised = _color(name) != MUTED
        label = name if emphasised else (f"Other models ({len(others)})" if name == others[0] else None)
        ax.plot(fpr, tpr, color=_color(name), linewidth=2 if emphasised else 1.2,
                alpha=1 if emphasised else 0.8, label=label, zorder=3 if emphasised else 2)
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)", loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    path = Path(out_dir) / "roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrices(predictions, y_test, out_dir):
    """Grid of confusion matrices (rows: true class, columns: predicted class)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    names = list(predictions)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.9 * rows))
    for ax in np.ravel(axes)[len(names):]:
        ax.axis("off")
    for ax, name in zip(np.ravel(axes), names):
        cm = confusion_matrix(y_test, predictions[name])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=len(y_test) * 0.65)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=12,
                    color="white" if v > len(y_test) * 0.4 else TEXT)
        ax.set_xticks([0, 1], ["Benign", "Malignant"])
        ax.set_yticks([0, 1], ["Benign", "Malignant"])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(name, loc="left", fontsize=11, color=HIGHLIGHT if _color(name) == HIGHLIGHT else TEXT)
    fig.supxlabel("Predicted", color=TEXT_SECONDARY, fontsize=10)
    fig.supylabel("True", color=TEXT_SECONDARY, fontsize=10)
    fig.tight_layout()
    path = Path(out_dir) / "confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rlt_feature_importance(rlt_model, feature_names, out_dir, top_n=15):
    """Share of the total Gini reduction attributed to each variable by the RLT splits."""
    imp = np.asarray(rlt_model.feature_importances_)
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * top_n + 1.2))
    ax.barh(np.asarray(feature_names)[idx], imp[idx], color=HIGHLIGHT, height=0.6)
    ax.set_title(f"RLT variable importance (top {top_n})", loc="left")
    ax.set_xlabel("Share of total split impurity reduction")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    path = Path(out_dir) / "rlt_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_visualizations(results_df, predictions, probas, y_test, rlt_model, feature_names, out_dir):
    """Create every benchmark figure in out_dir and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_metric_comparison(results_df, out_dir),
        plot_roc_curves(probas, y_test, out_dir),
        plot_confusion_matrices(predictions, y_test, out_dir),
        plot_rlt_feature_importance(rlt_model, feature_names, out_dir),
    ]
