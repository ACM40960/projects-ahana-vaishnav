from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# This script is expected to live in: Real Data pipeline/scripts/
BASE_FOLDER = Path(__file__).resolve().parent.parent
RESULTS_FOLDER = BASE_FOLDER / "results"
OUTPUT_FOLDER = RESULTS_FOLDER / "model_comparison"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# Keep only the four models you want to compare.
MODEL_FILES = {
    "Custom CNN": RESULTS_FOLDER / "real_data_cnn" / "test_metrics.json",
    "MobileNetV3 Transfer": RESULTS_FOLDER / "transfer_learning_cnn" / "test_metrics.json",
    "ViT-B/16 Transfer": RESULTS_FOLDER / "vit_transfer_learning" / "test_metrics.json",
    "CBAM CNN": RESULTS_FOLDER / "cbam_attention_cnn" / "test_metrics.json",
}


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def first_available(metrics: dict, possible_keys: list[str]):
    """Different scripts saved metric names slightly differently, so check all possible keys."""
    for key in possible_keys:
        if key in metrics:
            return metrics[key]
    return None


def get_confusion_matrix(metrics: dict):
    matrix = first_available(
        metrics,
        ["confusion_matrix", "confusion_matrix_best_threshold"],
    )

    if matrix is None:
        return None

    return np.array(matrix)


def extract_model_row(model_name: str, metrics_path: Path) -> dict:
    metrics = read_json(metrics_path)

    accuracy = first_available(
        metrics,
        ["test_accuracy", "test_accuracy_default_metric"],
    )
    precision = first_available(
        metrics,
        ["test_precision", "test_precision_default_metric"],
    )
    recall = first_available(
        metrics,
        ["test_recall", "test_recall_default_metric"],
    )
    f1_score = first_available(
        metrics,
        ["test_f1", "test_f1_default_metric"],
    )

    # Some of your JSON files do not store F1 directly, so calculate it from precision and recall.
    if f1_score is None and precision is not None and recall is not None:
        f1_score = 2 * precision * recall / (precision + recall)

    roc_auc = first_available(
        metrics,
        ["test_roc_auc", "roc_auc_sklearn"],
    )
    pr_auc = first_available(
        metrics,
        ["test_pr_auc", "pr_auc_sklearn"],
    )
    threshold = first_available(
        metrics,
        ["best_threshold"],
    )
    val_f1 = first_available(
        metrics,
        ["best_validation_f1"],
    )

    matrix = get_confusion_matrix(metrics)

    if matrix is not None:
        tn, fp = matrix[0]
        fn, tp = matrix[1]
    else:
        tn = fp = fn = tp = None

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision_lens": precision,
        "recall_lens": recall,
        "f1_lens": f1_score,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": threshold,
        "validation_f1": val_f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "metrics_file": str(metrics_path),
    }


def add_value_labels(ax, bars, as_percent=True):
    for bar in bars:
        width = bar.get_width()
        label = f"{width:.1f}%" if as_percent else f"{width:.0f}"
        ax.text(
            width + 0.4,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )


def save_grouped_metric_chart(df: pd.DataFrame):
    """One grouped bar chart for accuracy, precision, recall, and F1."""

    plot_df = df.copy()
    metric_columns = ["accuracy", "precision_lens", "recall_lens", "f1_lens"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    values = plot_df[metric_columns].to_numpy() * 100
    x = np.arange(len(plot_df["model"]))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, label in enumerate(metric_labels):
        ax.bar(
            x + (i - 1.5) * width,
            values[:, i],
            width,
            label=label,
        )

    ax.set_title("Model comparison: test-set classification metrics", fontsize=15, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=15, ha="right")
    ax.set_ylim(85, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28))

    plt.tight_layout()
    output_path = OUTPUT_FOLDER / "model_metrics_grouped_bar.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_auc_chart(df: pd.DataFrame):
    """Separate chart for ROC-AUC and PR-AUC."""

    plot_df = df.copy()
    metric_columns = ["roc_auc", "pr_auc"]
    metric_labels = ["ROC-AUC", "PR-AUC"]

    values = plot_df[metric_columns].to_numpy() * 100
    x = np.arange(len(plot_df["model"]))
    width = 0.32

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, label in enumerate(metric_labels):
        ax.bar(
            x + (i - 0.5) * width,
            values[:, i],
            width,
            label=label,
        )

    ax.set_title("Model comparison: ranking metrics", fontsize=15, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=15, ha="right")
    ax.set_ylim(95, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    plt.tight_layout()
    output_path = OUTPUT_FOLDER / "auc_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_error_count_chart(df: pd.DataFrame):
    """False positives and false negatives are important for lens search, so show them directly."""

    plot_df = df.copy()
    x = np.arange(len(plot_df["model"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    bars_fp = ax.bar(
        x - width / 2,
        plot_df["false_positive"],
        width,
        label="False positives",
    )
    bars_fn = ax.bar(
        x + width / 2,
        plot_df["false_negative"],
        width,
        label="False negatives",
    )

    ax.set_title("Model comparison: error counts on the test set", fontsize=15, fontweight="bold")
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    add_value_labels(ax, bars_fp, as_percent=False)
    add_value_labels(ax, bars_fn, as_percent=False)

    plt.tight_layout()
    output_path = OUTPUT_FOLDER / "error_counts.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    rows = []

    for model_name, metrics_path in MODEL_FILES.items():
        rows.append(extract_model_row(model_name, metrics_path))

    df = pd.DataFrame(rows)

    # Keep this order in the table and graphs.
    df["model"] = pd.Categorical(
        df["model"],
        categories=list(MODEL_FILES.keys()),
        ordered=True,
    )
    df = df.sort_values("model").reset_index(drop=True)

    summary_path = OUTPUT_FOLDER / "model_comparison_summary.csv"
    df.to_csv(summary_path, index=False)

    metric_chart_path = save_grouped_metric_chart(df)
    auc_chart_path = save_auc_chart(df)
    error_chart_path = save_error_count_chart(df)

    display_columns = [
        "model",
        "accuracy",
        "precision_lens",
        "recall_lens",
        "f1_lens",
        "roc_auc",
        "pr_auc",
        "best_threshold",
        "false_positive",
        "false_negative",
    ]

    print("\nModel comparison summary")
    print(df[display_columns].to_string(index=False))

    print("\nSaved files:")
    print(f"CSV: {summary_path}")
    print(f"Classification metrics chart: {metric_chart_path}")
    print(f"AUC chart: {auc_chart_path}")
    print(f"Error counts chart: {error_chart_path}")


if __name__ == "__main__":
    main()