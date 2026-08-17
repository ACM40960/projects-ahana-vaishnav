from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Go from scripts/ back to pipeline/
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "model_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODELS = {
    "Custom CNN": {"history": RESULTS_DIR / "real_data_cnn" / "training_history.csv","metrics": RESULTS_DIR / "real_data_cnn" / "test_metrics.json",},
    "MobileNetV3 Transfer": {"history": RESULTS_DIR / "transfer_learning_cnn" / "training_history.csv","metrics": RESULTS_DIR / "transfer_learning_cnn" / "test_metrics.json",},
    "ViT-B/16 Transfer": {"history": RESULTS_DIR / "vit_transfer_learning" / "training_history.csv","metrics": RESULTS_DIR / "vit_transfer_learning" / "test_metrics.json",},
    "CBAM CNN": {"history": RESULTS_DIR / "cbam_attention_cnn" / "training_history.csv","metrics": RESULTS_DIR / "cbam_attention_cnn" / "test_metrics.json",}
}


def load_history(model_name, history_path):
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file for {model_name}:\n{history_path}")
    df = pd.read_csv(history_path)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(1, len(df) + 1))
    df["model"] = model_name
    return df

def get_metric(data, possible_keys, default=np.nan):
    for key in possible_keys:
        if key in data:
            return data[key]
    return default

def get_confusion_matrix(data):
    if "confusion_matrix" in data:
        return np.array(data["confusion_matrix"])
    if "confusion_matrix_best_threshold" in data:
        return np.array(data["confusion_matrix_best_threshold"])
    return np.array([[np.nan, np.nan], [np.nan, np.nan]])

# Combine training histories
history_frames = []
for model_name, paths in MODELS.items():
    history_df = load_history(model_name, paths["history"])
    history_frames.append(history_df)
all_history = pd.concat(history_frames, ignore_index=True)
combined_history_path = OUTPUT_DIR / "combined_training_history.csv"
all_history.to_csv(combined_history_path, index=False)
print("Saved combined training history:")
print(combined_history_path)

# Plot epoch-wise line charts

def plot_learning_curve(metric_column, title, ylabel, output_name, y_min=None, y_max=None):
    plt.figure(figsize=(12, 6))
    for model_name in MODELS.keys():
        model_df = all_history[all_history["model"] == model_name]
        if metric_column not in model_df.columns:
            print(f"Skipping {model_name}: missing {metric_column}")
            continue
        plt.plot(model_df["epoch"],model_df[metric_column],marker="o",linewidth=2,markersize=4,label=model_name)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    output_path = OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved chart:")
    print(output_path)

plot_learning_curve(metric_column="val_accuracy",title="Validation Accuracy Across Epochs",ylabel="Validation Accuracy",output_name="validation_accuracy_learning_curves.png",y_min=0.85,y_max=1.00)
plot_learning_curve(metric_column="val_roc_auc",title="Validation ROC-AUC Across Epochs",ylabel="Validation ROC-AUC",output_name="validation_roc_auc_learning_curves.png",y_min=0.90,y_max=1.00)
plot_learning_curve(metric_column="val_loss",title="Validation Loss Across Epochs",ylabel="Validation Loss",output_name="validation_loss_learning_curves.png")
# Plot train vs validation accuracy for each model

for model_name in MODELS.keys():
    model_df = all_history[all_history["model"] == model_name]
    if "train_accuracy" not in model_df.columns or "val_accuracy" not in model_df.columns:
        print(f"Skipping train-vs-validation plot for {model_name}")
        continue
    plt.figure(figsize=(10, 5.5))
    plt.plot(model_df["epoch"],model_df["train_accuracy"],marker="o",linewidth=2,markersize=4,label="Train accuracy")
    plt.plot(model_df["epoch"],model_df["val_accuracy"],marker="o",linewidth=2,markersize=4,label="Validation accuracy")
    plt.title(f"{model_name}: Train vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.85, 1.00)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    safe_name = (model_name.lower().replace("/", "_").replace(" ", "_").replace("-", "_"))
    output_path = OUTPUT_DIR / f"{safe_name}_train_vs_val_accuracy.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved chart:")
    print(output_path)


# Read final test metrics
records = []

for model_name, paths in MODELS.items():
    metrics_path = paths["metrics"]
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file for {model_name}:\n{metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    accuracy = get_metric(data,["test_accuracy", "test_accuracy_default_metric"])
    precision = get_metric(data,["test_precision", "test_precision_default_metric"])
    recall = get_metric(data,["test_recall", "test_recall_default_metric"])
    f1_score = get_metric(data,["test_f1"],default=(2 * precision * recall) / (precision + recall))
    roc_auc = get_metric(data,["test_roc_auc", "roc_auc_sklearn"])
    pr_auc = get_metric(data,["test_pr_auc", "pr_auc_sklearn"])
    best_threshold = get_metric(data,["best_threshold"])
    cm = get_confusion_matrix(data)

    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    records.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "precision_lens": precision,
            "recall_lens": recall,
            "f1_lens": f1_score,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_threshold": best_threshold,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        }
    )


metrics_df = pd.DataFrame(records)
metrics_path = OUTPUT_DIR / "final_test_metrics_comparison.csv"
metrics_df.to_csv(metrics_path, index=False)
print("\nFinal test metrics:")
print(metrics_df.to_string(index=False))
print("\nSaved final metrics CSV:")
print(metrics_path)


#Final test metric grouped chart
plot_metrics = ["accuracy","precision_lens","recall_lens","f1_lens"]
x = np.arange(len(metrics_df["model"]))
bar_width = 0.18
plt.figure(figsize=(13, 6))
for i, metric in enumerate(plot_metrics):
    plt.bar(x + (i - 1.5) * bar_width,metrics_df[metric],width=bar_width,label=metric,)
plt.title("Final Test Metrics by Model", fontsize=14, fontweight="bold")
plt.ylabel("Score")
plt.ylim(0.85, 1.00)
plt.xticks(x, metrics_df["model"], rotation=20, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.legend()

for i, metric in enumerate(plot_metrics):
    for j, value in enumerate(metrics_df[metric]):
        plt.text(j + (i - 1.5) * bar_width,value + 0.004,f"{value:.3f}",ha="center",va="bottom",fontsize=8,rotation=90)
plt.tight_layout()
final_metrics_chart_path = OUTPUT_DIR / "final_test_metrics_grouped_bar.png"
plt.savefig(final_metrics_chart_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved chart:")
print(final_metrics_chart_path)


# Error count chart
error_metrics = ["false_positive", "false_negative"]
x = np.arange(len(metrics_df["model"]))
bar_width = 0.30
plt.figure(figsize=(12, 6))
for i, metric in enumerate(error_metrics):
    plt.bar(x + (i - 0.5) * bar_width,metrics_df[metric],width=bar_width,label=metric)
plt.title("False Positives and False Negatives by Model", fontsize=14, fontweight="bold")
plt.ylabel("Number of test images")
plt.xticks(x, metrics_df["model"], rotation=20, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.legend()

for i, metric in enumerate(error_metrics):
    for j, value in enumerate(metrics_df[metric]):
        plt.text(j + (i - 0.5) * bar_width,value + 0.5,str(value),ha="center",va="bottom",fontsize=9,)
plt.tight_layout()
error_chart_path = OUTPUT_DIR / "final_error_counts_bar.png"
plt.savefig(error_chart_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved chart:")
