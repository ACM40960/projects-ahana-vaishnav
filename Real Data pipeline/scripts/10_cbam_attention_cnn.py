"""
10_cbam_attention_cnn.py

Same base CNN architecture as 04_train_real_data_cnn.py, with CBAM
(Convolutional Block Attention Module) inserted after each conv block.
CBAM learns two lightweight attention maps per block — "which channels
matter" (channel attention) and "which spatial locations matter"
(spatial attention) — and rescales the feature maps accordingly before
passing them on.

This is a genuinely different mechanism from plain Grad-CAM explainability:
Grad-CAM explains an already-trained, unmodified CNN after the fact:
CBAM's attention is *learned during training* and actively shapes what
the network computes, not just what we visualise afterward. The spatial
attention maps saved by this script are what the model itself chose to
attend to, not a post-hoc gradient explanation.

Same splits, class weighting, thresholding, and output format as
04_train_real_data_cnn.py for a direct, fair comparison.

Run this AFTER 03_quality_filter.py.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T


# --------------------------------------------------
# 1. Paths (relative — same pattern as the other scripts)
# --------------------------------------------------

BASE_FOLDER = Path(__file__).resolve().parent.parent

METADATA_PATH = BASE_FOLDER / "data" / "metadata" / "quality_metadata.csv"
RESULTS_DIR = BASE_FOLDER / "results" / "cbam_attention_cnn"
MODELS_DIR = BASE_FOLDER / "models" / "cbam_attention_cnn"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# --------------------------------------------------
# 2. Load metadata (identical filtering to 04_train_real_data_cnn.py)
# --------------------------------------------------

metadata = pd.read_csv(METADATA_PATH)

trainable = metadata[
    (metadata["usable"] == 1)
    & (metadata["class_name"].isin(["lens", "presumed_non_lens"]))
].copy()

print(f"Trainable images: {len(trainable)}")


def load_image_array(file_path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    with Image.open(file_path) as image:
        image = image.convert("RGB").resize((size, size))
        return np.asarray(image, dtype="float32")


images, labels, rows = [], [], []
error_count = 0
for _, row in trainable.iterrows():
    try:
        images.append(load_image_array(row["file_path"]))
        labels.append(int(row["label"]))
        rows.append(row)
    except Exception as error:
        error_count += 1
        if error_count <= 5:
            print(f"  FAILED to load {row['image_id']}: {row['file_path']}")
            print(f"    Error: {error}")

print(f"\nSuccessfully loaded: {len(images)} / {len(trainable)}  (failed: {error_count})")

if len(images) == 0:
    raise RuntimeError(
        "No images loaded successfully — check the file paths printed above. "
        "This usually means quality_metadata.csv has stale file paths from "
        "before a folder rename/move. Rerun 03_quality_filter.py to regenerate it."
    )

X = np.stack(images, axis=0)
y = np.array(labels, dtype="int64")
rows_df = pd.DataFrame(rows).reset_index(drop=True)


# --------------------------------------------------
# 3. Same train/val/test split as 04_train_real_data_cnn.py
# --------------------------------------------------

train_idx, temp_idx = train_test_split(
    np.arange(len(X)), test_size=0.30, random_state=RANDOM_SEED, stratify=y
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=y[temp_idx]
)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]
test_rows = rows_df.iloc[test_idx].reset_index(drop=True)

print("\nSplit sizes")
print("Train:", X_train.shape, np.unique(y_train, return_counts=True))
print("Val:  ", X_val.shape, np.unique(y_val, return_counts=True))
print("Test: ", X_test.shape, np.unique(y_test, return_counts=True))


# --------------------------------------------------
# 4. Normalise using training set only
# --------------------------------------------------

mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6


# --------------------------------------------------
# 5. Dataset / DataLoader (same augmentation as 04)
# --------------------------------------------------

class LensDataset(Dataset):
    def __init__(self, X_arr, y_arr, mean, std, augment: bool):
        self.X = X_arr
        self.y = y_arr
        self.mean = torch.tensor(mean.reshape(3, 1, 1), dtype=torch.float32)
        self.std = torch.tensor(std.reshape(3, 1, 1), dtype=torch.float32)
        self.augment = augment
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(180),
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.X[idx]).permute(2, 0, 1).float()
        if self.augment:
            image = self.aug_transform(image)
        image = (image - self.mean) / self.std
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return image, label


train_dataset = LensDataset(X_train, y_train, mean, std, augment=True)
val_dataset = LensDataset(X_val, y_val, mean, std, augment=False)
test_dataset = LensDataset(X_test, y_test, mean, std, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --------------------------------------------------
# 6. CBAM module — channel attention then spatial attention
# --------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        attention = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attention[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.amax(dim=1, keepdim=True)
        attention_map = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * attention_map, attention_map


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x, spatial_map = self.spatial_attention(x)
        return x, spatial_map


# --------------------------------------------------
# 7. Model — same conv architecture as the plain CNN
# (16 -> 32 -> 64 channels), with a CBAM block after each
# convolution. Returns the final spatial attention map too,
# for visualisation.
# --------------------------------------------------

class CBAMAttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
        self.cbam1 = CBAM(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.cbam2 = CBAM(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.cbam3 = CBAM(64)

        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x, return_attention: bool = False):
        x = self.relu(self.conv1(x))
        x, _ = self.cbam1(x)
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x, _ = self.cbam2(x)
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x, spatial_map = self.cbam3(x)

        pooled = self.gap(x).flatten(1)
        logits = self.classifier(pooled).squeeze(1)

        if return_attention:
            return logits, spatial_map
        return logits


model = CBAMAttentionCNN().to(DEVICE)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")


# --------------------------------------------------
# 8. Class weighting, optimiser, scheduler
# --------------------------------------------------

n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
pos_weight_value = n_neg / max(n_pos, 1)
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
)


# --------------------------------------------------
# 9. Training loop (identical logic to 04)
# --------------------------------------------------

def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    probs = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)

    return {
        "loss": avg_loss,
        "accuracy": float((preds == labels_arr).mean()),
        "roc_auc": float(roc_auc_score(labels_arr, probs)) if len(np.unique(labels_arr)) > 1 else float("nan"),
    }


history = {f"{split}_{name}": [] for split in ["train", "val"] for name in ["loss", "accuracy", "roc_auc"]}

best_val_loss = float("inf")
epochs_without_improvement = 0
best_model_path = MODELS_DIR / "best_cbam_attention_cnn.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    train_metrics = run_epoch(train_loader, train=True)
    val_metrics = run_epoch(val_loader, train=False)
    scheduler.step(val_metrics["loss"])

    for name in ["loss", "accuracy", "roc_auc"]:
        history[f"train_{name}"].append(train_metrics[name])
        history[f"val_{name}"].append(val_metrics[name])

    print(
        f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
        f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
        f"val_acc={val_metrics['accuracy']:.4f} val_roc_auc={val_metrics['roc_auc']:.4f}"
    )

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

model.load_state_dict(torch.load(best_model_path))
print(f"\nRestored best model (val_loss={best_val_loss:.4f}).")


# --------------------------------------------------
# 10. Threshold tuning + test evaluation
# --------------------------------------------------

model.eval()
val_probs, val_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        logits = model(images.to(DEVICE))
        val_probs.append(torch.sigmoid(logits).cpu().numpy())
        val_labels.append(labels.numpy())
val_probs, val_labels = np.concatenate(val_probs), np.concatenate(val_labels)

thresholds = np.linspace(0.01, 0.99, 99)
best_threshold, best_f1 = 0.5, 0.0
for t in thresholds:
    f1 = f1_score(val_labels, (val_probs >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

print("\nBest threshold:", best_threshold, "| Best val F1:", best_f1)

test_probs, test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images.to(DEVICE))
        test_probs.append(torch.sigmoid(logits).cpu().numpy())
        test_labels.append(labels.numpy())
test_probs, test_labels = np.concatenate(test_probs), np.concatenate(test_labels)
test_preds = (test_probs >= best_threshold).astype(int)

cm = confusion_matrix(test_labels, test_preds)
report = classification_report(test_labels, test_preds, target_names=["non_lens", "lens"], zero_division=0)
roc_auc = roc_auc_score(test_labels, test_probs)
pr_auc = average_precision_score(test_labels, test_probs)

print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", report)
print("ROC-AUC:", roc_auc, "| PR-AUC:", pr_auc)


# --------------------------------------------------
# 11. Save model, metrics, plots
# --------------------------------------------------

torch.save(model.state_dict(), MODELS_DIR / "final_cbam_attention_cnn.pt")

metrics = {
    "model": "CBAM-augmented CNN (channel + spatial attention, trained from scratch)",
    "total_params": total_params,
    "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
    "best_threshold": float(best_threshold),
    "best_validation_f1": float(best_f1),
    "test_accuracy_default_metric": float((test_preds == test_labels).mean()),
    "test_precision_default_metric": float(precision_score(test_labels, test_preds, zero_division=0)),
    "test_recall_default_metric": float(recall_score(test_labels, test_preds, zero_division=0)),
    "roc_auc_sklearn": float(roc_auc),
    "pr_auc_sklearn": float(pr_auc),
    "confusion_matrix_best_threshold": cm.tolist(),
}

with open(RESULTS_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
with open(RESULTS_DIR / "classification_report.txt", "w") as f:
    f.write(report)

for metric_name in ["loss", "accuracy", "roc_auc"]:
    plt.figure()
    plt.plot(history[f"train_{metric_name}"], label=f"train_{metric_name}")
    plt.plot(history[f"val_{metric_name}"], label=f"val_{metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f"CBAM attention CNN: {metric_name}")
    plt.savefig(RESULTS_DIR / f"{metric_name}_curve.png", bbox_inches="tight")
    plt.close()

print("\nSaved outputs to:", RESULTS_DIR)


# --------------------------------------------------
# 12. Visualise CBAM's learned spatial attention maps
# (the model's own attention, not a post-hoc Grad-CAM explanation)
# on a sample of correct and incorrect test predictions.
# --------------------------------------------------

test_rows = test_rows.copy()
test_rows["true_label"] = y_test
test_rows["predicted_probability"] = test_probs
test_rows["predicted_label"] = test_preds
test_rows["correct"] = test_rows["true_label"] == test_rows["predicted_label"]

correct_lens = test_rows[(test_rows["true_label"] == 1) & (test_rows["correct"])].sample(
    n=min(3, (test_rows["true_label"].eq(1) & test_rows["correct"]).sum()), random_state=RANDOM_SEED
)
correct_nonlens = test_rows[(test_rows["true_label"] == 0) & (test_rows["correct"])].sample(
    n=min(3, (test_rows["true_label"].eq(0) & test_rows["correct"]).sum()), random_state=RANDOM_SEED
)
misclassified = test_rows[~test_rows["correct"]].head(6)

attention_selection = pd.concat([correct_lens, correct_nonlens, misclassified]).reset_index(drop=True)

if len(attention_selection) > 0:
    n = len(attention_selection)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
    axes = np.atleast_1d(axes).flatten()

    model.eval()
    for ax, (_, row) in zip(axes, attention_selection.iterrows()):
        image_array = load_image_array(row["file_path"])
        input_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        input_tensor = ((input_tensor - torch.tensor(mean.reshape(3, 1, 1))) /
                         torch.tensor(std.reshape(3, 1, 1))).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, spatial_map = model(input_tensor, return_attention=True)

        attention_map = spatial_map[0, 0].cpu().numpy()
        attention_resized = np.asarray(
            Image.fromarray((attention_map * 255).astype("uint8")).resize(
                (IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR
            ), dtype="float32"
        ) / 255.0

        true_name = "lens" if row["true_label"] == 1 else "non_lens"
        pred_name = "lens" if row["predicted_label"] == 1 else "non_lens"
        status = "CORRECT" if row["correct"] else "WRONG"

        ax.imshow(image_array.astype("uint8"))
        ax.imshow(attention_resized, cmap="jet", alpha=0.45)
        ax.set_title(
            f"{row['image_id']}\ntrue={true_name} pred={pred_name} ({row['predicted_probability']:.2f})\n{status}",
            fontsize=8,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cbam_attention_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved CBAM attention grid to {RESULTS_DIR / 'cbam_attention_grid.png'}")