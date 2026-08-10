"""
04_train_real_data_cnn.py  (PyTorch version)

Trains a CNN on the real, quality-controlled Legacy Survey lens / non-lens
images (as opposed to the simulated Baseline Prototype model, which was
trained in Keras/TensorFlow). Same architecture, same splits, same metrics,
same output file names/formats as the TensorFlow version, so the two
models compare cleanly in the report.

Run this AFTER 03_quality_filter.py has produced quality_metadata.csv.

'probable_lens' images are excluded from training (lower label
confidence) and instead scored separately as a held-out qualitative
test set at the end of this script.
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
# 1. Paths — same folder as your other scripts
# --------------------------------------------------

BASE_FOLDER = Path(
    r"D:\MSc Data and Computational Science"
    r"\Gravitational Lensing\Gravitational_Lensing_Project"
    r"\projects-ahana-vaishnav-ahanabhattacharji-Strong-Gravitational-Lens-Finding-Challenge"
    r"\Real Data pipeline"
)

METADATA_PATH = BASE_FOLDER / "data" / "metadata" / "quality_metadata.csv"
RESULTS_DIR = BASE_FOLDER / "results" / "real_data_cnn"
MODELS_DIR = BASE_FOLDER / "models" / "real_data_cnn"

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
# 2. Load quality-filtered metadata
# --------------------------------------------------

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {METADATA_PATH}\nRun 03_quality_filter.py first."
    )

metadata = pd.read_csv(METADATA_PATH)

print("Rows in quality_metadata.csv:", len(metadata))
print(metadata.groupby("class_name")["usable"].agg(["sum", "count"]))

trainable = metadata[
    (metadata["usable"] == 1)
    & (metadata["class_name"].isin(["lens", "presumed_non_lens"]))
].copy()

held_out_probable = metadata[
    (metadata["usable"] == 1)
    & (metadata["class_name"] == "probable_lens")
].copy()

print(f"\nTrainable images (lens + non-lens, usable): {len(trainable)}")
print(f"Held-out 'probable' images for qualitative testing: {len(held_out_probable)}")

if len(trainable) < 20:
    raise ValueError(
        "Very few usable trainable images found — check that the downloads "
        "finished and 03_quality_filter.py ran on the full dataset, not the pilot."
    )


# --------------------------------------------------
# 3. Load images into arrays
# --------------------------------------------------

def load_image_array(file_path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    with Image.open(file_path) as image:
        image = image.convert("RGB").resize((size, size))
        return np.asarray(image, dtype="float32")  # H, W, C


print("\nLoading images into memory...")

images = []
labels = []

for _, row in trainable.iterrows():
    try:
        images.append(load_image_array(row["file_path"]))
        labels.append(int(row["label"]))
    except Exception as error:
        print(f"  Skipping {row['image_id']}: {error}")

X = np.stack(images, axis=0)               # N, H, W, C
y = np.array(labels, dtype="int64")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label counts:", np.unique(y, return_counts=True))


# --------------------------------------------------
# 4. Train / validation / test split (same ratios as baseline)
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

print("\nSplit sizes")
print("Train:", X_train.shape, np.unique(y_train, return_counts=True))
print("Val:  ", X_val.shape, np.unique(y_val, return_counts=True))
print("Test: ", X_test.shape, np.unique(y_test, return_counts=True))


# --------------------------------------------------
# 5. Normalise using training set only
# --------------------------------------------------

mean = X_train.mean(axis=(0, 1, 2), keepdims=True)   # 1,1,1,C
std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6

print("\nNormalisation stats computed from training set")
print("Mean:", mean.ravel(), "Std:", std.ravel())


# --------------------------------------------------
# 6. Dataset / DataLoader
#
# Images are converted to CHW tensors. Augmentation (random flips +
# rotation — lensing has no preferred orientation) is applied only
# to the training split, after which all splits are normalised
# using the training set's mean/std.
# --------------------------------------------------

class LensDataset(Dataset):
    def __init__(self, X_arr, y_arr, mean, std, augment: bool):
        # Store as HWC uint8-range float32; convert per-item.
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
        image = torch.from_numpy(self.X[idx]).permute(2, 0, 1).float()  # C,H,W

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
# 7. Model — same architecture as the TensorFlow baseline:
# Conv(16) -> Pool -> Conv(32) -> Pool -> Conv(64) -> GAP
# -> Dense(32) -> Dropout(0.5) -> Dense(1)
# --------------------------------------------------

class RealDataCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),  # logits — sigmoid applied outside via loss/inference
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x).squeeze(1)  # logits, shape (N,)


model = RealDataCNN().to(DEVICE)
print(model)


# --------------------------------------------------
# 8. Class weighting (balanced, same formula as sklearn's
# compute_class_weight) via BCEWithLogitsLoss's pos_weight.
# --------------------------------------------------

n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
pos_weight_value = n_neg / max(n_pos, 1)

print(f"\nClass counts — non_lens: {n_neg}, lens: {n_pos}")
print("pos_weight (applied to lens/positive class):", pos_weight_value)

pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# --------------------------------------------------
# 9. Optimiser + LR scheduler
# --------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
)


# --------------------------------------------------
# 10. Training loop with early stopping + checkpointing
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

    metrics = {
        "loss": avg_loss,
        "accuracy": float((preds == labels_arr).mean()),
        "precision": float(precision_score(labels_arr, preds, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_arr, probs)) if len(np.unique(labels_arr)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(labels_arr, probs)) if len(np.unique(labels_arr)) > 1 else float("nan"),
    }
    return metrics


history = {f"{split}_{name}": [] for split in ["train", "val"]
           for name in ["loss", "accuracy", "roc_auc"]}

best_val_loss = float("inf")
epochs_without_improvement = 0
best_model_path = MODELS_DIR / "best_real_data_cnn.pt"

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
        f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
        f"val_roc_auc={val_metrics['roc_auc']:.4f}"
    )

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no val_loss improvement for {EARLY_STOP_PATIENCE} epochs).")
        break

# Restore best weights (mirrors Keras's restore_best_weights=True)
model.load_state_dict(torch.load(best_model_path))
print(f"\nRestored best model weights (val_loss={best_val_loss:.4f}).")


# --------------------------------------------------
# 11. Tune threshold on validation set
# --------------------------------------------------

model.eval()
val_probs, val_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        logits = model(images)
        val_probs.append(torch.sigmoid(logits).cpu().numpy())
        val_labels.append(labels.numpy())

val_probs = np.concatenate(val_probs)
val_labels = np.concatenate(val_labels)

thresholds = np.linspace(0.01, 0.99, 99)
best_threshold = 0.5
best_f1 = 0.0

for threshold in thresholds:
    preds = (val_probs >= threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest threshold from validation set:", best_threshold)
print("Best validation F1:", best_f1)


# --------------------------------------------------
# 12. Evaluate on test set
# --------------------------------------------------

test_probs, test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        logits = model(images)
        test_probs.append(torch.sigmoid(logits).cpu().numpy())
        test_labels.append(labels.numpy())

test_probs = np.concatenate(test_probs)
test_labels = np.concatenate(test_labels)
test_preds = (test_probs >= best_threshold).astype(int)

cm = confusion_matrix(test_labels, test_preds)
report = classification_report(
    test_labels, test_preds, target_names=["non_lens", "lens"], zero_division=0
)
roc_auc = roc_auc_score(test_labels, test_probs)
pr_auc = average_precision_score(test_labels, test_probs)
test_accuracy = float((test_preds == test_labels).mean())
test_precision = float(precision_score(test_labels, test_preds, zero_division=0))
test_recall = float(recall_score(test_labels, test_preds, zero_division=0))

print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", report)
print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)


# --------------------------------------------------
# 13. Save model, metrics, plots
# --------------------------------------------------

torch.save(model.state_dict(), MODELS_DIR / "final_real_data_cnn.pt")

metrics = {
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "best_threshold": float(best_threshold),
    "best_validation_f1": float(best_f1),
    "test_accuracy_default_metric": test_accuracy,
    "test_precision_default_metric": test_precision,
    "test_recall_default_metric": test_recall,
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
    plt.title(f"Training and validation {metric_name} (real data, PyTorch)")
    plt.savefig(RESULTS_DIR / f"{metric_name}_curve.png", bbox_inches="tight")
    plt.close()

print("\nSaved outputs to:", RESULTS_DIR)


# --------------------------------------------------
# 14. Score the held-out 'probable' images (qualitative only)
# --------------------------------------------------

if len(held_out_probable) > 0:
    probable_images = np.stack(
        [load_image_array(p) for p in held_out_probable["file_path"]], axis=0
    )
    probable_dataset = LensDataset(
        probable_images,
        np.zeros(len(probable_images)),  # labels unused/unreliable for this tier
        mean, std, augment=False,
    )
    probable_loader = DataLoader(probable_dataset, batch_size=BATCH_SIZE, shuffle=False)

    probable_probs = []
    with torch.no_grad():
        for images, _ in probable_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probable_probs.append(torch.sigmoid(logits).cpu().numpy())
    probable_probs = np.concatenate(probable_probs)

    held_out_probable = held_out_probable.copy()
    held_out_probable["predicted_probability"] = probable_probs
    held_out_probable["predicted_label"] = (probable_probs >= best_threshold).astype(int)

    held_out_probable.to_csv(RESULTS_DIR / "probable_tier_predictions.csv", index=False)

    print(
        f"\nScored {len(held_out_probable)} held-out 'probable' images — "
        f"saved to {RESULTS_DIR / 'probable_tier_predictions.csv'}"
    )
    print(
        "Mean predicted probability:", float(probable_probs.mean()),
        "| fraction predicted as lens:", float((probable_probs >= best_threshold).mean()),
    )