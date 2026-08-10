"""
07_transfer_learning_cnn.py

Transfer-learning comparison model: a frozen pretrained MobileNetV3-Small
backbone (ImageNet weights) with a small trainable classification head,
fine-tuned on the same real lens/non-lens dataset used by
04_train_real_data_cnn.py. This directly fulfils the original proposal's
"optional comparison to a transfer learning model" and gives a third
clean comparison point alongside the simulated baseline and the
custom CNN.

Same splits, class weighting, thresholding, and output format as
04_train_real_data_cnn.py for a fair, direct comparison.

Requires torchvision (already installed alongside torch).
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
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_FOLDER = Path(
    r"D:\MSc Data and Computational Science"
    r"\Gravitational Lensing\Gravitational_Lensing_Project"
    r"\projects-ahana-vaishnav-ahanabhattacharji-Strong-Gravitational-Lens-Finding-Challenge"
    r"\Real Data pipeline"
)

METADATA_PATH = BASE_FOLDER / "data" / "metadata" / "quality_metadata.csv"
RESULTS_DIR = BASE_FOLDER / "results" / "transfer_learning_cnn"
MODELS_DIR = BASE_FOLDER / "models" / "transfer_learning_cnn"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 40  # transfer learning converges faster than training from scratch
EARLY_STOP_PATIENCE = 8

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# --------------------------------------------------
# 2. Load metadata (same filtering as 04)
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


images, labels = [], []
for _, row in trainable.iterrows():
    try:
        images.append(load_image_array(row["file_path"]))
        labels.append(int(row["label"]))
    except Exception:
        continue

X = np.stack(images, axis=0)
y = np.array(labels, dtype="int64")


# --------------------------------------------------
# 3. Same split as 04_train_real_data_cnn.py
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

print("\nSplit sizes")
print("Train:", X_train.shape, np.unique(y_train, return_counts=True))
print("Val:  ", X_val.shape, np.unique(y_val, return_counts=True))
print("Test: ", X_test.shape, np.unique(y_test, return_counts=True))


# --------------------------------------------------
# 4. Dataset — uses ImageNet normalisation stats, since the
# backbone was pretrained with them. Also resizes to 128x128
# (MobileNet accepts variable input size via adaptive pooling).
# --------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class LensDatasetImageNet(Dataset):
    def __init__(self, X_arr, y_arr, augment: bool):
        self.X = X_arr
        self.y = y_arr
        self.augment = augment
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(180),
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.X[idx]).permute(2, 0, 1).float() / 255.0  # scale to [0,1]

        if self.augment:
            image = self.aug_transform(image)

        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return image, label


train_dataset = LensDatasetImageNet(X_train, y_train, augment=True)
val_dataset = LensDatasetImageNet(X_val, y_val, augment=False)
test_dataset = LensDatasetImageNet(X_test, y_test, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --------------------------------------------------
# 5. Model — frozen MobileNetV3-Small backbone + small trainable head
# --------------------------------------------------

class TransferLearningCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Freeze all backbone weights — only the new head trains.
        for param in backbone.features.parameters():
            param.requires_grad = False

        self.backbone = backbone.features  # feature extractor only
        self.gap = nn.AdaptiveAvgPool2d(1)

        # MobileNetV3-Small's last feature layer outputs 576 channels.
        self.classifier = nn.Sequential(
            nn.Linear(576, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x).squeeze(1)


model = TransferLearningCNN().to(DEVICE)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params: {trainable_params:,} / {total_params:,} total (backbone frozen)")


# --------------------------------------------------
# 6. Class weighting, optimiser, scheduler
# --------------------------------------------------

n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
pos_weight_value = n_neg / max(n_pos, 1)
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Only optimise the trainable head parameters.
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
)


# --------------------------------------------------
# 7. Training loop
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
best_model_path = MODELS_DIR / "best_transfer_learning_cnn.pt"

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
# 8. Threshold tuning + test evaluation
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
# 9. Save outputs
# --------------------------------------------------

metrics = {
    "model": "MobileNetV3-Small (frozen backbone, transfer learning)",
    "trainable_params": trainable_params,
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

torch.save(model.state_dict(), MODELS_DIR / "final_transfer_learning_cnn.pt")

for metric_name in ["loss", "accuracy", "roc_auc"]:
    plt.figure()
    plt.plot(history[f"train_{metric_name}"], label=f"train_{metric_name}")
    plt.plot(history[f"val_{metric_name}"], label=f"val_{metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f"Transfer learning (MobileNetV3-Small): {metric_name}")
    plt.savefig(RESULTS_DIR / f"{metric_name}_curve.png", bbox_inches="tight")
    plt.close()

print(f"\nSaved all outputs to: {RESULTS_DIR}")
print(f"\nCompare against:")
print(f"  Baseline (simulated):  ROC-AUC ~1.0 (see Baseline Prototype/results/test_metrics.json)")
print(f"  Real-data custom CNN:  ROC-AUC 0.9825 (see results/real_data_cnn/test_metrics.json)")
print(f"  Transfer learning:     ROC-AUC {roc_auc:.4f} (this run)")