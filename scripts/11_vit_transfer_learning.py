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
from torchvision.models import vit_b_16, ViT_B_16_Weights


# Paths

BASE_FOLDER = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_FOLDER / "data" / "metadata" / "quality_metadata.csv"
RESULTS_DIR = BASE_FOLDER / "results" / "vit_transfer_learning"
MODELS_DIR = BASE_FOLDER / "models" / "vit_transfer_learning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIT_IMAGE_SIZE = 224  # ViT-B/16's required input resolution
PATCH_SIZE = 16
GRID_SIZE = VIT_IMAGE_SIZE // PATCH_SIZE  # 14x14 patches
RANDOM_SEED = 25206621
BATCH_SIZE = 16  # smaller than other scripts — ViT at 224x224 uses more memory
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 8

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# Load metadata
metadata = pd.read_csv(METADATA_PATH)
trainable = metadata[(metadata["usable"] == 1) & (metadata["class_name"].isin(["lens", "presumed_non_lens"]))].copy().reset_index(drop=True)
print(f"Trainable images: {len(trainable)}")

y_all = trainable["label"].to_numpy(dtype="int64")

# Same train/val/test split 

train_idx, temp_idx = train_test_split(np.arange(len(trainable)), test_size=0.30, random_state=RANDOM_SEED, stratify=y_all)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, stratify=y_all[temp_idx])
train_rows = trainable.iloc[train_idx].reset_index(drop=True)
val_rows = trainable.iloc[val_idx].reset_index(drop=True)
test_rows = trainable.iloc[test_idx].reset_index(drop=True)
print("\nSplit sizes")
print("Train:", len(train_rows), np.unique(train_rows["label"], return_counts=True))
print("Val:  ", len(val_rows), np.unique(val_rows["label"], return_counts=True))
print("Test: ", len(test_rows), np.unique(test_rows["label"], return_counts=True))


# Dataset — loads images lazily at 224x224, ImageNet-normalised

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
class LensDatasetViT(Dataset):
    def __init__(self, rows_df: pd.DataFrame, augment: bool):
        self.rows = rows_df.reset_index(drop=True)
        self.augment = augment
        self.aug_transform = T.Compose([T.RandomHorizontalFlip(p=0.5),T.RandomVerticalFlip(p=0.5),T.RandomRotation(180),])
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        with Image.open(row["file_path"]) as image:
            image = image.convert("RGB").resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE))
            array = np.asarray(image, dtype="float32")
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        if self.augment:
            tensor = self.aug_transform(tensor)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        label = torch.tensor(int(row["label"]), dtype=torch.float32)
        return tensor, label

train_dataset = LensDatasetViT(train_rows, augment=True)
val_dataset = LensDatasetViT(val_rows, augment=False)
test_dataset = LensDatasetViT(test_rows, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. Model — frozen ViT-B/16 backbone

class ViTTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # Freeze everything — only the new head trains.
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone
        # ViT-B/16's hidden dimension is 768.
        self.classifier = nn.Sequential(nn.Linear(768, 32),nn.ReLU(),nn.Dropout(0.5),nn.Linear(32, 1))

    def forward(self, x):
        features = self.backbone._process_input(x)
        batch_size = features.shape[0]
        batch_class_token = self.backbone.class_token.expand(batch_size, -1, -1)
        features = torch.cat([batch_class_token, features], dim=1)
        features = self.backbone.encoder(features)
        cls_token = features[:, 0]
        return self.classifier(cls_token).squeeze(1)
model = ViTTransferModel().to(DEVICE)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params: {trainable_params:,} / {total_params:,} total (backbone frozen)")

# Class weighting, optimiser, scheduler
n_neg = int((train_rows["label"] == 0).sum())
n_pos = int((train_rows["label"] == 1).sum())
pos_weight_value = n_neg / max(n_pos, 1)
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)

# Training loop

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
best_model_path = MODELS_DIR / "best_vit_transfer.pt"

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
history_df = pd.DataFrame(history)
history_df.insert(0, "epoch", range(1, len(history_df) + 1))
history_path = RESULTS_DIR / "training_history.csv"
history_df.to_csv(history_path, index=False)
print(f"Saved training history to: {history_path}")
model.load_state_dict(torch.load(best_model_path))
print(f"\nRestored best model (val_loss={best_val_loss:.4f}).")

# Threshold tuning + test evaluation

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


# Save model, metrics, plots
torch.save(model.state_dict(), MODELS_DIR / "final_vit_transfer.pt")
metrics = {
    "model": "ViT-B/16 (frozen backbone, transfer learning)",
    "trainable_params": trainable_params,
    "total_params": total_params,
    "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
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
    plt.title(f"ViT-B/16 transfer learning: {metric_name}")
    plt.savefig(RESULTS_DIR / f"{metric_name}_curve.png", bbox_inches="tight")
    plt.close()

print("\nSaved outputs to:", RESULTS_DIR)

# 10. Attention-map visualisation

def get_attention_map(model: ViTTransferModel, image_tensor: torch.Tensor):
    last_block = model.backbone.encoder.layers[-1].self_attention
    original_forward = last_block.forward
    captured = {}

    def patched_forward(query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        output, weights = original_forward(query, key, value, **kwargs)
        captured["weights"] = weights.detach()
        return output, weights

    last_block.forward = patched_forward

    try:
        with torch.no_grad():
            logit = model(image_tensor)
            probability = torch.sigmoid(logit).item()
        if "weights" not in captured:
            return probability, None
        cls_attention = captured["weights"][0, 0, 1:].cpu().numpy()
        if cls_attention.shape[0] != GRID_SIZE * GRID_SIZE:
            return probability, None
        attention_grid = cls_attention.reshape(GRID_SIZE, GRID_SIZE)
        if attention_grid.max() > 0:
            attention_grid = attention_grid / attention_grid.max()
        return probability, attention_grid
    except Exception as error:
        print(f"  (attention capture skipped: {error})")
        return None, None
    finally:
        last_block.forward = original_forward


test_rows = test_rows.copy()
test_rows["true_label"] = test_labels
test_rows["predicted_probability"] = test_probs
test_rows["predicted_label"] = test_preds
test_rows["correct"] = test_rows["true_label"] == test_rows["predicted_label"]

correct_lens = test_rows[(test_rows["true_label"] == 1) & (test_rows["correct"])].sample(n=min(3, (test_rows["true_label"].eq(1) & test_rows["correct"]).sum()), random_state=RANDOM_SEED)
correct_nonlens = test_rows[(test_rows["true_label"] == 0) & (test_rows["correct"])].sample(n=min(3, (test_rows["true_label"].eq(0) & test_rows["correct"]).sum()), random_state=RANDOM_SEED)
misclassified = test_rows[~test_rows["correct"]].head(6)
attention_selection = pd.concat([correct_lens, correct_nonlens, misclassified]).reset_index(drop=True)

if len(attention_selection) > 0:
    n = len(attention_selection)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
    axes = np.atleast_1d(axes).flatten()
    any_attention_captured = False
    for ax, (_, row) in zip(axes, attention_selection.iterrows()):
        with Image.open(row["file_path"]) as image:
            display_image = np.asarray(image.convert("RGB").resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE)))
        input_tensor = torch.from_numpy(display_image.astype("float32")).permute(2, 0, 1) / 255.0
        input_tensor = ((input_tensor - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(DEVICE)
        probability, attention_grid = get_attention_map(model, input_tensor)
        true_name = "lens" if row["true_label"] == 1 else "non_lens"
        pred_name = "lens" if row["predicted_label"] == 1 else "non_lens"
        status = "CORRECT" if row["correct"] else "WRONG"
        ax.imshow(display_image)
        if attention_grid is not None:
            attention_resized = np.asarray(
                Image.fromarray((attention_grid * 255).astype("uint8")).resize(
                    (VIT_IMAGE_SIZE, VIT_IMAGE_SIZE), resample=Image.BILINEAR
                ), dtype="float32"
            ) / 255.0
            ax.imshow(attention_resized, cmap="jet", alpha=0.45)
            any_attention_captured = True
        ax.set_title(f"{row['image_id']}\ntrue={true_name} pred={pred_name} ({row['predicted_probability']:.2f})\n{status}",fontsize=8,)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "vit_attention_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
