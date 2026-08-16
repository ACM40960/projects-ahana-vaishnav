from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,roc_auc_score, average_precision_score,precision_score, recall_score, f1_score,)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T


# Go back to the main project folder from the scripts folder
BASE_FOLDER = Path(__file__).resolve().parent.parent

# Input metadata from the quality filtering step
METADATA_PATH = BASE_FOLDER / "data" / "metadata" / "quality_metadata.csv"

# Output folders for results and trained models
RESULTS_DIR = BASE_FOLDER / "results" / "real_data_cnn"
MODELS_DIR = BASE_FOLDER / "models" / "real_data_cnn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Main training settings
IMAGE_SIZE = 128
RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10

# Keep runs more reproducible
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Check that the quality metadata file exists
if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Run 03_quality_filter.py first.\nExpected: {METADATA_PATH}")

# Load the cleaned metadata
metadata = pd.read_csv(METADATA_PATH)
print(f"Rows: {len(metadata)}")
print(metadata.groupby("class_name")["usable"].agg(["sum", "count"]))

# Use only confirmed lens and presumed non-lens images for training
trainable = metadata[(metadata["usable"] == 1) & (metadata["class_name"].isin(["lens", "presumed_non_lens"]))].copy()

# Keep probable lenses separate so they are not mixed into the main train/test data
held_out_probable = metadata[(metadata["usable"] == 1) & (metadata["class_name"] == "probable_lens")].copy()

print(f"Trainable: {len(trainable)} | Held-out probable: {len(held_out_probable)}")
if len(trainable) < 20:
    raise ValueError("Too few trainable images — check downloads and quality filter ran on full dataset.")
def load_image_array(file_path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    # Load one image and resize it to the model input size
    with Image.open(file_path) as image:
        return np.asarray(image.convert("RGB").resize((size, size)),dtype="float32")

# Load all trainable images into memory
print("Loading images...")
images, labels = [], []
for _, row in trainable.iterrows():
    try:
        images.append(load_image_array(row["file_path"]))
        labels.append(int(row["label"]))
    except Exception as error:
        print(f"  Skipping {row['image_id']}: {error}")
X = np.stack(images, axis=0)
y = np.array(labels, dtype="int64")
print(f"X: {X.shape} | Labels: {np.unique(y, return_counts=True)}")

# Split into train, validation, and test sets
# Stratify keeps the lens/non-lens ratio similar in each split
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.30,random_state=RANDOM_SEED,stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.50,random_state=RANDOM_SEED,stratify=y_temp)
print(f"Split: {len(X_train)} / {len(X_val)} / {len(X_test)}")

# Normalise using only the training set to avoid data leakage
mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6

class LensDataset(Dataset):
    # Small PyTorch dataset for lens/non-lens images
    def __init__(self, X_arr, y_arr, mean, std, augment: bool):
        self.X = X_arr
        self.y = y_arr
        self.mean = torch.tensor(mean.reshape(3, 1, 1), dtype=torch.float32)
        self.std = torch.tensor(std.reshape(3, 1, 1), dtype=torch.float32)
        self.augment = augment
        # Simple augmentation for training images only
        self.aug = T.Compose([T.RandomHorizontalFlip(0.5),T.RandomVerticalFlip(0.5),T.RandomRotation(180)])
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        image = torch.from_numpy(self.X[idx]).permute(2, 0, 1).float()
        if self.augment:
            image = self.aug(image)
        image = (image - self.mean) / self.std
        return image, torch.tensor(self.y[idx], dtype=torch.float32)

# DataLoaders feed images to the model in batches
train_loader = DataLoader(LensDataset(X_train, y_train, mean, std, augment=True),batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(LensDataset(X_val, y_val, mean, std, augment=False),batch_size=BATCH_SIZE,shuffle=False)
test_loader = DataLoader(LensDataset(X_test, y_test, mean, std, augment=False),batch_size=BATCH_SIZE,shuffle=False)

class RealDataCNN(nn.Module):
    # Simple CNN baseline for the real image dataset
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding="same"),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(64, 32),nn.ReLU(),nn.Dropout(0.5),nn.Linear(32, 1))
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x).squeeze(1)
model = RealDataCNN().to(DEVICE)
print(model)

# Weight the positive class in case the split is not perfectly balanced
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
pos_weight = torch.tensor([n_neg / max(n_pos, 1)],dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"Class counts — non_lens: {n_neg}, lens: {n_pos} | pos_weight: {pos_weight.item():.3f}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Reduce learning rate if validation loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=5,min_lr=1e-6)

def run_epoch(loader, train: bool):
    # Runs one full pass through either the training or validation data
    model.train() if train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            if train: optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    probs = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)

    # Default 0.5 threshold is used during training logs only
    preds = (probs >= 0.5).astype(int)

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float((preds == labels_arr).mean()),
        "precision": float(precision_score(labels_arr, preds, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_arr, probs))
        if len(np.unique(labels_arr)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(labels_arr, probs))
        if len(np.unique(labels_arr)) > 1 else float("nan"),
    }

# Store training curves
history = {
    f"{split}_{metric}": []
    for split in ["train", "val"]
    for metric in ["loss", "accuracy", "roc_auc"]}
best_val_loss = float("inf")
epochs_without_improvement = 0
best_model_path = MODELS_DIR / "best_real_data_cnn.pt"

# Main training loop
for epoch in range(1, MAX_EPOCHS + 1):
    train_m = run_epoch(train_loader, train=True)
    val_m = run_epoch(val_loader, train=False)
    scheduler.step(val_m["loss"])
    for name in ["loss", "accuracy", "roc_auc"]:
        history[f"train_{name}"].append(train_m[name])
        history[f"val_{name}"].append(val_m[name])
    print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | loss {train_m['loss']:.4f}/{val_m['loss']:.4f} | acc {train_m['accuracy']:.4f}/{val_m['accuracy']:.4f} | val_roc {val_m['roc_auc']:.4f}")

    # Save best model based on validation loss
    if val_m["loss"] < best_val_loss:
        best_val_loss = val_m["loss"]
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_without_improvement += 1

    # Stop if validation loss has not improved for a while
    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

# Save training history so we can compare learning curves later
history_df = pd.DataFrame(history)
history_df.insert(0, "epoch", range(1, len(history_df) + 1))

history_path = RESULTS_DIR / "training_history.csv"
history_df.to_csv(history_path, index=False)

print(f"Saved training history to: {history_path}")

# Restore best model before final evaluation
model.load_state_dict(torch.load(best_model_path))
print(f"Restored best weights (val_loss={best_val_loss:.4f})")

# Get validation probabilities for threshold tuning
model.eval()
val_probs, val_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        val_probs.append(torch.sigmoid(model(images.to(DEVICE))).cpu().numpy())
        val_labels.append(labels.numpy())
val_probs = np.concatenate(val_probs)
val_labels = np.concatenate(val_labels)

# Choose the threshold that gives the best validation F1 score
best_threshold, best_f1 = 0.5, 0.0
for threshold in np.linspace(0.01, 0.99, 99):
    f1 = f1_score(val_labels,(val_probs >= threshold).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
print(f"Best threshold: {best_threshold:.2f} | Val F1: {best_f1:.4f}")

# Final test-set evaluation
test_probs, test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        test_probs.append(torch.sigmoid(model(images.to(DEVICE))).cpu().numpy())
        test_labels.append(labels.numpy())
test_probs = np.concatenate(test_probs)
test_labels = np.concatenate(test_labels)
test_preds = (test_probs >= best_threshold).astype(int)
cm = confusion_matrix(test_labels, test_preds)
report = classification_report(test_labels,test_preds,target_names=["non_lens", "lens"],zero_division=0)
roc_auc = roc_auc_score(test_labels, test_probs)
pr_auc = average_precision_score(test_labels, test_probs)
print("\nConfusion matrix:\n", cm)
print("\n", report)
print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

# Save final model
torch.save(model.state_dict(), MODELS_DIR / "final_real_data_cnn.pt")

# Save main metrics for the report/app
metrics = {
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "best_threshold": float(best_threshold),
    "best_validation_f1": float(best_f1),
    "test_accuracy_default_metric": float((test_preds == test_labels).mean()),
    "test_precision_default_metric": float(
        precision_score(test_labels, test_preds, zero_division=0)
    ),
    "test_recall_default_metric": float(
        recall_score(test_labels, test_preds, zero_division=0)
    ),
    "roc_auc_sklearn": float(roc_auc),
    "pr_auc_sklearn": float(pr_auc),
    "confusion_matrix_best_threshold": cm.tolist(),
}
with open(RESULTS_DIR / "test_metrics.json", "w") as f: json.dump(metrics, f, indent=4)
with open(RESULTS_DIR / "classification_report.txt", "w") as f: f.write(report)

# Save training curves
for name in ["loss", "accuracy", "roc_auc"]:
    plt.figure()
    plt.plot(history[f"train_{name}"], label="train")
    plt.plot(history[f"val_{name}"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.title(f"Real data CNN: {name}")
    plt.legend()
    plt.savefig(RESULTS_DIR / f"{name}_curve.png", bbox_inches="tight")
    plt.close()
print(f"Saved to: {RESULTS_DIR}")

# Extra check: run the model on probable lenses that were held out
if len(held_out_probable) > 0:
    probable_images = np.stack([load_image_array(path) for path in held_out_probable["file_path"]],axis=0)
    # Labels are set to zero here only because the dataset class requires labels
    probable_dataset = LensDataset(probable_images,np.zeros(len(probable_images)),mean,std,augment=False)
    probable_loader = DataLoader(probable_dataset,batch_size=BATCH_SIZE,shuffle=False)
    probable_probs = []
    with torch.no_grad():
        for images, _ in probable_loader:
            probable_probs.append(torch.sigmoid(model(images.to(DEVICE))).cpu().numpy())
    probable_probs = np.concatenate(probable_probs)
    held_out_probable = held_out_probable.copy()
    held_out_probable["predicted_probability"] = probable_probs
    held_out_probable["predicted_label"] = (probable_probs >= best_threshold).astype(int)
    held_out_probable.to_csv(RESULTS_DIR / "probable_tier_predictions.csv",index=False)
    print(f"Probable tier: mean prob={probable_probs.mean():.3f} | " f"fraction predicted lens={(probable_probs >= best_threshold).mean():.3f}")