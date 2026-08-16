from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,classification_report,roc_auc_score,average_precision_score,precision_score,recall_score,f1_score,)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# paths
BASE_FOLDER = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_FOLDER / "data" / "metadata" / "quality_metadata.csv"
RESULTS_DIR = BASE_FOLDER / "results" / "transfer_learning_cnn"
MODELS_DIR = BASE_FOLDER / "models" / "transfer_learning_cnn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# settings
IMAGE_SIZE = 128
RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 8
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# load metadata
metadata = pd.read_csv(METADATA_PATH)

# use only images that passed the quality check
trainable = metadata[(metadata["usable"] == 1) & (metadata["class_name"].isin(["lens", "presumed_non_lens"]))].copy()
print("Trainable images:", len(trainable))
def load_image_array(file_path, size=IMAGE_SIZE):
    # load image, convert to RGB and resize
    with Image.open(file_path) as image:
        image = image.convert("RGB")
        image = image.resize((size, size))
        return np.asarray(image, dtype="float32")

# load images into arrays
images = []
labels = []
for _, row in trainable.iterrows():
    try:
        images.append(load_image_array(row["file_path"]))
        labels.append(int(row["label"]))
    except Exception as e:
        print("Skipped:", row["file_path"], e)
X = np.stack(images, axis=0)
y = np.array(labels, dtype="int64")
print("X shape:", X.shape)
print("y shape:", y.shape)

# train, validation and test split
train_idx, temp_idx = train_test_split(np.arange(len(X)),test_size=0.30,random_state=RANDOM_SEED,stratify=y)
val_idx, test_idx = train_test_split(temp_idx,test_size=0.50,random_state=RANDOM_SEED,stratify=y[temp_idx])
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Train:", X_train.shape, np.unique(y_train, return_counts=True))
print("Val:", X_val.shape, np.unique(y_val, return_counts=True))
print("Test:", X_test.shape, np.unique(y_test, return_counts=True))

# ImageNet values are used because MobileNet was trained with them
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class LensDatasetImageNet(Dataset):
    def __init__(self, X_arr, y_arr, augment=False):
        self.X = X_arr
        self.y = y_arr
        self.augment = augment
        self.aug_transform = T.Compose([T.RandomHorizontalFlip(p=0.5),T.RandomVerticalFlip(p=0.5),T.RandomRotation(180),])
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        image = torch.from_numpy(self.X[idx]).permute(2, 0, 1).float()
        # scale image values to 0-1
        image = image / 255.0
        if self.augment: image = self.aug_transform(image)
        # normalise for pretrained MobileNet
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return image, label

# create data loaders
train_dataset = LensDatasetImageNet(X_train, y_train, augment=True)
val_dataset = LensDatasetImageNet(X_val, y_val, augment=False)
test_dataset = LensDatasetImageNet(X_test, y_test, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class TransferLearningCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # pretrained MobileNet model
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # freeze the feature extractor
        for param in backbone.features.parameters():
            param.requires_grad = False
        self.backbone = backbone.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        # small classifier for our binary task
        self.classifier = nn.Sequential(nn.Linear(576, 32),nn.ReLU(),nn.Dropout(0.5),nn.Linear(32, 1),)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x.squeeze(1)
model = TransferLearningCNN().to(DEVICE)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print("Trainable params:", trainable_params)
print("Total params:", total_params)

# loss and optimiser
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
pos_weight_value = n_neg / max(n_pos, 1)
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# only update the classifier layers
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=4,min_lr=1e-6)
def run_epoch(loader, train):
    if train: model.train()
    else: model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    if train: context = torch.enable_grad()
    else: context = torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    probs = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)
    accuracy = float((preds == labels_arr).mean())
    if len(np.unique(labels_arr)) > 1: roc_auc = float(roc_auc_score(labels_arr, probs))
    else: roc_auc = float("nan")
    return {"loss": avg_loss,"accuracy": accuracy,"roc_auc": roc_auc,}

# store training history for plots
history = {"train_loss": [],"train_accuracy": [],"train_roc_auc": [],"val_loss": [],"val_accuracy": [],"val_roc_auc": [],}

# train model
best_val_loss = float("inf")
epochs_without_improvement = 0
best_model_path = MODELS_DIR / "best_transfer_learning_cnn.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    train_metrics = run_epoch(train_loader, train=True)
    val_metrics = run_epoch(val_loader, train=False)
    scheduler.step(val_metrics["loss"])
    history["train_loss"].append(train_metrics["loss"])
    history["train_accuracy"].append(train_metrics["accuracy"])
    history["train_roc_auc"].append(train_metrics["roc_auc"])
    history["val_loss"].append(val_metrics["loss"])
    history["val_accuracy"].append(val_metrics["accuracy"])
    history["val_roc_auc"].append(val_metrics["roc_auc"])

    print(f"Epoch {epoch}/{MAX_EPOCHS} | train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | val_auc={val_metrics['roc_auc']:.4f}")
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print("Early stopping at epoch:", epoch)
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


# load best model
model.load_state_dict(torch.load(best_model_path))
print("Loaded best model. Best val loss:", best_val_loss)

# validation predictions for threshold tuning
model.eval()
val_probs = []
val_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        val_probs.append(probs)
        val_labels.append(labels.numpy())
val_probs = np.concatenate(val_probs)
val_labels = np.concatenate(val_labels)
# choose best threshold using validation F1
thresholds = np.linspace(0.01, 0.99, 99)
best_threshold = 0.5
best_f1 = 0.0

for threshold in thresholds:
    preds = (val_probs >= threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
print("Best threshold:", best_threshold)
print("Best validation F1:", best_f1)
# test predictions
test_probs = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        test_probs.append(probs)
        test_labels.append(labels.numpy())
test_probs = np.concatenate(test_probs)
test_labels = np.concatenate(test_labels)
test_preds = (test_probs >= best_threshold).astype(int)


# evaluation
cm = confusion_matrix(test_labels, test_preds)
report = classification_report(test_labels,test_preds,target_names=["non_lens", "lens"],zero_division=0)
roc_auc = roc_auc_score(test_labels, test_probs)
pr_auc = average_precision_score(test_labels, test_probs)
print("Confusion matrix:")
print(cm)
print("Classification report:")
print(report)
print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

# save metrics
metrics = {
    "model": "MobileNetV3-Small transfer learning",
    "trainable_params": int(trainable_params),
    "total_params": int(total_params),
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "best_threshold": float(best_threshold),
    "best_validation_f1": float(best_f1),
    "test_accuracy": float((test_preds == test_labels).mean()),
    "test_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
    "test_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
    "test_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
    "test_roc_auc": float(roc_auc),
    "test_pr_auc": float(pr_auc),
    "confusion_matrix": cm.tolist(),
}

with open(RESULTS_DIR / "test_metrics.json", "w") as f:json.dump(metrics, f, indent=4)
with open(RESULTS_DIR / "classification_report.txt", "w") as f:f.write(report)
torch.save(model.state_dict(), MODELS_DIR / "final_transfer_learning_cnn.pt")
# plot training curves
for metric_name in ["loss", "accuracy", "roc_auc"]:
    plt.figure()
    plt.plot(history[f"train_{metric_name}"],label=f"train_{metric_name}")
    plt.plot(history[f"val_{metric_name}"],label=f"val_{metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f"Transfer learning: {metric_name}")
    plt.savefig(RESULTS_DIR / f"{metric_name}_curve.png", bbox_inches="tight")
    plt.close()

print("Saved results to:", RESULTS_DIR)
print("Saved model to:", MODELS_DIR)