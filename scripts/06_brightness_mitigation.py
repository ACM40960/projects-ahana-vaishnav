from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,classification_report,roc_auc_score,average_precision_score,precision_score,recall_score,f1_score,)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

# paths
BASE_FOLDER = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_FOLDER / "data" / "metadata" / "quality_metadata.csv"
RESULTS_DIR = BASE_FOLDER / "results" / "real_data_cnn_brightness_mitigated"
MODELS_DIR = BASE_FOLDER / "models" / "real_data_cnn_brightness_mitigated"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# settings
IMAGE_SIZE = 128
RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# load metadata
metadata = pd.read_csv(METADATA_PATH)
# only use the images that passed the quality check
trainable = metadata[(metadata["usable"] == 1) & (metadata["class_name"].isin(["lens", "presumed_non_lens"]))].copy()
# keeping these separate because the label is not fully certain
held_out_probable = metadata[(metadata["usable"] == 1)& (metadata["class_name"] == "probable_lens")].copy()
print("Trainable images:", len(trainable))
print("Held-out probable images:", len(held_out_probable))

def load_image_array(file_path, size=IMAGE_SIZE):
    # load image as RGB and resize to the same size
    with Image.open(file_path) as image:
        image = image.convert("RGB")
        image = image.resize((size, size))
        return np.asarray(image, dtype="float32")

# load all images into numpy arrays
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

# split data into train, validation and test sets
train_idx, temp_idx = train_test_split(np.arange(len(X)),test_size=0.30,random_state=RANDOM_SEED,stratify=y)
val_idx, test_idx = train_test_split(temp_idx,test_size=0.50,random_state=RANDOM_SEED,stratify=y[temp_idx])
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]
test_rows = trainable.reset_index(drop=True).iloc[test_idx].reset_index(drop=True)

print("Train:", X_train.shape, np.unique(y_train, return_counts=True))
print("Val:", X_val.shape, np.unique(y_val, return_counts=True))
print("Test:", X_test.shape, np.unique(y_test, return_counts=True))

class LensDatasetPerImageNorm(Dataset):
    def __init__(self, X_arr, y_arr, augment=False):
        self.X = X_arr
        self.y = y_arr
        self.augment = augment
        self.aug_transform = T.Compose([T.RandomHorizontalFlip(p=0.5),T.RandomVerticalFlip(p=0.5),T.RandomRotation(180),])
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        image = torch.from_numpy(self.X[idx]).permute(2, 0, 1).float()
        if self.augment: image = self.aug_transform(image)
        # normalise each image separately
        image_mean = image.mean(dim=(1, 2), keepdim=True)
        image_std = image.std(dim=(1, 2), keepdim=True) + 1e-6
        image = (image - image_mean) / image_std
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return image, label

# make PyTorch datasets and loaders
train_dataset = LensDatasetPerImageNorm(X_train, y_train, augment=True)
val_dataset = LensDatasetPerImageNorm(X_val, y_val, augment=False)
test_dataset = LensDatasetPerImageNorm(X_test, y_test, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class RealDataCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # small CNN for binary image classification
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding="same"),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),nn.ReLU()
            )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(64, 32),nn.ReLU(),nn.Dropout(0.5),nn.Linear(32, 1))
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x.squeeze(1)
model = RealDataCNN().to(DEVICE)
# loss and optimiser
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())

pos_weight_value = n_neg / max(n_pos, 1)
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=5,min_lr=1e-6)

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
    return {"loss": avg_loss,"accuracy": accuracy,"roc_auc": roc_auc}

# train the model
best_val_loss = float("inf")
epochs_without_improvement = 0
best_model_path = MODELS_DIR / "best_brightness_mitigated_cnn.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    train_metrics = run_epoch(train_loader, train=True)
    val_metrics = run_epoch(val_loader, train=False)
    scheduler.step(val_metrics["loss"])
    print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | val_roc_auc={val_metrics['roc_auc']:.4f} ")
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_path)
    else: epochs_without_improvement += 1
    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print("Early stopping at epoch:", epoch)
        break
model.load_state_dict(torch.load(best_model_path))
print("Loaded best model. Best val loss:", best_val_loss)

# get validation probabilities to choose a threshold
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

# choose the threshold with best validation F1 score
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

# test set predictions
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

# metrics
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

# save results
metrics = {
    "note": "CNN trained using per-image normalisation",
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
    "confusion_matrix": cm.tolist()
}

with open(RESULTS_DIR / "test_metrics.json", "w") as f: json.dump(metrics, f, indent=4)
with open(RESULTS_DIR / "classification_report.txt", "w") as f: f.write(report)
torch.save(model.state_dict(), MODELS_DIR / "final_brightness_mitigated_cnn.pt")

# check if predictions are still related to brightness ratio
test_rows = test_rows.copy()
test_rows["predicted_probability"] = test_probs
correlation, p_value = stats.pearsonr(test_rows["center_edge_ratio"],test_rows["predicted_probability"])
print("Brightness correlation:")
print("r =", round(correlation, 3))
print("p-value =", p_value)

# plot brightness ratio against predicted probability
plt.figure(figsize=(7, 6))
colors = test_rows["label"].map({0: "tab:blue", 1: "tab:orange"})
plt.scatter(test_rows["center_edge_ratio"],test_rows["predicted_probability"],c=colors,alpha=0.6,edgecolors="none")
plt.xlabel("Center / edge brightness ratio")
plt.ylabel("Predicted probability of lens")
plt.title(f"Prediction vs brightness ratio after per-image normalisation\nr = {correlation:.3f}")
plt.axhline(best_threshold,color="gray",linestyle="--",linewidth=1,label=f"threshold = {best_threshold:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "brightness_vs_prediction_mitigated.png",dpi=150,bbox_inches="tight")
plt.close()
print("Saved results to:", RESULTS_DIR)
print("Saved model to:", MODELS_DIR)