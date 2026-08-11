"""
08_export_deployment_artifacts.py

Exports everything the Streamlit demo app needs, so the app doesn't have
to re-load the full training dataset just to get normalisation stats:
  - train-set mean/std (needed to preprocess new uploaded images the
    same way the model was trained)
  - the tuned decision threshold
  - a copy of the trained model weights

Run this AFTER 04_train_real_data_cnn.py. Recomputes the identical
train/val/test split (same seed) purely to get the train-set mean/std —
does not retrain anything.

Outputs go to: Real Data pipeline/deployment/artifacts/
"""

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


# Relative path: this script lives in "Real Data pipeline/scripts/",
# so its parent's parent is "Real Data pipeline/" itself. Works on any
# machine the repo is cloned to, no hardcoded drive/user path needed.
BASE_FOLDER = Path(__file__).resolve().parent.parent

METADATA_PATH = BASE_FOLDER / "data" / "metadata" / "quality_metadata.csv"
MODEL_PATH = BASE_FOLDER / "models" / "real_data_cnn" / "best_real_data_cnn.pt"
TEST_METRICS_PATH = BASE_FOLDER / "results" / "real_data_cnn" / "test_metrics.json"

DEPLOYMENT_DIR = BASE_FOLDER / "deployment"
ARTIFACTS_DIR = DEPLOYMENT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
RANDOM_SEED = 42


# --------------------------------------------------
# 1. Recompute the identical split to get train-set mean/std
# --------------------------------------------------

metadata = pd.read_csv(METADATA_PATH)

trainable = metadata[
    (metadata["usable"] == 1)
    & (metadata["class_name"].isin(["lens", "presumed_non_lens"]))
].copy()


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

train_idx, _ = train_test_split(
    np.arange(len(X)), test_size=0.30, random_state=RANDOM_SEED, stratify=y
)
X_train = X[train_idx]

mean = X_train.mean(axis=(0, 1, 2))  # shape (3,)
std = X_train.std(axis=(0, 1, 2)) + 1e-6

print("Train mean (RGB):", mean)
print("Train std (RGB):", std)


# --------------------------------------------------
# 2. Load threshold from saved test metrics
# --------------------------------------------------

with open(TEST_METRICS_PATH) as f:
    test_metrics = json.load(f)

best_threshold = test_metrics["best_threshold"]
print("Best threshold:", best_threshold)


# --------------------------------------------------
# 3. Save normalisation + threshold artifact
# --------------------------------------------------

artifact = {
    "mean": mean.tolist(),
    "std": std.tolist(),
    "best_threshold": best_threshold,
    "image_size": IMAGE_SIZE,
    "test_roc_auc": test_metrics.get("roc_auc_sklearn"),
    "test_accuracy": test_metrics.get("test_accuracy_default_metric"),
}

with open(ARTIFACTS_DIR / "preprocessing_config.json", "w") as f:
    json.dump(artifact, f, indent=4)

print(f"\nSaved preprocessing config to {ARTIFACTS_DIR / 'preprocessing_config.json'}")


# --------------------------------------------------
# 4. Copy the trained model weights into the deployment folder
# --------------------------------------------------

model_dest = ARTIFACTS_DIR / "model_weights.pt"
shutil.copy(MODEL_PATH, model_dest)

model_size_mb = model_dest.stat().st_size / (1024 * 1024)
print(f"Copied model weights to {model_dest} ({model_size_mb:.2f} MB)")

print("\nDeployment artifacts ready in:", ARTIFACTS_DIR)
print("Next: put app.py in the 'deployment' folder alongside 'artifacts/', then follow the deployment steps.")