from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
BASE_FOLDER = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_FOLDER / "data" / "metadata" / "quality_metadata.csv"
DEPLOY_DIR = BASE_FOLDER / "deployment" / "artifacts"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SIZE = 128
RANDOM_SEED = 42

# Model definitions
class RealDataCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding="same"), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x).squeeze(1)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels))

    def forward(self, x):
        avg_pool, max_pool = x.mean(dim=(2, 3)), x.amax(dim=(2, 3))
        return x * torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))[:, :, None, None]

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.amax(dim=1, keepdim=True)
        attention_map = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * attention_map, attention_map

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x, spatial_map = self.spatial_attention(x)
        return x, spatial_map

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
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x, _ = self.cbam1(x)
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x, _ = self.cbam2(x)
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x, _ = self.cbam3(x)
        return self.classifier(self.gap(x).flatten(1)).squeeze(1)

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone
        in_features = backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, 1)
        for param in self.backbone.classifier[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x).squeeze(1)

# Source weight files from the training results
MODELS = [
    {
        "key": "cnn",
        "name": "Custom CNN",
        "arch": RealDataCNN,
        "weights": BASE_FOLDER / "models" / "real_data_cnn" / "final_real_data_cnn.pt",
        "metrics_json": BASE_FOLDER / "results" / "real_data_cnn" / "test_metrics.json",
        "gradcam_layer": "features.6",
        "input_size": 128
    },
    {
        "key": "cbam",
        "name": "CBAM-Attention CNN",
        "arch": CBAMAttentionCNN,
        "weights": BASE_FOLDER / "models" / "cbam_attention_cnn" / "final_cbam_attention_cnn.pt",
        "metrics_json": BASE_FOLDER / "results" / "cbam_attention_cnn" / "test_metrics.json",
        "gradcam_layer": "conv3",
        "input_size": 128
    },
    {
        "key": "mobilenet",
        "name": "MobileNetV3 (Transfer Learning)",
        "arch": MobileNetTransfer,
        "weights": BASE_FOLDER / "models" / "transfer_learning_cnn" / "final_transfer_learning_cnn.pt",
        "metrics_json": BASE_FOLDER / "results" / "transfer_learning_cnn" / "test_metrics.json",
        "gradcam_layer": "backbone.features.12.0",
        "input_size": 128
    }
]

# Compute normalisation stats from training images (same split as training)
from sklearn.model_selection import train_test_split
from PIL import Image

metadata = pd.read_csv(METADATA_PATH)
trainable = metadata[(metadata["usable"] == 1) & (metadata["class_name"].isin(["lens", "presumed_non_lens"]))].copy()

print(f"Computing normalisation stats from {len(trainable)} images...")
images = []
labels = []
for _, row in trainable.iterrows():
    try:
        with Image.open(row["file_path"]) as img:
            arr = np.asarray(img.convert("RGB").resize((128, 128)), dtype="float32")
            images.append(arr)
            labels.append(int(row["label"]))
    except Exception:
        continue

X = np.stack(images)
y = np.array(labels)

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)

mean = X_train.mean(axis=(0, 1, 2)).tolist()
std = (X_train.std(axis=(0, 1, 2)) + 1e-6).tolist()
print(f"Mean: {[round(m, 4) for m in mean]}")
print(f"Std:  {[round(s, 4) for s in std]}")

import shutil

# Export each model
for cfg in MODELS:
    print(f"\nExporting {cfg['name']}...")
    if not cfg["weights"].exists():
        print(f"  SKIPPED — weights not found: {cfg['weights']}")
        continue
    out_weights = DEPLOY_DIR / f"{cfg['key']}_weights.pt"
    if cfg["key"] == "mobilenet":
        shutil.copy2(cfg["weights"], out_weights)
        print(f"  Weights copied: {out_weights.name}")
    else:
        model = cfg["arch"]()
        model.load_state_dict(torch.load(cfg["weights"], map_location="cpu"))
        model.eval()
        torch.save(model.state_dict(), out_weights)
        print(f"  Weights saved: {out_weights.name}")

    metrics = {}
    if cfg["metrics_json"].exists():
        with open(cfg["metrics_json"]) as f:
            raw = json.load(f)
        metrics = {
            "test_roc_auc": raw.get("roc_auc_sklearn", raw.get("roc_auc", 0.0)),
            "test_accuracy": raw.get("test_accuracy_default_metric", raw.get("test_accuracy", 0.0)),
            "best_threshold": raw.get("best_threshold", 0.5)
        }

    config = {
        "model_key": cfg["key"],
        "model_name": cfg["name"],
        "input_size": cfg["input_size"],
        "mean": mean,
        "std": std,
        "gradcam_layer": cfg["gradcam_layer"],
        **metrics
    }

    out_config = DEPLOY_DIR / f"{cfg['key']}_config.json"
    with open(out_config, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Config saved: {out_config.name}")

print(f"\nAll artifacts saved to: {DEPLOY_DIR}")
print("\nFiles in deployment/artifacts/:")
for f in sorted(DEPLOY_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")