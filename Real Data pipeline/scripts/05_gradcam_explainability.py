from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Go back to the main project folder from the scripts folder
BASE_FOLDER = Path(__file__).resolve().parent.parent
# Input/output paths
METADATA_PATH = BASE_FOLDER / "data" / "metadata" / "quality_metadata.csv"
RESULTS_DIR = BASE_FOLDER / "results" / "real_data_cnn"
MODEL_PATH = BASE_FOLDER / "models" / "real_data_cnn" / "best_real_data_cnn.pt"
IMAGE_SIZE = 128
RANDOM_SEED = 42

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the same metadata used during training
metadata = pd.read_csv(METADATA_PATH)

# Use the same trainable classes as the CNN training script
trainable = metadata[(metadata["usable"] == 1) & (metadata["class_name"].isin(["lens", "presumed_non_lens"]))].copy()

# Keep probable lenses separate for an extra visual check
held_out_probable = metadata[(metadata["usable"] == 1) & (metadata["class_name"] == "probable_lens")].copy()

def load_image_array(file_path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    # Load and resize one image
    with Image.open(file_path) as image:
        image = image.convert("RGB").resize((size, size))
        return np.asarray(image, dtype="float32")

# Load images again so the split can be recreated
images, labels, rows = [], [], []

for _, row in trainable.iterrows():
    try:
        images.append(load_image_array(row["file_path"]))
        labels.append(int(row["label"]))
        rows.append(row)
    except Exception:
        # Skip unreadable files instead of stopping the script
        continue

X = np.stack(images, axis=0)
y = np.array(labels, dtype="int64")
rows_df = pd.DataFrame(rows).reset_index(drop=True)

# Recreate the same train/validation/test split as the training script
train_idx, temp_idx = train_test_split(np.arange(len(X)),test_size=0.30,random_state=RANDOM_SEED,stratify=y)
val_idx, test_idx = train_test_split(temp_idx,test_size=0.50,random_state=RANDOM_SEED,stratify=y[temp_idx])

X_train = X[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]
test_rows = rows_df.iloc[test_idx].reset_index(drop=True)

# Use training-set statistics only, same as the model training script
mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6

class RealDataCNN(nn.Module):
    # Same CNN structure used during training
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding="same"),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(64, 32),nn.ReLU(),nn.Dropout(0.5),nn.Linear(32, 1),)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x).squeeze(1)

# Load the trained model weights
model = RealDataCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Use the last convolution layer for Grad-CAM
TARGET_LAYER = model.features[6]

activations = {}
gradients = {}

def forward_hook(module, input, output):
    # Save feature maps from the forward pass
    activations["value"] = output.detach()

def backward_hook(module, grad_input, grad_output):
    # Save gradients from the backward pass
    gradients["value"] = grad_output[0].detach()

# Register hooks so Grad-CAM can access activations and gradients
TARGET_LAYER.register_forward_hook(forward_hook)
TARGET_LAYER.register_full_backward_hook(backward_hook)

def to_normalised_tensor(image_array: np.ndarray) -> torch.Tensor:
    # Convert image array into the same tensor format used during training
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
    tensor = (tensor - torch.tensor(mean.reshape(3, 1, 1))) / torch.tensor(std.reshape(3, 1, 1))
    return tensor.unsqueeze(0).to(DEVICE)

def grad_cam(image_array: np.ndarray):
    # Get model probability and Grad-CAM heatmap for one image
    input_tensor = to_normalised_tensor(image_array)
    input_tensor.requires_grad_(True)
    model.zero_grad()
    logit = model(input_tensor)
    probability = torch.sigmoid(logit).item()
    logit.backward()
    acts = activations["value"][0]
    grads = gradients["value"][0]

    # Average gradients are used as weights for the feature maps
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam.cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize the heatmap back to the image size
    cam_image = Image.fromarray((cam * 255).astype("uint8")).resize((IMAGE_SIZE, IMAGE_SIZE),resample=Image.BILINEAR)

    cam_resized = np.asarray(cam_image, dtype="float32") / 255.0
    return probability, cam_resized

def overlay_heatmap(ax, image_array: np.ndarray, cam: np.ndarray, title: str):
    # Show image with Grad-CAM heatmap on top
    ax.imshow(image_array.astype("uint8"))
    ax.imshow(cam, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=8)
    ax.axis("off")

# Load the threshold chosen during validation
BEST_THRESHOLD_PATH = RESULTS_DIR / "test_metrics.json"

with open(BEST_THRESHOLD_PATH) as f:
    saved_metrics = json.load(f)

best_threshold = saved_metrics["best_threshold"]

# Get predictions for the test set
test_probs = []

for image_array in X_test:
    with torch.no_grad():
        input_tensor = to_normalised_tensor(image_array)
        logit = model(input_tensor)
        test_probs.append(torch.sigmoid(logit).item())

test_probs = np.array(test_probs)
test_preds = (test_probs >= best_threshold).astype(int)

# Add prediction results back to the test metadata
test_rows = test_rows.copy()
test_rows["true_label"] = y_test
test_rows["predicted_probability"] = test_probs
test_rows["predicted_label"] = test_preds
test_rows["correct"] = test_rows["true_label"] == test_rows["predicted_label"]

# Pick a few correct and wrong examples for the Grad-CAM grid
correct_lens = test_rows[(test_rows["true_label"] == 1) & (test_rows["correct"])].sample(n=min(3, (test_rows["true_label"].eq(1) & test_rows["correct"]).sum()),random_state=RANDOM_SEED)
correct_nonlens = test_rows[(test_rows["true_label"] == 0) & (test_rows["correct"])].sample(n=min(3, (test_rows["true_label"].eq(0) & test_rows["correct"]).sum()),random_state=RANDOM_SEED)

misclassified = test_rows[~test_rows["correct"]].head(6)
gradcam_selection = pd.concat([correct_lens, correct_nonlens, misclassified]).reset_index(drop=True)

# Create Grad-CAM grid for test examples
n = len(gradcam_selection)
cols = 4
rows_n = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows_n,cols,figsize=(4 * cols, 4 * rows_n))
axes = np.atleast_1d(axes).flatten()

for ax, (_, row) in zip(axes, gradcam_selection.iterrows()):
    image_array = load_image_array(row["file_path"])
    prob, cam = grad_cam(image_array)
    true_name = "lens" if row["true_label"] == 1 else "non_lens"
    pred_name = "lens" if row["predicted_label"] == 1 else "non_lens"
    status = "CORRECT" if row["correct"] else "WRONG"
    overlay_heatmap(ax,image_array,cam,f"{row['image_id']}\ntrue={true_name} pred={pred_name} ({prob:.2f})\n{status}",)

for ax in axes[n:]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "gradcam_grid.png",dpi=150,bbox_inches="tight")
plt.close()

print(f"Saved Grad-CAM grid to {RESULTS_DIR / 'gradcam_grid.png'}")

# Run Grad-CAM on probable lenses that were not used for main training/testing
if len(held_out_probable) > 0:
    probable_sample = held_out_probable.sample(n=min(8, len(held_out_probable)),random_state=RANDOM_SEED).reset_index(drop=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, probable_sample.iterrows()):
        image_array = load_image_array(row["file_path"])
        prob, cam = grad_cam(image_array)
        overlay_heatmap(ax,image_array,cam,f"{row['image_id']}\npred_prob={prob:.2f}")

    for ax in axes[len(probable_sample):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gradcam_probable_grid.png",dpi=150,bbox_inches="tight")
    plt.close()
    print(f"Saved probable-tier Grad-CAM grid to {RESULTS_DIR / 'gradcam_probable_grid.png'}")

# Collect false positives and false negatives for manual checking
false_positives = test_rows[(test_rows["true_label"] == 0) & (test_rows["predicted_label"] == 1)]
false_negatives = test_rows[(test_rows["true_label"] == 1) & (test_rows["predicted_label"] == 0)]
fp_fn_sample = pd.concat([false_positives.head(8),false_negatives.head(8),]).reset_index(drop=True)

# Save a plain image gallery of the model mistakes
if len(fp_fn_sample) > 0:
    n = len(fp_fn_sample)
    cols = 4
    rows_n = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows_n,cols,figsize=(4 * cols, 4 * rows_n))
    axes = np.atleast_1d(axes).flatten()

    for ax, (_, row) in zip(axes, fp_fn_sample.iterrows()):
        image = Image.open(row["file_path"]).convert("RGB")
        error_type = (
            "False Positive"
            if row["true_label"] == 0
            else "False Negative"
        )
        ax.imshow(image)
        ax.set_title(f"{row['image_id']}\n{error_type}\nprob={row['predicted_probability']:.2f}",fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "false_positive_negative_gallery.png",dpi=150,bbox_inches="tight")
    plt.close()
    print(f"Saved FP/FN gallery to {RESULTS_DIR / 'false_positive_negative_gallery.png'}")

print(f"\nFalse positives: {len(false_positives)}, False negatives: {len(false_negatives)}")

# Check if model probability is related to the brightness ratio
plt.figure(figsize=(7, 6))
colors = test_rows["true_label"].map({0: "tab:blue",1: "tab:orange"})
plt.scatter(test_rows["center_edge_ratio"],test_rows["predicted_probability"],c=colors,alpha=0.6,edgecolors="none",)
plt.xlabel("Center / edge brightness ratio")
plt.ylabel("Predicted probability (lens)")
plt.title("Predicted probability vs. brightness ratio (test set)\n""blue=non_lens, orange=lens")
plt.axhline(best_threshold,color="gray",linestyle="--",linewidth=1,label=f"decision threshold ({best_threshold:.2f})")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "brightness_vs_prediction.png",dpi=150,bbox_inches="tight")
plt.close()

# Pearson correlation gives a simple shortcut-learning check
correlation, p_value = stats.pearsonr(test_rows["center_edge_ratio"],test_rows["predicted_probability"])

print(
    "\nPearson correlation "
    f"(predicted probability vs. brightness ratio): "
    f"r={correlation:.3f}, p={p_value:.4g}"
)

if abs(correlation) > 0.4:
    print("NOTE: Moderate-to-strong correlation — the model's confidence may be")
    print("partly tracking overall brightness rather than lens-specific structure.")
    print("Discuss this explicitly in your Limitations section.")
else:
    print("Correlation is weak — little evidence the model is using brightness as")
    print("a simple shortcut, though Grad-CAM heatmaps should still be reviewed by eye.")

print(
    f"\nSaved brightness-vs-prediction plot to "
    f"{RESULTS_DIR / 'brightness_vs_prediction.png'}"
)