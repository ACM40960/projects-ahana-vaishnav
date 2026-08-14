from pathlib import Path
import hashlib
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


# Go back to the main project folder from the scripts folder
BASE_FOLDER = Path(__file__).resolve().parent.parent

# Image folders
LENS_FOLDER = BASE_FOLDER / "data" / "raw" / "lens"
NON_LENS_FOLDER = BASE_FOLDER / "data" / "raw" / "non_lens"

# Output folders
METADATA_FOLDER = BASE_FOLDER / "data" / "metadata"
FIGURE_FOLDER = BASE_FOLDER / "figures"

# Output files
LABELS_PATH = METADATA_FOLDER / "labels.csv"
DUPLICATES_PATH = METADATA_FOLDER / "duplicate_images.csv"
GRID_PATH = FIGURE_FOLDER / "dataset_sample_grid.png"

METADATA_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(RANDOM_SEED)

def calculate_file_hash(file_path: Path) -> str:
    # Used to find exact duplicate image files
    hasher = hashlib.sha256()
    with file_path.open("rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def scan_image_folder(folder: Path, label: int, class_name: str) -> list[dict]:
    # Read all images in a folder and collect basic information about them
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found:\n{folder}")
    image_paths = sorted(p for p in folder.rglob("*")if p.suffix.lower() in VALID_EXTENSIONS)
    records = []

    for image_path in image_paths:
        record = {
            "image_id": image_path.stem,
            "file_path": str(image_path),
            "label": label,
            "class_name": class_name,
            "readable": 0,
            "width": None,
            "height": None,
            "channels": None,
            "mean_pixel": None,
            "pixel_std": None,
            "file_hash": None,
            "usable": 1,
            "notes": "",
        }

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_array = np.asarray(image)
                record["readable"] = 1
                record["width"] = image.width
                record["height"] = image.height
                record["channels"] = image_array.shape[2]
                record["mean_pixel"] = float(image_array.mean())
                record["pixel_std"] = float(image_array.std())
                record["file_hash"] = calculate_file_hash(image_path)

        except (UnidentifiedImageError, OSError) as error:
            # Mark bad images instead of stopping the whole script
            record["usable"] = 0
            record["notes"] = f"Unreadable image: {error}"
        records.append(record)
    return records

# Scan both classes
records = []
records.extend(scan_image_folder(LENS_FOLDER,label=1,class_name="lens"))
records.extend(scan_image_folder(NON_LENS_FOLDER,label=0,class_name="presumed_non_lens"))
metadata = pd.DataFrame(records)

# Check for exact duplicate files using image hashes
duplicates = metadata[metadata.duplicated(subset="file_hash", keep=False) & metadata["file_hash"].notna()].copy()
duplicates.to_csv(DUPLICATES_PATH, index=False)
metadata.to_csv(LABELS_PATH, index=False)

# Basic dataset checks
print("\nDataset summary:")
print(metadata["class_name"].value_counts())

print("\nReadable images:")
print(metadata.groupby("class_name")["readable"].sum())

print("\nImage dimensions:")
print(metadata.groupby(["class_name", "width", "height"]).size().reset_index(name="count"))

print(f"\nDuplicates found: {len(duplicates)}")

if not duplicates.empty:
    print(duplicates[["image_id", "class_name", "file_path", "file_hash"]].to_string(index=False))
print(f"\nLabels saved to: {LABELS_PATH}")
print(f"Duplicates saved to: {DUPLICATES_PATH}")

# Keep only images that can actually be used
usable = metadata[(metadata["readable"] == 1) & (metadata["usable"] == 1)].copy()\

# Take a small random sample from both classes for visual checking
lens_sample = usable[usable["label"] == 1].sample(n=min(8, (usable["label"] == 1).sum()),random_state=RANDOM_SEED)
non_lens_sample = usable[usable["label"] == 0].sample(n=min(8, (usable["label"] == 0).sum()),random_state=RANDOM_SEED)
sample = pd.concat([lens_sample, non_lens_sample],ignore_index=True)

# Make a quick grid so the dataset can be inspected visually
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

for ax, (_, row) in zip(axes, sample.iterrows()):
    ax.imshow(Image.open(row["file_path"]).convert("RGB"))
    ax.set_title(f"{row['image_id']}\n{row['class_name']}",fontsize=8)
    ax.axis("off")

# Turn off unused grid spaces if there are fewer than 16 images
for ax in axes[len(sample):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(GRID_PATH, dpi=200, bbox_inches="tight")
plt.show()
print(f"\nSample grid saved to: {GRID_PATH}")