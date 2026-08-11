from pathlib import Path
import hashlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


# ==================================================
# 1. Project paths
# ==================================================

# Relative path: this script lives in "Real Data pipeline/scripts/",
# so its parent's parent is "Real Data pipeline/" itself. Works on any
# machine the repo is cloned to, no hardcoded drive/user path needed.
BASE_FOLDER = Path(__file__).resolve().parent.parent

# Change "lens" to "confident" here if that is your folder name.
LENS_FOLDER = BASE_FOLDER / "data" / "raw" / "lens"
NON_LENS_FOLDER = BASE_FOLDER / "data" / "raw" / "non_lens"

METADATA_FOLDER = BASE_FOLDER / "data" / "metadata"
FIGURE_FOLDER = BASE_FOLDER / "figures"

LABELS_PATH = METADATA_FOLDER / "labels.csv"
DUPLICATES_PATH = METADATA_FOLDER / "duplicate_images.csv"
GRID_PATH = FIGURE_FOLDER / "dataset_sample_grid.png"

METADATA_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. Settings
# ==================================================

RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(RANDOM_SEED)


# ==================================================
# 3. Helper functions
# ==================================================

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash for duplicate detection."""

    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(8192)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def scan_image_folder(
    folder: Path,
    label: int,
    class_name: str,
) -> list[dict]:
    """Read and validate all images in one class folder."""

    if not folder.exists():
        raise FileNotFoundError(
            f"Image folder was not found:\n{folder}"
        )

    image_paths = sorted(
        path
        for path in folder.rglob("*")
        if path.suffix.lower() in VALID_EXTENSIONS
    )

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
            record["usable"] = 0
            record["notes"] = f"Unreadable image: {error}"

        records.append(record)

    return records


# ==================================================
# 4. Scan both classes
# ==================================================

records = []

records.extend(
    scan_image_folder(
        folder=LENS_FOLDER,
        label=1,
        class_name="lens",
    )
)

records.extend(
    scan_image_folder(
        folder=NON_LENS_FOLDER,
        label=0,
        class_name="presumed_non_lens",
    )
)

metadata = pd.DataFrame(records)


# ==================================================
# 5. Detect duplicates
# ==================================================

duplicate_mask = metadata.duplicated(
    subset="file_hash",
    keep=False,
) & metadata["file_hash"].notna()

duplicates = metadata[duplicate_mask].copy()

duplicates.to_csv(
    DUPLICATES_PATH,
    index=False,
)


# ==================================================
# 6. Save labels
# ==================================================

metadata.to_csv(
    LABELS_PATH,
    index=False,
)


# ==================================================
# 7. Print audit results
# ==================================================

print("\nDataset summary:")
print(metadata["class_name"].value_counts())

print("\nReadable images:")
print(metadata.groupby("class_name")["readable"].sum())

print("\nImage dimensions:")
print(
    metadata.groupby(
        ["class_name", "width", "height"]
    )
    .size()
    .reset_index(name="count")
)

print(f"\nDuplicate records found: {len(duplicates)}")

if not duplicates.empty:
    print(
        duplicates[
            [
                "image_id",
                "class_name",
                "file_path",
                "file_hash",
            ]
        ].to_string(index=False)
    )

print(f"\nMetadata saved to:\n{LABELS_PATH}")
print(f"Duplicate report saved to:\n{DUPLICATES_PATH}")


# ==================================================
# 8. Create a sample image grid
# ==================================================

usable_metadata = metadata[
    (metadata["readable"] == 1)
    & (metadata["usable"] == 1)
].copy()

lens_sample = usable_metadata[
    usable_metadata["label"] == 1
].sample(
    n=min(8, sum(usable_metadata["label"] == 1)),
    random_state=RANDOM_SEED,
)

non_lens_sample = usable_metadata[
    usable_metadata["label"] == 0
].sample(
    n=min(8, sum(usable_metadata["label"] == 0)),
    random_state=RANDOM_SEED,
)

sample = pd.concat(
    [lens_sample, non_lens_sample],
    ignore_index=True,
)

fig, axes = plt.subplots(
    4,
    4,
    figsize=(12, 12),
)

axes = axes.flatten()

for ax, (_, row) in zip(axes, sample.iterrows()):
    image = Image.open(row["file_path"]).convert("RGB")

    ax.imshow(image)
    ax.set_title(
        f"{row['image_id']}\n{row['class_name']}",
        fontsize=8,
    )
    ax.axis("off")

for ax in axes[len(sample):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    GRID_PATH,
    dpi=200,
    bbox_inches="tight",
)
plt.show()

print(f"\nSample grid saved to:\n{GRID_PATH}")