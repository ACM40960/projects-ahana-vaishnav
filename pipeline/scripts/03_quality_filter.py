from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

# Go back to the main project folder from the scripts folder
BASE_FOLDER = Path(__file__).resolve().parent.parent

# Main folders
RAW_FOLDER = BASE_FOLDER / "data" / "raw"
METADATA_FOLDER = BASE_FOLDER / "data" / "metadata"
FIGURE_FOLDER = BASE_FOLDER / "figures"

# Output files
OUTPUT_METADATA_PATH = METADATA_FOLDER / "quality_metadata.csv"
FLAGGED_GRID_PATH = FIGURE_FOLDER / "flagged_for_review_grid.png"

# Folders to check and their labels
CLASS_FOLDERS = {"lens": (1, "lens"),"non_lens": (0, "presumed_non_lens"),"probable": (1, "probable_lens"),}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# These values were chosen after checking a few bad-looking images by eye
MIN_PIXEL_STD = 12.0
MIN_CENTER_EDGE_RATIO = 0.95


def center_edge_ratio(image_array: np.ndarray) -> tuple[float, float, float]:
    # Compare the centre brightness with the image edges
    height, width, _ = image_array.shape
    center_y, center_x = height // 2, width // 2
    half_box = height // 6
    center = image_array[center_y - half_box:center_y + half_box,center_x - half_box:center_x + half_box].astype(float)
    border = int(height * 0.12)

    edge_pixels = np.concatenate([image_array[:border, :, :].reshape(-1, 3),image_array[-border:, :, :].reshape(-1, 3),
        image_array[:, :border, :].reshape(-1, 3),
        image_array[:, -border:, :].reshape(-1, 3),]).astype(float)

    center_mean = float(center.mean())
    edge_mean = float(edge_pixels.mean())
    return center_mean, edge_mean, center_mean / (edge_mean + 1e-6)


def scan_folder_for_quality(folder: Path, label: int, class_name: str) -> list[dict]:
    # Scan one image folder and mark images that look unusable
    if not folder.exists():
        print(f"  (skipping, not found): {folder}")
        return []

    image_paths = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    )

    total = len(image_paths)
    print(f"  found {total} files")

    records = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"  [{index}/{total}] {image_path.name}", flush=True)

        record = {
            "image_id": image_path.stem,
            "file_path": str(image_path),
            "label": label,
            "class_name": class_name,
            "readable": 0,
            "mean_pixel": None,
            "pixel_std": None,
            "center_mean": None,
            "edge_mean": None,
            "center_edge_ratio": None,
            "usable": 0,
            "flag_reason": "",
        }

        try:
            with Image.open(image_path) as image:
                image.load()
                image_array = np.asarray(image.convert("RGB"))
            record["readable"] = 1
            record["mean_pixel"] = float(image_array.mean())
            record["pixel_std"] = float(image_array.std())
            center_mean, edge_mean, ratio = center_edge_ratio(image_array)
            record["center_mean"] = center_mean
            record["edge_mean"] = edge_mean
            record["center_edge_ratio"] = ratio
            reasons = []

            # Very low variation usually means blank or almost blank images
            if record["pixel_std"] < MIN_PIXEL_STD:
                reasons.append("low_variance")

            # If the centre is not brighter than the edge, the object may not be centred
            if ratio < MIN_CENTER_EDGE_RATIO:
                reasons.append("no_central_source")

            record["usable"] = 0 if reasons else 1
            record["flag_reason"] = ";".join(reasons)

        except (UnidentifiedImageError, OSError) as error:
            record["flag_reason"] = f"unreadable: {error}"

        except Exception as error:
            record["flag_reason"] = f"unexpected error: {error}"

        records.append(record)
    return records

# Run the quality check for each class folder
all_records = []

for folder_name, (label, class_name) in CLASS_FOLDERS.items():
    print(f"Scanning {folder_name} ...")
    all_records.extend(scan_folder_for_quality(RAW_FOLDER / folder_name,label,class_name))

# Save the full quality-check metadata
quality = pd.DataFrame(all_records)
quality.to_csv(OUTPUT_METADATA_PATH, index=False)
print(f"\nSaved to: {OUTPUT_METADATA_PATH}")

# Basic summary of how many images passed the checks
print("\n=== Usable counts by class ===")
print(quality.groupby("class_name")["usable"].agg(["sum", "count"]))

# Show the images that were flagged
print("\nFlagged images")
flagged = quality[quality["usable"] == 0]
print(flagged[["image_id", "class_name", "flag_reason"]].to_string(index=False))

# Check whether the centre/edge ratio looks different across classes
print("\nCenter/edge brightness ratio by class")
print(quality.groupby("class_name")["center_edge_ratio"].describe()[["count", "mean", "50%", "min", "max"]])

# Make a grid of flagged images so they can be checked manually
if not flagged.empty:
    sample = flagged.head(16)
    n = len(sample)
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows,4,figsize=(12, 3 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, row) in zip(axes, sample.iterrows()):
        ax.imshow(Image.open(row["file_path"]).convert("RGB"))
        ax.set_title(f"{row['image_id']}\n{row['flag_reason']}",fontsize=8)
        ax.axis("off")

    # Hide empty plot spaces
    for ax in axes[n:]:ax.axis("off")
    plt.tight_layout()
    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.savefig(FLAGGED_GRID_PATH,dpi=150,bbox_inches="tight")