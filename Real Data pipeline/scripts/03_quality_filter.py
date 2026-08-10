"""
03_quality_filter.py

Adds a real, computed 'usable' flag to the dataset metadata, replacing the
hardcoded usable=1 in 01_prepare_dataset.py. Flags likely blank/off-target
cutouts (common in the 'probable' grading tier) and reports the
lens vs non-lens brightness imbalance so it can be cited/addressed in the
report and checked against Grad-CAM later.

Run this after 01_prepare_dataset.py. It reads the raw image folders
directly (lens, non_lens, probable) so it also works for images not yet
in labels.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt


# ==================================================
# 1. Project paths — EDIT THIS to match your machine
# ==================================================

BASE_FOLDER = Path(
    r"D:\MSc Data and Computational Science"
    r"\Gravitational Lensing\Gravitational_Lensing_Project"
    r"\projects-ahana-vaishnav-ahanabhattacharji-Strong-Gravitational-Lens-Finding-Challenge"
    r"\Real Data pipeline"
)

RAW_FOLDER = BASE_FOLDER / "data" / "raw"
METADATA_FOLDER = BASE_FOLDER / "data" / "metadata"
FIGURE_FOLDER = BASE_FOLDER / "figures"

OUTPUT_METADATA_PATH = METADATA_FOLDER / "quality_metadata.csv"
FLAGGED_GRID_PATH = FIGURE_FOLDER / "flagged_for_review_grid.png"

# Folder name -> (label, class_name). Add/rename to match your folders.
CLASS_FOLDERS = {
    "lens": (1, "lens"),
    "non_lens": (0, "presumed_non_lens"),
    "probable": (1, "probable_lens"),  # not yet split/trained on
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ==================================================
# 2. Quality-control thresholds
#
# These were calibrated by inspecting the lowest-ratio images by eye
# (see the flagged grid this script produces) — re-check them on your
# machine before trusting them blindly.
# ==================================================

MIN_PIXEL_STD = 12.0          # near-flat / no-structure images fall below this
MIN_CENTER_EDGE_RATIO = 0.95  # no source brighter than background at center


# ==================================================
# 3. Helper: compute quality stats for one image
# ==================================================

def center_edge_ratio(image_array: np.ndarray) -> tuple[float, float, float]:
    """Compare central-source brightness to background/edge brightness."""

    height, width, _ = image_array.shape
    center_y, center_x = height // 2, width // 2
    half_box = height // 6

    center = image_array[
        center_y - half_box : center_y + half_box,
        center_x - half_box : center_x + half_box,
    ].astype(float)

    border = int(height * 0.12)

    edge_pixels = np.concatenate([
        image_array[:border, :, :].reshape(-1, 3),
        image_array[-border:, :, :].reshape(-1, 3),
        image_array[:, :border, :].reshape(-1, 3),
        image_array[:, -border:, :].reshape(-1, 3),
    ]).astype(float)

    center_mean = float(center.mean())
    edge_mean = float(edge_pixels.mean())
    ratio = center_mean / (edge_mean + 1e-6)

    return center_mean, edge_mean, ratio


def scan_folder_for_quality(folder: Path, label: int, class_name: str) -> list[dict]:
    if not folder.exists():
        print(f"  (skipping, not found): {folder}")
        return []

    image_paths = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS
    )

    total = len(image_paths)
    print(f"  found {total} candidate files")

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
            if record["pixel_std"] < MIN_PIXEL_STD:
                reasons.append("low_variance")
            if ratio < MIN_CENTER_EDGE_RATIO:
                reasons.append("no_central_source")

            record["usable"] = 0 if reasons else 1
            record["flag_reason"] = ";".join(reasons)

        except (UnidentifiedImageError, OSError) as error:
            record["flag_reason"] = f"unreadable: {error}"

        except Exception as error:
            # Catch-all so one bad file never kills the whole scan.
            record["flag_reason"] = f"unexpected error: {error}"

        records.append(record)

    return records


# ==================================================
# 4. Scan all class folders
# ==================================================

all_records = []

for folder_name, (label, class_name) in CLASS_FOLDERS.items():
    print(f"Scanning {folder_name} ...")
    all_records.extend(
        scan_folder_for_quality(RAW_FOLDER / folder_name, label, class_name)
    )

quality = pd.DataFrame(all_records)
quality.to_csv(OUTPUT_METADATA_PATH, index=False)

print(f"\nSaved quality metadata to:\n{OUTPUT_METADATA_PATH}")


# ==================================================
# 5. Report: how many flagged, and the brightness imbalance
# ==================================================

print("\n=== Usable counts by class ===")
print(quality.groupby("class_name")["usable"].agg(["sum", "count"]))

print("\n=== Flagged (unusable) images ===")
flagged = quality[quality["usable"] == 0]
print(flagged[["image_id", "class_name", "flag_reason"]].to_string(index=False))

print("\n=== Center/edge brightness ratio by class (imbalance check) ===")
print(
    quality.groupby("class_name")["center_edge_ratio"]
    .describe()[["count", "mean", "50%", "min", "max"]]
)


# ==================================================
# 6. Save a grid of flagged images for a manual sanity check
# ==================================================

if not flagged.empty:
    sample = flagged.head(16)
    n = len(sample)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        image = Image.open(row["file_path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"{row['image_id']}\n{row['flag_reason']}", fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.savefig(FLAGGED_GRID_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFlagged-image grid saved to:\n{FLAGGED_GRID_PATH}")
    print("Open this and eyeball it — adjust MIN_PIXEL_STD / MIN_CENTER_EDGE_RATIO if it's too strict or too loose.")