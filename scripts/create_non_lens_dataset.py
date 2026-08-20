from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
import time

import h5py
import numpy as np
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError


# Go back to the main project folder from the scripts folder
BASE_FOLDER = Path(__file__).resolve().parent.parent

# Input files
H5_PATH = BASE_FOLDER / "data" / "raw" / "Galaxy10_DECals.h5"
LENSCAT_PATH = BASE_FOLDER / "data" / "catalog.csv"

# Output folders/files
OUTPUT_IMAGE_FOLDER = BASE_FOLDER / "data" / "raw" / "non_lens"
OUTPUT_METADATA_FOLDER = BASE_FOLDER / "data" / "metadata"
OUTPUT_METADATA_PATH = OUTPUT_METADATA_FOLDER / "galaxy10_non_lens_download_log.csv"

# Make output folders if they do not already exist
OUTPUT_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_METADATA_FOLDER.mkdir(parents=True, exist_ok=True)

# Match the number of confident lens images
NUMBER_TO_DOWNLOAD = 2038

# Sample more than needed because some objects are removed after Lenscat filtering
CANDIDATE_POOL_SIZE = 6000

RANDOM_SEED = 25206621

# Use the same cutout settings as the lens images
IMAGE_SIZE = 128
PIXEL_SCALE = 0.262
SURVEY_LAYER = "ls-dr10"

# Avoid selecting objects too close to known lens candidates
MINIMUM_DISTANCE_FROM_LENSCAT_ARCSEC = 30.0

# Save progress every 100 images in case the script stops
CHECKPOINT_EVERY = 100

# Classes 0 and 1 are excluded because disturbed/merging galaxies may look too lens-like
ALLOWED_CLASSES = [2, 3, 4, 5, 6, 7, 8, 9]

CLASS_NAMES = {
    0: "disturbed",
    1: "merging",
    2: "round_smooth",
    3: "in_between_round_smooth",
    4: "cigar_shaped_smooth",
    5: "barred_spiral",
    6: "unbarred_tight_spiral",
    7: "unbarred_loose_spiral",
    8: "edge_on_without_bulge",
    9: "edge_on_with_bulge",
}

# Check that the required input files are present
if not H5_PATH.exists():
    raise FileNotFoundError(f"Galaxy10 H5 file not found:\n{H5_PATH}")

if not LENSCAT_PATH.exists():
    raise FileNotFoundError(f"Lenscat catalog.csv not found:\n{LENSCAT_PATH}")


def build_cutout_url(ra: float, dec: float) -> str:
    # Build the Legacy Survey image cutout URL for one object
    params = {"ra": ra,"dec": dec,"layer": SURVEY_LAYER,"size": IMAGE_SIZE,"pixscale": PIXEL_SCALE,}
    return "https://www.legacysurvey.org/viewer/jpeg-cutout?" + urlencode(params)


def minimum_separation_arcsec(object_ra, object_dec, catalogue_ra, catalogue_dec) -> float:
    # Find how close this object is to the nearest known Lenscat object
    ra1, dec1 = np.radians(object_ra), np.radians(object_dec)
    ra2, dec2 = np.radians(catalogue_ra), np.radians(catalogue_dec)
    cosine_angle = (np.sin(dec1) * np.sin(dec2)+ np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)).min()) * 3600.0)


# Load Lenscat so known lens positions can be excluded from the non-lens set
lenscat = pd.read_csv(LENSCAT_PATH)
lenscat.columns = lenscat.columns.str.strip()
missing_columns = {"RA [deg]", "DEC [deg]"}.difference(lenscat.columns)

if missing_columns:
    raise ValueError(f"Lenscat is missing columns: {sorted(missing_columns)}")

# Keep only rows with valid sky coordinates
lenscat = lenscat.dropna(subset=["RA [deg]", "DEC [deg]"]).copy()
lenscat_ra = lenscat["RA [deg]"].to_numpy(dtype=float)
lenscat_dec = lenscat["DEC [deg]"].to_numpy(dtype=float)

print(f"Lenscat coordinates loaded: {len(lenscat):,}")


# Load Galaxy10 labels and coordinates from the H5 file
with h5py.File(H5_PATH, "r") as h5_file:
    labels = np.asarray(h5_file["ans"]).reshape(-1)
    ra_values = np.asarray(h5_file["ra"]).reshape(-1)
    dec_values = np.asarray(h5_file["dec"]).reshape(-1)
    redshift_values = np.asarray(h5_file["redshift"]).reshape(-1)
    pixel_scale_values = np.asarray(h5_file["pxscale"]).reshape(-1)

print(f"Galaxy10 objects loaded: {len(labels):,}")
# Quick check of how many objects are in each Galaxy10 class
for cls, count in zip(*np.unique(labels, return_counts=True)):
    print(f"  Class {int(cls)}: {CLASS_NAMES.get(int(cls), 'unknown')} = {count}")


# Keep only allowed Galaxy10 classes with valid coordinates
valid_indices = np.where(np.isin(labels, ALLOWED_CLASSES)& np.isfinite(ra_values)& np.isfinite(dec_values))[0]

rng = np.random.default_rng(RANDOM_SEED)

# Randomly choose a candidate pool before removing objects near known lenses
candidate_indices = rng.choice(valid_indices,size=min(CANDIDATE_POOL_SIZE, len(valid_indices)),replace=False)

candidate_records = []

print("\nChecking distance from known Lenscat objects...")

for position, galaxy10_index in enumerate(candidate_indices, start=1):
    ra = float(ra_values[galaxy10_index])
    dec = float(dec_values[galaxy10_index])
    galaxy_class = int(labels[galaxy10_index])

    nearest_distance = minimum_separation_arcsec(ra,dec,lenscat_ra,lenscat_dec)

    candidate_records.append({
        "galaxy10_index": int(galaxy10_index),
        "ra": ra,
        "dec": dec,
        "galaxy10_class": galaxy_class,
        "morphology": CLASS_NAMES[galaxy_class],
        "redshift": float(redshift_values[galaxy10_index]),
        "original_pxscale": float(pixel_scale_values[galaxy10_index]),
        "nearest_lenscat_arcsec": nearest_distance,
    })

    if position % 500 == 0:
        print(f"  Checked {position}/{len(candidate_indices)}")

candidates = pd.DataFrame(candidate_records)

# Remove objects that are too close to known lens candidates
candidates = candidates[
    candidates["nearest_lenscat_arcsec"] > MINIMUM_DISTANCE_FROM_LENSCAT_ARCSEC
].copy()

# Remove duplicate coordinates and keep only the required number
candidates = (
    candidates
    .drop_duplicates(subset=["ra", "dec"])
    .head(NUMBER_TO_DOWNLOAD)
    .reset_index(drop=True)
)

if len(candidates) < NUMBER_TO_DOWNLOAD:
    print(
        f"Warning: only {len(candidates)} candidates after filtering. "
        "Increase CANDIDATE_POOL_SIZE if needed."
    )

# Add labels and image IDs for the final non-lens dataset
candidates["image_id"] = [f"nonlens_{i + 1:04d}" for i in range(len(candidates))]
candidates["label"] = 0
candidates["class_name"] = "presumed_non_lens"

print(f"Non-lens candidates selected: {len(candidates)}")

# Reuse one session for all image downloads
session = requests.Session()
session.headers.update({"User-Agent": "University gravitational lens classification project"})
download_records = []

for index, row in candidates.iterrows():
    image_id = row["image_id"]
    cutout_url = build_cutout_url(float(row["ra"]), float(row["dec"]))
    output_path = OUTPUT_IMAGE_FOLDER / f"{image_id}.jpg"

    # Start with the candidate information, then add download details
    record = row.to_dict()
    record.update({
        "cutout_url": cutout_url,
        "file_path": str(output_path),
        "survey_layer": SURVEY_LAYER,
        "cutout_size": IMAGE_SIZE,
        "cutout_pxscale": PIXEL_SCALE,
        "download_status": "failed",
        "http_status": None,
        "width": None,
        "height": None,
        "mean_pixel": None,
        "pixel_std": None,
        "usable": "",
        "suspected_lens": "",
        "notes": "",
        "error": None,
    })
    # Skip files that were already downloaded in a previous run
    if output_path.exists() and output_path.stat().st_size > 0:
        record["download_status"] = "already_downloaded"
        download_records.append(record)

        if (index + 1) % 50 == 0:
            print(f"[{index + 1}/{len(candidates)}] {image_id} — skipped")
        continue
    print(f"[{index + 1}/{len(candidates)}] Downloading {image_id}")

    try:
        response = session.get(cutout_url, timeout=30)
        record["http_status"] = response.status_code
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.load()
        image = image.convert("RGB")
        image_array = np.asarray(image)

        # Save some simple image statistics for later checks
        record["width"] = image.width
        record["height"] = image.height
        record["mean_pixel"] = float(image_array.mean())
        record["pixel_std"] = float(image_array.std())
        image.save(output_path, format="JPEG", quality=95)
        record["download_status"] = "success"
        print(f"  Saved: {image.size}")

    except requests.RequestException as error:
        record["error"] = f"Request error: {error}"
        print(f"  Failed: {error}")

    except UnidentifiedImageError:
        record["error"] = "Downloaded response was not a valid image."
        print("  Failed: not an image")

    except Exception as error:
        record["error"] = str(error)
        print(f"  Failed: {error}")
    download_records.append(record)

    # Small delay so the server is not hit too aggressively
    time.sleep(0.5)

    # Save a checkpoint every few downloads
    if (index + 1) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(download_records).to_csv(OUTPUT_METADATA_PATH, index=False)

metadata = pd.DataFrame(download_records)
metadata.to_csv(OUTPUT_METADATA_PATH, index=False)
successful = metadata["download_status"].isin(["success", "already_downloaded"]).sum()

print(f"\nDone. {successful} successful / {len(metadata) - successful} failed")
print(f"Images: {OUTPUT_IMAGE_FOLDER}")
print(f"Metadata: {OUTPUT_METADATA_PATH}")
print("\nMorphology distribution:")
print(metadata["morphology"].value_counts())
