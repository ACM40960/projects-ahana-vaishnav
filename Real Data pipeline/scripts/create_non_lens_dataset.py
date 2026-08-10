from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
import time

import h5py
import numpy as np
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError


# ==================================================
# 1. Paths
# ==================================================

BASE_FOLDER = Path(
    r"D:\MSc Data and Computational Science"
    r"\Gravitational Lensing\Gravitational_Lensing_Project"
    r"\projects-ahana-vaishnav-ahanabhattacharji-Strong-Gravitational-Lens-Finding-Challenge"
    r"\Real Data pipeline"
)

# Galaxy10 file lives outside the repo folder — set directly.
H5_PATH = Path(r"D:\Galaxy10_DECals.h5")

LENSCAT_PATH = BASE_FOLDER / "data" / "catalog.csv"

OUTPUT_IMAGE_FOLDER = (
    BASE_FOLDER
    / "data"
    / "raw"
    / "non_lens"
)

OUTPUT_METADATA_FOLDER = (
    BASE_FOLDER
    / "data"
    / "metadata"
)

OUTPUT_METADATA_PATH = (
    OUTPUT_METADATA_FOLDER
    / "galaxy10_non_lens_download_log.csv"
)

OUTPUT_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_METADATA_FOLDER.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. Settings
# ==================================================

# Matched to the expanded confident-lens download (all_data_extraction.py).
NUMBER_TO_DOWNLOAD = 2038

# Sample a larger candidate pool before excluding known lenses —
# raised proportionally so the 30-arcsec Lenscat exclusion still
# leaves enough candidates after filtering.
CANDIDATE_POOL_SIZE = 6000

RANDOM_SEED = 42

IMAGE_SIZE = 128
PIXEL_SCALE = 0.262

# This must match the setting used for your positive lens images.
SURVEY_LAYER = "ls-dr10"

# Start with ordinary galaxy morphologies.
# Disturbed and merging galaxies can be added later as hard negatives.
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

# Exclude Galaxy10 objects close to anything listed in Lenscat.
MINIMUM_DISTANCE_FROM_LENSCAT_ARCSEC = 30.0

# How often to write a metadata checkpoint (in images), so an
# interruption partway through doesn't lose all progress.
CHECKPOINT_EVERY = 100


# ==================================================
# 3. Validate input files
# ==================================================

if not H5_PATH.exists():
    raise FileNotFoundError(
        f"Galaxy10 H5 file not found:\n{H5_PATH}"
    )

if not LENSCAT_PATH.exists():
    raise FileNotFoundError(
        f"Lenscat catalog.csv not found:\n{LENSCAT_PATH}"
    )


# ==================================================
# 4. Helper functions
# ==================================================

def build_cutout_url(
    ra: float,
    dec: float,
) -> str:
    """Build a Legacy Survey JPEG cutout URL."""

    parameters = {
        "ra": ra,
        "dec": dec,
        "layer": SURVEY_LAYER,
        "size": IMAGE_SIZE,
        "pixscale": PIXEL_SCALE,
    }

    return (
        "https://www.legacysurvey.org/viewer/jpeg-cutout?"
        + urlencode(parameters)
    )


def minimum_separation_arcsec(
    object_ra: float,
    object_dec: float,
    catalogue_ra: np.ndarray,
    catalogue_dec: np.ndarray,
) -> float:
    """
    Return the minimum angular separation between one object
    and all Lenscat coordinates, measured in arcseconds.
    """

    ra1 = np.radians(object_ra)
    dec1 = np.radians(object_dec)

    ra2 = np.radians(catalogue_ra)
    dec2 = np.radians(catalogue_dec)

    cosine_angle = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1)
        * np.cos(dec2)
        * np.cos(ra1 - ra2)
    )

    angle_radians = np.arccos(
        np.clip(cosine_angle, -1.0, 1.0)
    )

    return float(
        np.degrees(angle_radians.min()) * 3600.0
    )


# ==================================================
# 5. Load Lenscat coordinates
# ==================================================

lenscat = pd.read_csv(LENSCAT_PATH)
lenscat.columns = lenscat.columns.str.strip()

required_lenscat_columns = {
    "RA [deg]",
    "DEC [deg]",
}

missing_columns = required_lenscat_columns.difference(
    lenscat.columns
)

if missing_columns:
    raise ValueError(
        f"Lenscat is missing columns: {sorted(missing_columns)}"
    )

lenscat = lenscat.dropna(
    subset=["RA [deg]", "DEC [deg]"]
).copy()

lenscat_ra = lenscat["RA [deg]"].to_numpy(dtype=float)
lenscat_dec = lenscat["DEC [deg]"].to_numpy(dtype=float)

print(f"Lenscat coordinates loaded: {len(lenscat):,}")


# ==================================================
# 6. Load Galaxy10 labels and coordinates
# ==================================================

with h5py.File(H5_PATH, "r") as h5_file:
    labels = np.asarray(h5_file["ans"]).reshape(-1)
    ra_values = np.asarray(h5_file["ra"]).reshape(-1)
    dec_values = np.asarray(h5_file["dec"]).reshape(-1)
    redshift_values = np.asarray(
        h5_file["redshift"]
    ).reshape(-1)
    pixel_scale_values = np.asarray(
        h5_file["pxscale"]
    ).reshape(-1)


print(f"Galaxy10 objects loaded: {len(labels):,}")

print("\nGalaxy10 class counts:")

unique_classes, class_counts = np.unique(
    labels,
    return_counts=True,
)

for class_number, count in zip(
    unique_classes,
    class_counts,
):
    print(
        f"Class {class_number}: "
        f"{CLASS_NAMES.get(int(class_number), 'unknown')} "
        f"= {count}"
    )


# ==================================================
# 7. Select candidate ordinary galaxies
# ==================================================

valid_indices = np.where(
    np.isin(labels, ALLOWED_CLASSES)
    & np.isfinite(ra_values)
    & np.isfinite(dec_values)
)[0]

rng = np.random.default_rng(RANDOM_SEED)

candidate_pool_size = min(
    CANDIDATE_POOL_SIZE,
    len(valid_indices),
)

candidate_indices = rng.choice(
    valid_indices,
    size=candidate_pool_size,
    replace=False,
)


candidate_records = []

print("\nChecking distance from known Lenscat objects...")

for position, galaxy10_index in enumerate(
    candidate_indices,
    start=1,
):
    ra = float(ra_values[galaxy10_index])
    dec = float(dec_values[galaxy10_index])
    galaxy_class = int(labels[galaxy10_index])

    nearest_distance = minimum_separation_arcsec(
        object_ra=ra,
        object_dec=dec,
        catalogue_ra=lenscat_ra,
        catalogue_dec=lenscat_dec,
    )

    candidate_records.append({
        "galaxy10_index": int(galaxy10_index),
        "ra": ra,
        "dec": dec,
        "galaxy10_class": galaxy_class,
        "morphology": CLASS_NAMES[galaxy_class],
        "redshift": float(
            redshift_values[galaxy10_index]
        ),
        "original_pxscale": float(
            pixel_scale_values[galaxy10_index]
        ),
        "nearest_lenscat_arcsec": nearest_distance,
    })

    if position % 500 == 0:
        print(
            f"Checked {position}/{candidate_pool_size}"
        )


candidates = pd.DataFrame(candidate_records)

candidates = candidates[
    candidates["nearest_lenscat_arcsec"]
    > MINIMUM_DISTANCE_FROM_LENSCAT_ARCSEC
].copy()

candidates = candidates.drop_duplicates(
    subset=["ra", "dec"]
)

candidates = candidates.head(
    NUMBER_TO_DOWNLOAD
).reset_index(drop=True)

if len(candidates) < NUMBER_TO_DOWNLOAD:
    print(
        f"\nWarning: only {len(candidates)} candidates remained. "
        f"Increase CANDIDATE_POOL_SIZE if you need the full {NUMBER_TO_DOWNLOAD}."
    )

candidates["image_id"] = [
    f"nonlens_{index + 1:04d}"
    for index in range(len(candidates))
]

candidates["label"] = 0
candidates["class_name"] = "presumed_non_lens"

print(
    f"\nNon-lens candidates selected: {len(candidates)}"
)


# ==================================================
# 8. Download identical Legacy Survey cutouts
# ==================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "University gravitational lens classification project"
    )
})

download_records = []

for index, row in candidates.iterrows():
    image_id = row["image_id"]
    ra = float(row["ra"])
    dec = float(row["dec"])

    cutout_url = build_cutout_url(ra, dec)

    output_path = (
        OUTPUT_IMAGE_FOLDER
        / f"{image_id}.jpg"
    )

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

    # Resume support: skip files already downloaded in a previous run.
    if output_path.exists() and output_path.stat().st_size > 0:
        record["download_status"] = "already_downloaded"
        download_records.append(record)
        if (index + 1) % 50 == 0:
            print(f"[{index + 1}/{len(candidates)}] {image_id} — already downloaded, skipped")
        continue

    print(
        f"[{index + 1}/{len(candidates)}] "
        f"Downloading {image_id}"
    )

    try:
        response = session.get(
            cutout_url,
            timeout=30,
        )

        record["http_status"] = response.status_code
        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        )

        image.load()
        image = image.convert("RGB")

        image_array = np.asarray(image)

        record["width"] = image.width
        record["height"] = image.height
        record["mean_pixel"] = float(
            image_array.mean()
        )
        record["pixel_std"] = float(
            image_array.std()
        )

        image.save(
            output_path,
            format="JPEG",
            quality=95,
        )

        record["download_status"] = "success"

        print(f"  Saved: {image.size}")

    except requests.RequestException as error:
        record["error"] = f"Request error: {error}"
        print(f"  Request failed: {error}")

    except UnidentifiedImageError:
        record["error"] = (
            "Downloaded response was not a valid image."
        )
        print("  Failed: response was not an image")

    except Exception as error:
        record["error"] = str(error)
        print(f"  Failed: {error}")

    download_records.append(record)

    time.sleep(0.5)

    # Checkpoint: save progress periodically.
    if (index + 1) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(download_records).to_csv(
            OUTPUT_METADATA_PATH, index=False
        )


# ==================================================
# 9. Save metadata
# ==================================================

metadata = pd.DataFrame(download_records)

metadata.to_csv(
    OUTPUT_METADATA_PATH,
    index=False,
)

successful = (
    metadata["download_status"]
    .isin(["success", "already_downloaded"])
    .sum()
)

failed = len(metadata) - successful

print("\nDownload completed.")
print(f"Successful downloads: {successful}")
print(f"Failed downloads: {failed}")
print(f"\nImages saved to:\n{OUTPUT_IMAGE_FOLDER}")
print(f"\nMetadata saved to:\n{OUTPUT_METADATA_PATH}")

print("\nMorphology distribution:")
print(metadata["morphology"].value_counts())