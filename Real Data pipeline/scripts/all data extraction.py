from pathlib import Path
from io import BytesIO
from urllib.parse import urlencode
import time

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_FOLDER = Path(
    r"D:\MSc Data and Computational Science"
    r"\Gravitational Lensing\Gravitational_Lensing_Project"
    r"\projects-ahana-vaishnav-ahanabhattacharji-Strong-Gravitational-Lens-Finding-Challenge"
    r"\Real Data pipeline"
)

CATALOGUE_PATH = BASE_FOLDER / "data" / "catalog.csv"

# Renamed from "confident" to "lens" to match 01_prepare_dataset.py
# and 03_quality_filter.py, which both expect data/raw/lens.
CONFIDENT_FOLDER = BASE_FOLDER / "data" / "raw" / "lens"
PROBABLE_FOLDER = BASE_FOLDER / "data" / "raw" / "probable"
METADATA_FOLDER = BASE_FOLDER / "data" / "metadata"

CONFIDENT_FOLDER.mkdir(parents=True, exist_ok=True)
PROBABLE_FOLDER.mkdir(parents=True, exist_ok=True)
METADATA_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Control download size
# --------------------------------------------------

MAX_CONFIDENT = 2038   # all unique confident-grade galaxy lenses in catalog.csv
MAX_PROBABLE = 250     # capped — probable tier has a high blank/miss rate

IMAGE_SIZE = 128
PIXEL_SCALE = 0.262
SURVEY_LAYER = "ls-dr10"

# How often to write a metadata checkpoint (in images), so an
# interruption partway through doesn't lose all progress.
CHECKPOINT_EVERY = 100


# --------------------------------------------------
# 3. Load catalogue
# --------------------------------------------------

if not CATALOGUE_PATH.exists():
    raise FileNotFoundError(f"Could not find catalog.csv at:\n{CATALOGUE_PATH}")

catalogue = pd.read_csv(CATALOGUE_PATH)
catalogue.columns = catalogue.columns.str.strip()

required_columns = ["name", "RA [deg]", "DEC [deg]", "grading", "type"]

missing = [col for col in required_columns if col not in catalogue.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Catalogue loaded.")
print("Total rows:", len(catalogue))
print(catalogue["grading"].value_counts(dropna=False))


# --------------------------------------------------
# 4. Build Legacy Survey cutout URL
# --------------------------------------------------

def build_cutout_url(ra, dec, size=128):
    params = {
        "ra": ra,
        "dec": dec,
        "layer": SURVEY_LAYER,
        "size": size,
        "pixscale": PIXEL_SCALE,
    }

    return (
        "https://www.legacysurvey.org/viewer/jpeg-cutout?"
        + urlencode(params)
    )


# --------------------------------------------------
# 5. Filter confident and probable galaxy lenses
# --------------------------------------------------

def filter_by_grading(df, grading_value, max_rows):
    selected = df[
        df["grading"].astype(str).str.strip().str.lower().eq(grading_value)
        &
        df["type"].astype(str).str.strip().str.lower().eq("galaxy")
    ].copy()

    selected = selected.drop_duplicates(
        subset=["RA [deg]", "DEC [deg]"]
    )

    selected = selected.sample(
        n=min(max_rows, len(selected)),
        random_state=42
    ).reset_index(drop=True)

    selected["label"] = 1
    selected["grading_group"] = grading_value

    selected["image_id"] = [
        f"{grading_value}_{i + 1:04d}"
        for i in range(len(selected))
    ]

    selected["cutout_url"] = selected.apply(
        lambda row: build_cutout_url(
            row["RA [deg]"],
            row["DEC [deg]"],
            size=IMAGE_SIZE
        ),
        axis=1
    )

    return selected


confident_df = filter_by_grading(
    catalogue,
    grading_value="confident",
    max_rows=MAX_CONFIDENT
)

probable_df = filter_by_grading(
    catalogue,
    grading_value="probable",
    max_rows=MAX_PROBABLE
)

print("\nSelected confident images:", len(confident_df))
print("Selected probable images:", len(probable_df))


# --------------------------------------------------
# 6. Download image function
# --------------------------------------------------

session = requests.Session()
session.headers.update({
    "User-Agent": "University gravitational lens detection project"
})


def download_image(row, output_folder):
    image_id = row["image_id"]
    url = row["cutout_url"]

    output_path = output_folder / f"{image_id}.jpg"

    result = {
        "image_id": image_id,
        "name": row.get("name", ""),
        "RA [deg]": row["RA [deg]"],
        "DEC [deg]": row["DEC [deg]"],
        "type": row["type"],
        "grading": row["grading"],
        "grading_group": row["grading_group"],
        "label": row["label"],
        "cutout_url": url,
        "file_path": str(output_path),
        "download_status": "failed",
        "http_status": None,
        "width": None,
        "height": None,
        "error": None,
    }

    # Resume support: if this file was already downloaded successfully
    # in a previous (possibly interrupted) run, skip it.
    if output_path.exists() and output_path.stat().st_size > 0:
        result["download_status"] = "already_downloaded"
        return result

    try:
        response = session.get(url, timeout=30)
        result["http_status"] = response.status_code
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image.load()
        image = image.convert("RGB")

        result["width"], result["height"] = image.size

        if image.width < 32 or image.height < 32:
            raise ValueError(f"Image too small: {image.size}")

        image.save(output_path, format="JPEG", quality=95)

        result["download_status"] = "success"

    except requests.RequestException as error:
        result["error"] = f"Request error: {error}"

    except UnidentifiedImageError:
        result["error"] = "Downloaded file is not a valid image."

    except Exception as error:
        result["error"] = str(error)

    return result


# --------------------------------------------------
# 7. Download confident and probable images
# --------------------------------------------------

def download_group(df, output_folder, metadata_filename):
    records = []
    metadata_path = METADATA_FOLDER / metadata_filename

    for index, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc=f"Downloading {metadata_filename}"),
        start=1,
    ):
        result = download_image(row, output_folder)
        records.append(result)

        # Only sleep when we actually hit the network — skipped
        # (already-downloaded) files don't need to be polite-delayed.
        if result["download_status"] not in ("already_downloaded",):
            time.sleep(0.5)

        # Checkpoint: save progress periodically so an interruption
        # doesn't lose everything downloaded so far.
        if index % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(metadata_path, index=False)

    metadata = pd.DataFrame(records)
    metadata.to_csv(metadata_path, index=False)

    print(f"\nSaved metadata to:\n{metadata_path}")
    print(metadata["download_status"].value_counts(dropna=False))

    return metadata


confident_metadata = download_group(
    confident_df,
    CONFIDENT_FOLDER,
    "confident_images_metadata.csv"
)

probable_metadata = download_group(
    probable_df,
    PROBABLE_FOLDER,
    "probable_images_metadata.csv"
)


# --------------------------------------------------
# 8. Save combined metadata
# --------------------------------------------------

combined_metadata = pd.concat(
    [confident_metadata, probable_metadata],
    ignore_index=True
)

combined_path = METADATA_FOLDER / "confident_probable_images_metadata.csv"
combined_metadata.to_csv(combined_path, index=False)

print(f"\nCombined metadata saved to:\n{combined_path}")
print("\nDone.")