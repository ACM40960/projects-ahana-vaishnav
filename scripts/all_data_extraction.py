from pathlib import Path
from io import BytesIO
from urllib.parse import urlencode
import time

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Project paths
BASE_FOLDER = Path(__file__).resolve().parent.parent
CATALOGUE_PATH = BASE_FOLDER / "data" / "catalog.csv"
CONFIDENT_FOLDER = BASE_FOLDER / "data" / "raw" / "lens"
PROBABLE_FOLDER = BASE_FOLDER / "data" / "raw" / "probable"
METADATA_FOLDER = BASE_FOLDER / "data" / "metadata"

# Create folders if they are not already present
CONFIDENT_FOLDER.mkdir(parents=True, exist_ok=True)
PROBABLE_FOLDER.mkdir(parents=True, exist_ok=True)
METADATA_FOLDER.mkdir(parents=True, exist_ok=True)

# Download settings
MAX_CONFIDENT = 2038
MAX_PROBABLE = 250
IMAGE_SIZE = 128
PIXEL_SCALE = 0.262
SURVEY_LAYER = "ls-dr10"
CHECKPOINT_EVERY = 100

# Load catalogue
if not CATALOGUE_PATH.exists():
    raise FileNotFoundError(f"Could not find catalog.csv at:\n{CATALOGUE_PATH}")

catalogue = pd.read_csv(CATALOGUE_PATH)
catalogue.columns = catalogue.columns.str.strip()

# Basic column check before filtering
required_columns = ["name", "RA [deg]", "DEC [deg]", "grading", "type"]
missing = [col for col in required_columns if col not in catalogue.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Catalogue loaded:", len(catalogue), "rows")
print(catalogue["grading"].value_counts(dropna=False))

def build_cutout_url(ra, dec, size=128):
    # Build the Legacy Survey cutout URL for one sky position
    params = {
        "ra": ra,
        "dec": dec,
        "layer": SURVEY_LAYER,
        "size": size,
        "pixscale": PIXEL_SCALE
    }
    return "https://www.legacysurvey.org/viewer/jpeg-cutout?" + urlencode(params)

def filter_by_grading(df, grading_value, max_rows):
    # Keep only galaxy candidates with the required grading
    selected = df[df["grading"].astype(str).str.strip().str.lower().eq(grading_value) & df["type"].astype(str).str.strip().str.lower().eq("galaxy")].copy()

    # Remove duplicate coordinates
    selected = selected.drop_duplicates(subset=["RA [deg]", "DEC [deg]"])

    # Use a fixed random seed so the same sample is chosen each run
    selected = selected.sample(n=min(max_rows, len(selected)),random_state=42).reset_index(drop=True)

    # These are all lens candidates, so label is 1
    selected["label"] = 1
    selected["grading_group"] = grading_value

    # Simple filename-friendly ID
    selected["image_id"] = [f"{grading_value}_{i + 1:04d}" for i in range(len(selected))]

    # Add image download links
    selected["cutout_url"] = selected.apply(
        lambda row: build_cutout_url(
            row["RA [deg]"],
            row["DEC [deg]"],
            size=IMAGE_SIZE
        ),
        axis=1
    )

    return selected

confident_df = filter_by_grading(catalogue, "confident", MAX_CONFIDENT)
probable_df = filter_by_grading(catalogue, "probable", MAX_PROBABLE)

print(f"\nSelected: {len(confident_df)} confident | {len(probable_df)} probable")

# Reuse the same session for all downloads
session = requests.Session()
session.headers.update({
    "User-Agent": "University gravitational lens detection project"
})

def download_image(row, output_folder):
    image_id = row["image_id"]
    url = row["cutout_url"]
    output_path = output_folder / f"{image_id}.jpg"

    # Store download info for later checking
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
        "error": None
    }

    # Skip files that were already downloaded
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

        # Very small images are probably not useful
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

def download_group(df, output_folder, metadata_filename):
    records = []
    metadata_path = METADATA_FOLDER / metadata_filename

    for index, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Downloading {metadata_filename}"),start=1,):
        result = download_image(row, output_folder)
        records.append(result)
        # Small delay to avoid hitting the server too aggressively
        if result["download_status"] != "already_downloaded":
            time.sleep(0.5)
        # Save partial progress in case the script stops
        if index % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(metadata_path, index=False)

    metadata = pd.DataFrame(records)
    metadata.to_csv(metadata_path, index=False)
    print(f"\nSaved metadata to: {metadata_path}")
    print(metadata["download_status"].value_counts(dropna=False))
    return metadata

# Download both groups
confident_metadata = download_group(confident_df,CONFIDENT_FOLDER,"confident_images_metadata.csv")
probable_metadata = download_group(probable_df,PROBABLE_FOLDER,"probable_images_metadata.csv")

# Save one combined metadata file
combined = pd.concat([confident_metadata, probable_metadata], ignore_index=True)
combined_path = METADATA_FOLDER / "confident_probable_images_metadata.csv"
combined.to_csv(combined_path, index=False)