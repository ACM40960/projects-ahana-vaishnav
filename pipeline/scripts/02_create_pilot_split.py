from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Go back to the main project folder from the scripts folder
BASE_FOLDER = Path(__file__).resolve().parent.parent

# Input labels file made from the image checking script
LABELS_PATH = BASE_FOLDER / "data" / "metadata" / "labels.csv"

# Output file with train/validation/test split information
OUTPUT_PATH = BASE_FOLDER / "data" / "metadata" / "pilot_splits.csv"

# Load image metadata
metadata = pd.read_csv(LABELS_PATH)

# Keep only images that were readable and marked as usable
metadata = metadata[(metadata["readable"] == 1) & (metadata["usable"] == 1)].copy()

# First split: 70% train and 30% temporary data
# Stratify keeps the lens/non-lens ratio similar in each split
train_df, temp_df = train_test_split(metadata,test_size=0.30,stratify=metadata["label"],random_state=42)
# Second split: split the temporary data into 15% validation and 15% test
val_df, test_df = train_test_split(temp_df,test_size=0.50,stratify=temp_df["label"],random_state=42)

# Add split names so the model script knows which rows to use
train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

# Put all splits back into one CSV file
splits = pd.concat([train_df, val_df, test_df],ignore_index=True)
splits.to_csv(OUTPUT_PATH, index=False)

# Quick check to make sure both classes are present in each split
print("\nSplit summary:")
print(splits.groupby(["split", "class_name"]).size())
print(f"\nSaved to: {OUTPUT_PATH}")