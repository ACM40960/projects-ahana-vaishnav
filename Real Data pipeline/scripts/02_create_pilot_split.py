from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_FOLDER = Path(
    r"D:\MSc Data and Computational Science"
    r"\Gravitational Lensing\Datasets"
)

LABELS_PATH = (
    BASE_FOLDER
    / "data"
    / "metadata"
    / "labels.csv"
)

OUTPUT_PATH = (
    BASE_FOLDER
    / "data"
    / "metadata"
    / "pilot_splits.csv"
)


metadata = pd.read_csv(LABELS_PATH)

metadata = metadata[
    (metadata["readable"] == 1)
    & (metadata["usable"] == 1)
].copy()


# First split: 70% train, 30% temporary
train_df, temporary_df = train_test_split(
    metadata,
    test_size=0.30,
    stratify=metadata["label"],
    random_state=42,
)


# Second split: divide temporary set equally
validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    stratify=temporary_df["label"],
    random_state=42,
)


train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"


splits = pd.concat(
    [train_df, validation_df, test_df],
    ignore_index=True,
)

splits.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\nSplit summary:")
print(
    splits.groupby(
        ["split", "class_name"]
    )
    .size()
)

print(f"\nSaved to:\n{OUTPUT_PATH}")