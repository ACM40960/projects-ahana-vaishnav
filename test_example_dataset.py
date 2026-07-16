import os
import deeplenstronomy.deeplenstronomy as dl

# example.yaml expects paths like data/..., so we run from Notebooks
os.chdir("Notebooks")

dataset = dl.make_dataset(
    "data/example.yaml",
    survey="des",
    verbose=True,
    save_to_disk=True,
    image_file_format="npy"
)

print("\nExample dataset generated successfully.")
print("Dataset object:", dataset)
