[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-bKyY6qM)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=24090133&assignment_repo_type=AssignmentRepo)
<br>
# Explainable Deep Learning for Gravitational Lens Detection

This project builds a prototype CNN-based classifier for simulated gravitational lens detection.

## Dataset

The dataset was generated using deeplenstronomy. The positive class consists of simulated lens-like astronomical image cutouts. The negative class consists of simplified synthetic non-lensing galaxy images.

The final CNN-ready data files are:

- `data/X.npy`
- `data/y.npy`

These files are in the data folder and are generated data files.

## Model

The baseline model is a small convolutional neural network implemented in:

- `src/train_test_cnn.py`

The model performs binary classification:

- `0 = non-lens`
- `1 = lens`

## Running the model

```bash
conda activate deeplens
python src/train_test_cnn.py
