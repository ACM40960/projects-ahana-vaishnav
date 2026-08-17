# Gravitational Lens Detector

**A deep learning prototype for screening gravitational lens candidates in astronomical survey images, with visual explanations for model predictions.**

Ahana Bhattacharji & Vaishnav Malvankar  
MSc Data and Computational Science, University College Dublin, 2026

[![Try the live demo](https://img.shields.io/badge/demo-Streamlit%20App-orange)](https://gravitational-lens-detector.streamlit.app/)

---

## Overview

Strong gravitational lensing occurs when a massive foreground galaxy bends light from a more distant background source. In survey images, this can appear as arcs, rings, or multiple distorted images.

Finding these systems manually is difficult because modern sky surveys contain huge numbers of objects and genuine lenses are rare. This project builds a student-scale screening pipeline that classifies image cutouts as **lens candidates** or **presumed non-lenses**.

The project compares four models:

- Custom CNN
- CBAM-attention CNN
- MobileNetV3 transfer learning
- ViT-B/16 transfer learning

The deployed app also provides Grad-CAM or attention-based heatmaps so users can inspect which parts of an image influenced the prediction.

---

## Live Demo

Try the app here:

**https://gravitational-lens-detector.streamlit.app/**

The app lets you:

- choose one of the trained models;
- upload a galaxy image cutout;
- use built-in sample images;
- view the predicted lens probability;
- inspect the heatmap explaining the prediction.

The app is a research prototype. It should be treated as a **candidate-screening tool**, not a validated astronomical discovery system.

---

## Results

All models were evaluated on the same stratified test split.

| Model | ROC-AUC | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| Custom CNN | 0.983 | 94.4% | 94.0% | 94.3% |
| CBAM-Attention CNN | 0.985 | 93.2% | 90.6% | **95.7%** |
| MobileNetV3 | 0.981 | 94.7% | 94.7% | 94.3% |
| **ViT-B/16** | **0.992** | **96.3%** | **98.5%** | 93.6% |

**Best overall model:** ViT-B/16  
**Best recall:** CBAM-Attention CNN  
**Lowest false positives:** ViT-B/16, with 4 false positives out of 306 non-lens test images

Full metrics, thresholds, confusion matrices, and plots are stored in:

```text
pipeline/results/
```

---

## Dataset

The final training dataset contains **3,920 usable real Legacy Survey DR10 image cutouts**.

| Class | Count | Source |
|---|---:|---|
| Lens candidates | 1,882 | Lens catalogue candidates downloaded as Legacy Survey DR10 cutouts |
| Presumed non-lenses | 2,038 | Galaxy10-DECaLS galaxies downloaded through the same Legacy Survey cutout service |

Image size:

```text
128 × 128 pixels
```

The raw data also includes a separate **probable lens** tier. These images were kept out of the main train/test split and used only for additional inspection.

### Data construction summary

1. Lens candidates were selected from the lens catalogue.
2. Non-lens examples were sampled from Galaxy10-DECaLS galaxy classes.
3. Non-lens examples were filtered to avoid known Lenscat coordinates.
4. Images were downloaded as Legacy Survey DR10 JPEG cutouts.
5. Basic quality checks removed blank, unreadable, or off-target images.
6. The final data was split into train, validation, and test sets.

The quality metadata is stored in:

```text
pipeline/data/metadata/quality_metadata.csv
```

---

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── Literature Review.pdf
├── Ahana & Vaishnav_Literature Review.pdf
├── data/
│   ├── catalog.csv
│   ├── gravitational_lenses.parquet
│   ├── metadata/
│   └── raw/
│       ├── lens/
│       ├── non_lens/
│       └── probable/
├── deployment/
│   ├── app.py
│   ├── artifacts/
│   ├── sample_images/
│   └── website images/
├── figures/
├── models/
├── results/
└── scripts/
```

---

## Quickstart

Clone the repository:

```bash
git clone https://github.com/ACM40960/projects-ahana-vaishnav.git
cd projects-ahana-vaishnav
```

Install dependencies:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Run the Streamlit app locally:

```bash
cd pipeline/deployment
streamlit run app.py
```

The app uses the exported model files in:

```text
pipeline/deployment/artifacts/
```

The ViT-B/16 model uses a frozen ImageNet backbone from `torchvision`. Only the trained head is stored in the repository; the backbone may be downloaded automatically the first time ViT is used.

---

## Reproducing the Pipeline

The main scripts are stored in:

```text
pipeline/scripts/
```

Run them in numbered order.

```text
01_prepare_dataset.py
02_create_pilot_split.py
03_quality_filter.py
04_train_real_data_cnn.py
05_gradcam_explainability.py
06_brightness_mitigation.py
07_transfer_learning_cnn.py
08_export_deployment_artifacts.py
10_cbam_attention_cnn.py
11_vit_transfer_learning.py
12_compare_models.py
```

Useful extra scripts:

```text
all_data_extraction.py
create_non_lens_dataset.py
export_vit_head_only.py
```

The full Galaxy10-DECaLS H5 file is not included in the repository. To rebuild the non-lens dataset from scratch, download `Galaxy10_DECals.h5` separately and place it here:

```text
pipeline/data/raw/Galaxy10_DECals.h5
```

Then run:

```bash
python pipeline/scripts/create_non_lens_dataset.py
```

---

## Explainability

The project uses:

- **Grad-CAM** for CNN-based models;
- **attention visualisation / attention rollout** for ViT-B/16.

These visualisations show which image regions influenced the model prediction. They do **not** prove that the model understands the physics of gravitational lensing.

A good lens-candidate prediction should ideally focus on arc-like or ring-like structures. If the heatmap mainly highlights a bright central source, image edge, star, or artefact, the prediction should be treated with caution.

Generated explainability outputs are stored in:

```text
pipeline/results/real_data_cnn/
pipeline/results/cbam_attention_cnn/
pipeline/results/vit_transfer_learning/
```

---

## Limitations

This project is a useful prototype, but it has important limitations.

- **Not a validated discovery tool.** The model can help prioritise candidates, but it cannot confirm a real gravitational lens.
- **Labels are imperfect.** Lens labels come from catalogues, and non-lens examples are presumed non-lenses rather than expert-confirmed negatives.
- **Dataset bias may exist.** The model may partly learn differences in brightness, compactness, or image quality rather than pure lens morphology.
- **Brightness shortcut risk.** Model confidence correlates with a centre-to-edge brightness ratio (`r ≈ -0.81`). Per-image normalisation did not fully remove this effect.
- **No independent external benchmark yet.** The models have not yet been validated on a separate expert-reviewed survey dataset.

These limitations are part of the analysis rather than a failure of the project. They show why explainability and false-positive inspection are necessary.

---

## Key Files

| File / folder | Purpose |
|---|---|
| `pipeline/deployment/app.py` | Streamlit demo app |
| `pipeline/deployment/artifacts/` | Exported model weights and preprocessing configs |
| `pipeline/scripts/` | Data, training, evaluation, and deployment scripts |
| `pipeline/results/model_comparison/` | Model comparison outputs |
| `pipeline/results/*/test_metrics.json` | Test metrics for each model |
| `pipeline/results/*/classification_report.txt` | Classification reports |
| `pipeline/data/metadata/quality_metadata.csv` | Final quality-filtered image metadata |

---

## References

- Lanusse et al. (2017), *CMU DeepLens: Deep learning for automatic image-based galaxy-galaxy strong lens finding*
- Metcalf et al. (2019), *The Strong Gravitational Lens Finding Challenge*
- Bom et al. (2022), *Developing a victorious strategy to the Second Strong Gravitational Lensing Data Challenge*
- Selvaraju et al. (2017), *Grad-CAM: Visual explanations from deep networks via gradient-based localization*
- Woo et al. (2018), *CBAM: Convolutional Block Attention Module*
- Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
- Abnar and Zuidema (2020), *Quantifying Attention Flow in Transformers*
- Pearce-Casey et al. (2024), *Euclid: Searches for strong gravitational lenses using convolutional neural nets in Early Release Observations of the Perseus field*

Non-lens imagery uses Galaxy10-DECaLS. Lens and non-lens cutouts were downloaded from the DESI Legacy Imaging Surveys / Legacy Survey DR10 viewer.

---
