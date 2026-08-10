"""
app.py — Gravitational Lens Detector (Streamlit demo)

Explainable Deep Learning for Gravitational Lens Detection in
Astronomical Survey Images — student research prototype demo.

Upload an astronomical image cutout, or click a sample, to see the
model's prediction, confidence, and a Grad-CAM heatmap of what it
focused on.

This is a student-project prototype demo, not a validated scientific
tool — see the "Model and Dataset Notes" and "Limitations" sections
in the app itself.

REQUIRED FILES (relative to this script's folder):
  artifacts/model_weights.pt           — trained model state_dict (REQUIRED)
  artifacts/preprocessing_config.json  — mean/std/threshold/test metrics (REQUIRED)
  sample_images/*.jpg                  — optional example images for the
                                          "Try a sample" gallery (OPTIONAL —
                                          the app still works without these,
                                          it just skips the sample gallery)

To run locally:      streamlit run app.py
To deploy publicly:  push this file + artifacts/ + sample_images/ to a
                      public GitHub repo, then deploy via
                      share.streamlit.io
"""

from pathlib import Path
import io
import json

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_weights.pt"
CONFIG_PATH = ARTIFACTS_DIR / "preprocessing_config.json"
SAMPLES_DIR = APP_DIR / "sample_images"

IMAGE_SIZE = 128


# --------------------------------------------------
# 2. Model definition (UNCHANGED — must exactly match training script)
# --------------------------------------------------

class RealDataCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x).squeeze(1)


@st.cache_resource
def load_model_and_config():
    """Loads model + config. Raises on failure — caller handles the error
    message so the app can show a helpful Streamlit error instead of a
    raw traceback/crash."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    model = RealDataCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, config


# --------------------------------------------------
# 3. Grad-CAM (UNCHANGED logic)
# --------------------------------------------------

activations = {}
gradients = {}


def register_gradcam_hooks(model):
    target_layer = model.features[6]  # final Conv2d(32, 64, ...)

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)


def preprocess_image(pil_image: Image.Image, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    image = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image, dtype="float32")
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - torch.tensor(mean)) / torch.tensor(std)
    return tensor.unsqueeze(0)


def predict_with_gradcam(pil_image: Image.Image, model, mean: np.ndarray, std: np.ndarray):
    input_tensor = preprocess_image(pil_image, mean, std)
    input_tensor.requires_grad_(True)

    model.zero_grad()
    logit = model(input_tensor)
    probability = torch.sigmoid(logit).item()

    logit.backward()

    acts = activations["value"][0]
    grads = gradients["value"][0]

    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam.detach().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    cam_image = Image.fromarray((cam * 255).astype("uint8")).resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR
    )
    cam_resized = np.asarray(cam_image, dtype="float32") / 255.0

    return probability, cam_resized


# --------------------------------------------------
# 4. Page setup + dark space/science theme
# --------------------------------------------------

st.set_page_config(
    page_title="Gravitational Lens Detector",
    page_icon="🔭",
    layout="centered",
)

st.markdown(
"""
<style>
/* ===============================
   Global astronomical dark theme
   =============================== */

.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(60, 90, 180, 0.25), transparent 30%),
        radial-gradient(circle at 80% 10%, rgba(130, 80, 200, 0.22), transparent 25%),
        radial-gradient(circle at 50% 80%, rgba(0, 180, 220, 0.12), transparent 30%),
        linear-gradient(135deg, #030712 0%, #07111f 45%, #020617 100%);
    color: #e5e7eb;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f8fafc;
    letter-spacing: -0.03em;
}

p, li, span, div {
    color: #dbeafe;
}

/* Hero panel */
.hero-card {
    padding: 2rem 2.2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.72));
    border: 1px solid rgba(148, 163, 184, 0.28);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.6rem;
    background: linear-gradient(90deg, #e0f2fe, #93c5fd, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #cbd5e1;
    max-width: 850px;
    line-height: 1.7;
}

/* Generic content cards */
.glass-card {
    padding: 1.25rem 1.5rem;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.22);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
    margin-bottom: 1rem;
}

/* Small feature cards (Why This Matters) */
.feature-card {
    padding: 1.1rem 1.1rem;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.22);
    height: 100%;
    min-height: 150px;
}

.feature-card-title {
    font-weight: 700;
    font-size: 1.02rem;
    color: #93c5fd;
    margin-bottom: 0.4rem;
}

.feature-card-body {
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.5;
}

/* Pipeline step chips */
.pipeline-step {
    padding: 0.9rem 0.6rem;
    border-radius: 14px;
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(147, 197, 253, 0.3);
    text-align: center;
    font-size: 0.82rem;
    color: #e0f2fe;
    min-height: 90px;
}

.pipeline-arrow {
    text-align: center;
    font-size: 1.4rem;
    color: #93c5fd;
    padding-top: 1.6rem;
}

.warning-card {
    padding: 1rem 1.25rem;
    border-radius: 16px;
    background: rgba(120, 53, 15, 0.30);
    border: 1px solid rgba(251, 191, 36, 0.35);
    color: #fde68a;
    margin-top: 1rem;
}

.footer-card {
    padding: 1rem 1.5rem;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.18);
    color: #94a3b8;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 1.5rem;
}

/* Prediction cards */
.prediction-lens {
    padding: 1.5rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(88, 28, 135, 0.8), rgba(30, 64, 175, 0.78));
    border: 1px solid rgba(196, 181, 253, 0.45);
    box-shadow: 0 18px 50px rgba(88, 28, 135, 0.35);
    margin-bottom: 1rem;
}

.prediction-nonlens {
    padding: 1.5rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 64, 175, 0.45));
    border: 1px solid rgba(96, 165, 250, 0.35);
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.5);
    margin-bottom: 1rem;
}

.prediction-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.35rem;
}

.prediction-subtitle {
    font-size: 0.95rem;
    color: #cbd5e1;
}

/* Widgets */
.stButton button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(147, 197, 253, 0.35);
    background: rgba(15, 23, 42, 0.82);
    color: #dbeafe;
    transition: all 0.2s ease-in-out;
}

.stButton button:hover {
    border-color: #93c5fd;
    color: #ffffff;
    background: rgba(30, 64, 175, 0.85);
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.68);
    border: 1px dashed rgba(147, 197, 253, 0.45);
    border-radius: 18px;
    padding: 1rem;
}

[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 16px;
}

[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 14px;
    padding: 0.6rem 0.8rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 10px 10px 0 0;
    padding: 0.5rem 1rem;
}

.stProgress > div > div > div > div {
    background-image: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
}

hr {
    border-color: rgba(148, 163, 184, 0.22);
}

img {
    border-radius: 14px;
}

footer {
    visibility: hidden;
}
</style>
""",
unsafe_allow_html=True,
)


# --------------------------------------------------
# 5. HEADER — title + subtitle (as specified)
# --------------------------------------------------

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🔭 Explainable Deep Learning for Gravitational Lens Detection</div>
        <div class="hero-subtitle">
            A student prototype for classifying astronomical survey image cutouts as
            <b>lens</b> or <b>non-lens</b> candidates.
        </div>
        <div class="warning-card">
            <b>Scientific caution:</b> this is not a validated astronomical discovery tool.
            It is an academic prototype and should not be used to draw real astrophysical
            conclusions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 6. Load model + config robustly.
# If required files are missing, show a helpful error and stop
# instead of crashing with a raw traceback.
# --------------------------------------------------

model = None
config = None
load_error = None

try:
    model, config = load_model_and_config()
    register_gradcam_hooks(model)
except FileNotFoundError as error:
    load_error = (
        "**Required model files are missing.**\n\n"
        f"Could not find one of:\n"
        f"- `{MODEL_PATH}`\n"
        f"- `{CONFIG_PATH}`\n\n"
        "Make sure `artifacts/model_weights.pt` and "
        "`artifacts/preprocessing_config.json` exist next to this app.py "
        "(run `08_export_deployment_artifacts.py` first if you haven't)."
    )
except Exception as error:
    load_error = f"**Could not load the model.**\n\nDetails: `{error}`"

if load_error:
    st.error(load_error)
    st.stop()

mean = np.array(config["mean"], dtype="float32").reshape(3, 1, 1)
std = np.array(config["std"], dtype="float32").reshape(3, 1, 1)
best_threshold = config["best_threshold"]
test_roc_auc = config.get("test_roc_auc", 0.0) or 0.0
test_accuracy = config.get("test_accuracy", 0.0) or 0.0


# --------------------------------------------------
# 7. Top-level navigation via tabs
# --------------------------------------------------

tab_overview, tab_pipeline, tab_try, tab_interpret, tab_references = st.tabs(
    [
        "🏠 Overview",
        "🧬 How It Works",
        "🧪 Try the Model",
        "📖 Interpretation & Notes",
        "📚 References",
    ]
)


# ====================================================
# TAB 1 — Project Overview + Why This Matters
# ====================================================

with tab_overview:

    # ---- Project Overview ----
    st.markdown("### Project Overview")
    st.markdown(
        """
        <div class="glass-card">
        Strong gravitational lensing occurs when a massive foreground object —
        a galaxy or galaxy cluster — bends light from a more distant background
        source, producing visible <b>arcs</b>, <b>rings</b>, or <b>multiple
        distorted images</b>. These systems are scientifically valuable for
        studying dark matter and distant galaxies, but they are also
        <b>rare</b>.
        <br><br>
        Modern astronomical surveys capture millions to billions of objects —
        far too many for manual inspection to scale. This app demonstrates a
        <b>CNN-based classification prototype</b> that flags likely lens
        candidates automatically, paired with <b>Grad-CAM explainability</b>
        so predictions can be visually inspected rather than trusted blindly.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick metric snapshot of the model's test performance
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Test ROC-AUC", f"{test_roc_auc:.3f}")
    with metric_col2:
        st.metric("Test Accuracy", f"{test_accuracy:.1%}")
    with metric_col3:
        st.metric("Decision Threshold", f"{best_threshold:.2f}")

    st.divider()

    # ---- Why This Matters (4 cards) ----
    st.markdown("### Why This Matters")

    why_col1, why_col2, why_col3, why_col4 = st.columns(4)

    why_cards = [
        ("🔍", "Rare-Object Detection",
         "Genuine strong lenses are a tiny fraction of all survey objects, "
         "making this a hard imbalanced-classification problem, not a "
         "simple binary split."),
        ("⚠️", "False Positives",
         "Spiral galaxies, ring galaxies, mergers, bright stars, and imaging "
         "artefacts can all visually resemble lensing features."),
        ("🛰️", "Astronomical Survey Scale",
         "Modern surveys produce millions of image cutouts — manual "
         "inspection by expert astronomers cannot scale to this volume."),
        ("🧠", "Explainable AI",
         "Grad-CAM visualises which image regions drove each prediction, "
         "supporting scrutiny of the model rather than blind trust."),
    ]

    for col, (icon, title, body) in zip([why_col1, why_col2, why_col3, why_col4], why_cards):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-card-title">{icon} {title}</div>
                    <div class="feature-card-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ====================================================
# TAB 2 — How the Pipeline Works
# ====================================================

with tab_pipeline:
    st.markdown("### How the Pipeline Works")
    st.caption("The same flow runs every time you submit an image in the 'Try the Model' tab.")

    steps = [
        "📤\n\nImage input",
        "🧹\n\nPreprocessing\n(resize, normalise)",
        "🧠\n\nCNN prediction",
        "📊\n\nConfidence score",
        "🔥\n\nGrad-CAM heatmap",
        "🧾\n\nInterpretation",
    ]

    # Render as a row of step-cards separated by arrow glyphs.
    pipeline_cols = st.columns(len(steps) * 2 - 1)
    for i, step_text in enumerate(steps):
        col_index = i * 2
        with pipeline_cols[col_index]:
            st.markdown(f'<div class="pipeline-step">{step_text}</div>', unsafe_allow_html=True)
        if col_index + 1 < len(pipeline_cols):
            with pipeline_cols[col_index + 1]:
                st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        """
        <div class="glass-card">
        <b>1. Image input</b> — a user-uploaded cutout or one of the built-in samples.<br>
        <b>2. Preprocessing</b> — resized to 128×128 and normalised using the same
        mean/std statistics computed from the training set.<br>
        <b>3. CNN prediction</b> — a convolutional neural network (trained on real
        Legacy Survey imagery) outputs a raw probability.<br>
        <b>4. Confidence score</b> — the probability is compared against a
        validation-tuned decision threshold (not the default 0.5).<br>
        <b>5. Grad-CAM heatmap</b> — gradients from the final convolutional layer
        highlight which image regions most influenced the prediction.<br>
        <b>6. Interpretation</b> — the result is presented as a candidate flag for
        human review, not a scientific confirmation.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ====================================================
# TAB 3 — Try the Model (core functionality, UNCHANGED logic)
# ====================================================

with tab_try:
    st.markdown("### Try the Model")

    if "selected_image_bytes" not in st.session_state:
        st.session_state.selected_image_bytes = None
        st.session_state.selected_image_name = None

    # ---- Sample gallery (robust: skipped entirely if folder/files missing) ----
    sample_files = []
    try:
        if SAMPLES_DIR.exists():
            sample_files = sorted(SAMPLES_DIR.glob("*.jpg"))
    except Exception:
        sample_files = []

    if sample_files:
        st.subheader("Try a sample image")
        cols = st.columns(len(sample_files))
        for col, sample_path in zip(cols, sample_files):
            with col:
                st.image(str(sample_path), use_container_width=True)
                if st.button(sample_path.stem, key=f"sample_{sample_path.stem}"):
                    st.session_state.selected_image_bytes = sample_path.read_bytes()
                    st.session_state.selected_image_name = sample_path.name
        st.divider()
    else:
        st.caption(
            "ℹ️ No sample images found in `sample_images/` — you can still upload "
            "your own image below."
        )

    # ---- Upload ----
    st.markdown("#### Or upload your own")
    st.caption(
        "Upload a square galaxy-scale cutout in JPG or PNG format. For best "
        "results, use an image similar to the training data: centred object, "
        "survey-style cutout, minimal cropping artefacts."
    )
    uploaded_file = st.file_uploader(
        "Upload an astronomical image cutout (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        st.session_state.selected_image_bytes = uploaded_file.read()
        st.session_state.selected_image_name = uploaded_file.name

    # ---- Run prediction on whichever image is selected ----
    if st.session_state.selected_image_bytes is not None:
        try:
            pil_image = Image.open(io.BytesIO(st.session_state.selected_image_bytes))
        except Exception as error:
            st.error(f"Could not read this image file: `{error}`")
            st.stop()

        with st.spinner("Running model..."):
            probability, cam = predict_with_gradcam(pil_image, model, mean, std)

        predicted_label = "Lens candidate" if probability >= best_threshold else "Non-lens"

        st.divider()
        st.caption(f"Analysing: **{st.session_state.selected_image_name}**")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Image")
            st.image(
                pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)),
                use_container_width=True,
            )

        with col2:
            st.subheader("Grad-CAM heatmap")
            display_image = np.asarray(pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(display_image)
            ax.imshow(cam, cmap="magma", alpha=0.48)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Bright yellow/white regions indicate stronger influence on the prediction.")

        st.divider()

        # ---- Prediction card ----
        if predicted_label == "Lens candidate":
            st.markdown(
                f"""
                <div class="prediction-lens">
                    <div class="prediction-title">Prediction: {predicted_label}</div>
                    <div class="prediction-subtitle">
                        The model output is above the validation-tuned threshold.
                        Treat this as a candidate flag, not a scientific confirmation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-nonlens">
                    <div class="prediction-title">Prediction: {predicted_label}</div>
                    <div class="prediction-subtitle">
                        The model output is below the validation-tuned lens threshold.
                        This does not prove the object is not a lens.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- Confidence ----
        conf_col1, conf_col2 = st.columns([2, 1])
        with conf_col1:
            st.write("**Lens confidence**")
            st.progress(min(max(probability, 0.0), 1.0), text=f"{probability:.1%}")
            st.caption(
                f"Decision threshold used: {best_threshold:.2f} (tuned on "
                "validation-set F1, not the default 0.5 — see the "
                "Interpretation & Notes tab for why)."
            )
        with conf_col2:
            st.metric("Lens probability", f"{probability:.1%}")

    else:
        st.info("Click a sample above, or upload your own image, to get a prediction.")


# ====================================================
# TAB 4 — How to Interpret the Result + Model/Dataset Notes + Limitations
# ====================================================

with tab_interpret:
    st.markdown("### How to Interpret the Result")
    st.markdown(
        """
        <div class="glass-card">
        <ul>
            <li><b>High confidence does not guarantee a true lens.</b> The score
            reflects the model's learned pattern-matching, not a physical
            confirmation of gravitational lensing.</li>
            <li><b>Grad-CAM shows influence, not proof.</b> It highlights the image
            regions that most affected the prediction — it does not verify that
            the model has learned the actual physics of lensing.</li>
            <li><b>A good lens prediction should focus on arc-like or ring-like
            structure.</b> If the heatmap instead concentrates on a bright point
            source, a star, or an image edge, treat the prediction with more
            caution.</li>
            <li><b>False positives may happen</b> due to spiral arms, bright stars,
            imaging artefacts, or overlapping/blended galaxies that visually
            resemble lensing features.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Model and Dataset Notes")
    st.markdown(
        """
        <div class="glass-card">
        <ul>
            <li>This is a <b>student-scale prototype</b>, built for a final-year
            academic project — not a production astronomical pipeline.</li>
            <li>The model is <b>not a validated astronomical discovery tool</b> and
            should not be used to draw real scientific conclusions.</li>
            <li><b>Dataset quality and class imbalance</b> directly affect
            performance: positive examples come from catalogue-listed lens
            systems of varying confidence, and negative examples are
            <i>presumed</i> non-lens galaxies, not objects proven impossible to
            lens.</li>
            <li>Results should be treated as <b>candidate screening</b> to
            prioritise objects for expert review — not as final scientific
            confirmation.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    with st.expander("Known limitations of this model", expanded=False):
        st.markdown(
            """
            <div class="glass-card">
            <ul>
                <li>
                    The model is trained on a modest student-project dataset,
                    not on a fully expert-verified astronomical benchmark.
                </li>
                <li>
                    Positive examples are based on catalogue-listed lens systems
                    or candidates. Some lensing features may be faint or not
                    clearly visible in the Legacy Survey JPEG cutouts.
                </li>
                <li>
                    Negative examples are presumed non-lens galaxies, not objects
                    proven impossible to lens.
                </li>
                <li>
                    The model may partially rely on brightness, compactness,
                    centring, or image-quality differences rather than pure
                    lens morphology.
                </li>
                <li>
                    Grad-CAM highlights influential image regions, but it does
                    not prove that the CNN has learned the physical process of
                    gravitational lensing.
                </li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("What we found — project insights", expanded=False):
        st.markdown(
            """
            <div class="glass-card">
            <ul>
                <li>
                    <b>Three independently trained models converge to ~0.98 ROC-AUC</b>
                    on the real dataset — a custom CNN, a per-image-normalised variant,
                    and a transfer-learning model with a frozen MobileNetV3 backbone.
                    Agreement across different architectures suggests this is close to
                    the practical ceiling for what this dataset supports, rather than
                    an artifact of one model.
                </li>
                <li>
                    <b>A model trained on simulated lens data (deeplenstronomy) scored a
                    near-perfect ROC-AUC of 1.0</b> — far higher than any model trained
                    on real survey images. This sim-to-real gap is consistent with prior
                    literature (Pearce-Casey et al., 2024) and is the core reason this
                    project moved to real Legacy Survey imagery rather than relying on
                    simulation alone.
                </li>
                <li>
                    <b>The model's confidence correlates with image brightness/compactness</b>
                    (Pearson r ≈ -0.81 between predicted probability and a center-vs-edge
                    brightness ratio). Per-image brightness normalisation did not remove
                    this correlation, suggesting it reflects a genuine structural
                    difference between the training classes — real lens candidates tend
                    to be fainter and more distant than the resolved, nearby non-lens
                    galaxies used as negatives — rather than a simple normalisation
                    artifact. Full discussion in the accompanying report.
                </li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ====================================================
# TAB 5 — References / Research Context
# ====================================================

with tab_references:
    st.markdown("### References / Research Context")
    st.markdown(
        """
        <div class="glass-card">
        <ul>
            <li><b>CNN-based lens detection:</b> Lanusse et al. (2017), CMU DeepLens —
            early deep learning methods for automatic galaxy-galaxy strong lens
            finding.</li>
            <li><b>Strong Gravitational Lens Finding Challenge:</b> Metcalf et al.
            (2019) and Bom et al. (2022) — machine learning challenges framing
            lens detection as rare-object classification and highlighting the
            importance of controlling false positives.</li>
            <li><b>Sim-to-real gap:</b> Pearce-Casey et al. (2024) — Euclid strong
            lens searches, showing that models trained on simulated/controlled
            data can be harder to apply to real survey images (directly
            motivating this project's use of real Legacy Survey imagery).</li>
            <li><b>Grad-CAM:</b> Selvaraju et al. (2017) — gradient-based visual
            explanations used throughout this app's explainability features.</li>
            <li><b>Simulated data tooling:</b> lenstronomy / deeplenstronomy
            (Birrer &amp; Amara, 2018) — used for this project's simulated
            baseline model, for comparison against the real-data model shown
            here.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 8. Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer-card">
        Built by Ahana Bhattacharji &amp; Vaishnav Malvankar ·
        Explainable Deep Learning for Gravitational Lens Detection in
        Astronomical Survey Images · Student research prototype, not a
        validated scientific tool.
    </div>
    """,
    unsafe_allow_html=True,
)