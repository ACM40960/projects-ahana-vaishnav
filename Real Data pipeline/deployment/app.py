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
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import streamlit.components.v1 as components


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


import base64


@st.cache_data
def get_base64_image(image_path: Path) -> str | None:
    """Reads a local image file and returns it as a base64 data URI string,
    so it can be embedded directly in CSS (background-image: url(...))
    without needing a separate public URL. Returns None if the file is
    missing, so callers can fall back gracefully instead of crashing."""
    if not image_path.exists():
        return None
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{encoded}"


HERO_IMAGE_PATH = APP_DIR / "website images" / "main_screen_img.jpg"
hero_image_data_uri = get_base64_image(HERO_IMAGE_PATH)


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
# 4. Page setup + dark space/science theme & Animated Starfield Background
# --------------------------------------------------

st.set_page_config(
    page_title="Gravitational Lens Detector",
    layout="centered",
)

st.html(textwrap.dedent("""
<style>
/* ===============================
   DESIGN TOKENS — single source of truth for spacing,
   radius, elevation, motion, and accent colors. Every
   component below references these instead of hardcoding
   its own one-off values.
   =============================== */
html {
    scroll-behavior: smooth;
}
.scroll-anchor {
    display: block;
    scroll-margin-top: 90px;
}
:root {
    --radius-sm: 10px;   /* buttons, chips */
    --radius-md: 16px;   /* cards */
    --radius-lg: 24px;   /* hero only */
    --space-1: 0.5rem;
    --space-2: 0.75rem;
    --space-3: 1rem;
    --space-4: 1.25rem;
    --space-5: 1.5rem;
    --space-6: 2rem;
    --shadow-sm: 0 8px 24px rgba(0, 0, 0, 0.22);
    --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.45);
    --transition: all 0.2s ease;
    --accent-blue: #38bdf8;
    --accent-indigo: #818cf8;
    --accent-purple: #c4b5fd;
    --accent-pink: #f472b6;
    --accent-green: #34d399;
    --accent-amber: #fbbf24;
    --text-primary: #f8fafc;
    --text-body: #dbeafe;
    --text-muted: #94a3b8;
    --border-subtle: rgba(148, 163, 184, 0.22);
    --border-strong: rgba(148, 163, 184, 0.28);
}
/* ===============================
   NEW: Animated Pure CSS Starfield
   =============================== */
.starfield {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
@keyframes twinkle {
    0%, 100% { opacity: 0.20; }
    50% { opacity: 0.65; }
}
@keyframes slow-drift {
    from { transform: translateY(0px) translateX(0px); }
    to { transform: translateY(-100px) translateX(-35px); }
}
.stars-sm {
    width: 1px;
    height: 1px;
    background: transparent;
    box-shadow: 
        100px 150px #e0f2fe, 350px 420px #93c5fd, 700px 800px #c4b5fd, 950px 200px #e0f2fe, 
        1250px 650px #93c5fd, 180px 780px #e0f2fe, 480px 180px #a5f3fc, 820px 520px #e0f2fe, 
        1100px 880px #93c5fd, 1380px 320px #a5f3fc, 280px 920px #e0f2fe, 580px 360px #c4b5fd, 
        780px 110px #a5f3fc, 1180px 450px #e0f2fe, 1420px 810px #93c5fd;
    animation: twinkle 5s ease-in-out infinite alternate, slow-drift 90s linear infinite;
    opacity: 0.35;
}
.stars-md {
    width: 2px;
    height: 2px;
    background: transparent;
    border-radius: 50%;
    box-shadow: 
        210px 340px #38bdf8, 550px 120px #818cf8, 880px 680px #a5f3fc, 1150px 250px #38bdf8, 
        410px 850px #818cf8, 1010px 420px #a5f3fc, 1320px 900px #38bdf8;
    animation: twinkle 7s ease-in-out infinite alternate, slow-drift 140s linear infinite;
    opacity: 0.28;
}
.stars-lg {
    width: 2.5px;
    height: 2.5px;
    background: transparent;
    border-radius: 50%;
    box-shadow: 
        300px 530px #e0f2fe, 760px 290px #c4b5fd, 1050px 750px #93c5fd, 500px 210px #e0f2fe;
    animation: twinkle 4s ease-in-out infinite alternate;
    opacity: 0.35;
}
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
    padding-top: 0;
    padding-bottom: var(--space-6);
    position: relative;
    z-index: 1;
}
h1, h2, h3 {
    color: #f8fafc;
    letter-spacing: -0.03em;
}
p, li, span, div {
    color: var(--text-body);
}
/* Hero panel */
.hero-card {
    position: relative;
    overflow: hidden;
    /* Full-bleed: breaks out of the centered/max-width content column
       to span the full browser viewport width, regardless of screen
       size — the standard CSS trick for an edge-to-edge section inside
       a constrained parent. */
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    margin-right: calc(-50vw + 50%);
    padding: var(--space-6) 6vw calc(var(--space-6) * 3.5);
    border-radius: 0;
    /* Real astronomical photograph (NASA/ESA/CSA/STScI, public domain)
       with a dark gradient overlay so headline text stays legible —
       replaces the flat CSS gradient with an actual space image,
       matching the reference template's photographic hero style. */
    background:
        linear-gradient(115deg, rgba(3, 7, 18, 0.88) 0%, rgba(3, 7, 18, 0.55) 45%, rgba(3, 7, 18, 0.82) 100%),
        url('https://www.nasa.gov/wp-content/uploads/2022/12/artemis_i_earth_after_opf.jpg');
    background-size: cover;
    background-position: center 22%;
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5);
    margin-bottom: var(--space-5);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}
.hero-image-credit {
    font-size: 0.72rem;
    color: rgba(203, 213, 225, 0.65);
    letter-spacing: 0.02em;
    margin-top: var(--space-2);
}
.hero-image-credit a {
    color: rgba(147, 197, 253, 0.75);
    text-decoration: underline;
}
/* NEW: minimal vertical nav-style list, bottom-right of the hero —
   decorative labels mirroring the app's tab sections (actual
   navigation still happens via the st.tabs widget below). */
.hero-nav-list {
    position: absolute;
    bottom: var(--space-5);
    right: 2.2rem;
    text-align: right;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    z-index: 2;
}
.hero-nav-item {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(226, 232, 240, 0.75);
    text-decoration: none;
    cursor: pointer;
    transition: var(--transition);
}
.hero-nav-item:hover {
    color: #ffffff;
    text-shadow: 0 0 12px rgba(147, 197, 253, 0.6);
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: var(--space-1);
    background: linear-gradient(90deg, #e0f2fe, var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-body);
    max-width: 850px;
    line-height: 1.7;
}
/* Generic content cards */
.glass-card {
    padding: var(--space-4) var(--space-5);
    border-radius: var(--radius-md);
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid var(--border-subtle);
    box-shadow: var(--shadow-sm);
    margin-bottom: var(--space-3);
    transition: var(--transition);
}
.glass-card:hover {
    border-color: rgba(147, 197, 253, 0.35);
}
/* Small feature cards (Why This Matters) */
.feature-card {
    padding: var(--space-4);
    border-radius: var(--radius-md);
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid var(--border-subtle);
    height: 100%;
    min-height: 150px;
    transition: var(--transition);
}
.feature-card:hover {
    transform: translateY(-2px);
    border-color: rgba(147, 197, 253, 0.4);
    box-shadow: var(--shadow-sm);
}
.feature-card-title {
    font-weight: 700;
    font-size: 1.02rem;
    color: var(--accent-blue);
    margin-bottom: var(--space-1);
}
.feature-card-body {
    font-size: 0.88rem;
    color: var(--text-body);
    line-height: 1.5;
}
/* Pipeline step chips */
.pipeline-row {
    display: flex;
    align-items: stretch;
    justify-content: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: var(--space-3);
}
.pipeline-step {
    flex: 1 1 130px;
    max-width: 160px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.3rem;
    padding: var(--space-3) var(--space-2);
    border-radius: var(--radius-md);
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(147, 197, 253, 0.3);
    text-align: center;
    font-size: 0.82rem;
    color: #e0f2fe;
    min-height: 90px;
    transition: var(--transition);
}
.pipeline-step:hover {
    transform: translateY(-2px);
    border-color: var(--accent-blue);
}
.pipeline-step-icon {
    font-size: 1.3rem;
}
.pipeline-arrow {
    align-self: center;
    flex: 0 0 auto;
    font-size: 1.4rem;
    color: var(--accent-blue);
}
.warning-card {
    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius-md);
    background: rgba(120, 53, 15, 0.30);
    border: 1px solid rgba(251, 191, 36, 0.35);
    color: #fde68a;
    margin-top: var(--space-3);
}
.footer-card {
    padding: var(--space-3) var(--space-5);
    border-radius: var(--radius-md);
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.18);
    color: var(--text-muted);
    font-size: 0.85rem;
    text-align: center;
    margin-top: var(--space-5);
}
/* Prediction cards */
.prediction-lens {
    padding: var(--space-5);
    border-radius: var(--radius-md);
    background: linear-gradient(135deg, rgba(88, 28, 135, 0.8), rgba(30, 64, 175, 0.78));
    border: 1px solid rgba(196, 181, 253, 0.45);
    box-shadow: var(--shadow-lg);
    margin-bottom: var(--space-3);
}
.prediction-nonlens {
    padding: var(--space-5);
    border-radius: var(--radius-md);
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 64, 175, 0.45));
    border: 1px solid rgba(96, 165, 250, 0.35);
    box-shadow: var(--shadow-lg);
    margin-bottom: var(--space-3);
}
.prediction-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: var(--space-1);
}
.prediction-subtitle {
    font-size: 0.95rem;
    color: var(--text-body);
}
/* Widgets */
.stButton button {
    width: 100%;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(147, 197, 253, 0.35);
    background: rgba(15, 23, 42, 0.82);
    color: var(--text-body);
    transition: var(--transition);
}
.stButton button:hover {
    border-color: #93c5fd;
    color: #ffffff;
    background: rgba(30, 64, 175, 0.85);
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
}
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.68);
    border: 1px dashed rgba(147, 197, 253, 0.45);
    border-radius: var(--radius-md);
    padding: var(--space-3);
    transition: var(--transition);
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue);
}
[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
}
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: var(--space-2) var(--space-3);
    transition: var(--transition);
}
[data-testid="stMetric"]:hover {
    border-color: rgba(147, 197, 253, 0.35);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(15, 23, 42, 0.6);
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    padding: var(--space-1) var(--space-3);
    transition: var(--transition);
}
/* Custom section nav bar (replaces st.tabs — needed so the hero's nav
   links can actually switch sections, which st.tabs cannot do
   programmatically). Styled to look like the previous tab bar. */
.section-nav-bar {
    display: flex;
    gap: 4px;
    border-bottom: none;
    flex-wrap: wrap;
    justify-content: center;
    /* Fixed: pins to the browser viewport directly. Hidden by default
       (opacity/visibility) — a small script toggles the "visible" class
       once the user scrolls past the hero, so the bar stays out of the
       way while on the full-screen hero and appears only once there's
       somewhere else on the page to jump to. */
    position: fixed;
    top: 52px;
    left: 0;
    right: 0;
    width: 100%;
    z-index: 999;
    /* Transparent container — no boxed background of its own, so there's
       no rigid rectangle with visible edges. Each individual nav item
       still has its own subtle background (below) for legibility. */
    background: transparent;
    padding: 0.6rem 0;
    opacity: 0;
    visibility: hidden;
    transform: translateY(-12px);
    transition: opacity 0.25s ease, transform 0.25s ease, visibility 0.25s;
}
.section-nav-bar.visible {
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
}
/* Reserves space where the nav bar used to sit in normal document flow,
   so content isn't hidden underneath the now-fixed bar. */
.section-nav-spacer {
    height: 64px;
}
.section-nav-item {
    display: inline-block;
    background: rgba(15, 23, 42, 0.6);
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    padding: var(--space-2) var(--space-3);
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-body);
    text-decoration: none;
    border-bottom: 2px solid transparent;
    transition: var(--transition);
}
.section-nav-item:hover {
    color: #ffffff;
    background: rgba(30, 64, 175, 0.35);
}
.section-nav-item.active {
    color: var(--text-primary);
    border-bottom: 2px solid var(--accent-blue);
    background: rgba(30, 64, 175, 0.25);
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
<!-- Animated Starfield HTML Markup -->
<div class="starfield">
    <div class="stars-sm"></div>
    <div class="stars-md"></div>
    <div class="stars-lg"></div>
</div>
"""))

# Hero background image: use the local file if present, otherwise fall
# back to the NASA public-domain URL. Injected as a separate small style
# override (rather than editing the big CSS block above) so the dynamic
# Python variable doesn't require turning that whole block into an
# f-string.
_hero_bg = hero_image_data_uri or "https://www.nasa.gov/wp-content/uploads/2022/12/artemis_i_earth_after_opf.jpg"

st.html(textwrap.dedent(f"""
<style>
.hero-card {{
    background:
        linear-gradient(115deg, rgba(3, 7, 18, 0.88) 0%, rgba(3, 7, 18, 0.55) 45%, rgba(3, 7, 18, 0.82) 100%),
        url('{_hero_bg}') !important;
    background-size: cover !important;
    background-position: center 22% !important;
}}
</style>
"""))


# --------------------------------------------------
# 5. HEADER — title + subtitle + constellation accent
# --------------------------------------------------

st.html(textwrap.dedent("""
    <div class="hero-card">
        <!-- Minimal vertical nav-style list, bottom-right (template-inspired layout) -->
        <div class="hero-nav-list">
            <a href="#overview" class="hero-nav-item">Overview</a>
            <a href="#how-it-works" class="hero-nav-item">How It Works</a>
            <a href="#try-the-model" class="hero-nav-item">Try The Model</a>
            <a href="#interpretation" class="hero-nav-item">Interpretation</a>
        </div>
        <div class="hero-title">Explainable Deep Learning for Gravitational Lens Detection</div>
        <div class="hero-subtitle">
            A student prototype for classifying astronomical survey image cutouts as
            <b>lens</b> or <b>non-lens</b> candidates.
        </div>
        <div class="hero-image-credit">
            Background image: NASA — Earthrise, Artemis I (Orion spacecraft, Dec. 2022)
            (<a href="https://www.nasa.gov/humans-in-space/view-the-best-images-from-nasas-artemis-i-mission/" target="_blank">public domain</a>)
        </div>
    </div>
"""))


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

# Section definitions: (anchor id, icon, label). All sections render on
# one continuous page — nav links jump/smooth-scroll to their #anchor
# rather than triggering a rerun, so switching sections feels instant
# and doesn't reload anything.
SECTIONS = [
    ("overview", "Overview"),
    ("how-it-works", "How It Works"),
    ("model-comparison", "Model Comparison"),
    ("try-the-model", "Try the Model"),
    ("interpretation", "Interpretation & Notes"),
    ("references", "References"),
]

_nav_items_html = "".join(
    f'<a href="#{key}" class="section-nav-item">{label}</a>'
    for key, label in SECTIONS
)
st.html(f'<div class="section-nav-bar" id="section-nav-bar">{_nav_items_html}</div>')
st.html('<div class="section-nav-spacer"></div>')

# Toggles the nav bar's visibility based on scroll position: hidden while
# on the full-screen hero, shown once scrolled past it. Pure client-side
# JS — no Streamlit rerun involved, this never touches the Python side.
# st.html() inserts content via innerHTML-style DOM insertion, and
# browsers never execute <script> tags inserted that way (true in every
# browser, not a Streamlit restriction) — that's why earlier attempts
# via st.html() silently did nothing. components.v1.html() renders via
# an iframe's srcdoc instead, which DOES execute scripts normally. The
# script reaches across into the real page via window.parent, since the
# iframe itself is a separate (same-origin) document.
components.html(
    """
    <script>
    (function () {
        function getBar() {
            return window.parent.document.getElementById("section-nav-bar");
        }

        function currentScrollTop(eventTarget) {
            if (eventTarget && typeof eventTarget.scrollTop === "number"
                && eventTarget !== window.parent.document) {
                return eventTarget.scrollTop;
            }
            return window.parent.scrollY
                || window.parent.document.documentElement.scrollTop
                || window.parent.document.body.scrollTop
                || 0;
        }

        function toggleNavBar(eventTarget) {
            var bar = getBar();
            if (!bar) return;
            var threshold = window.parent.innerHeight * 0.6;
            var scrollTop = currentScrollTop(eventTarget);
            if (scrollTop > threshold) {
                bar.classList.add("visible");
            } else {
                bar.classList.remove("visible");
            }
        }

        // Scroll-spy: highlights whichever section is currently in view
        // by checking each anchor's position relative to the top of the
        // viewport (just below the fixed nav bar), not just on click.
        var SECTION_IDS = [
            "overview", "how-it-works", "model-comparison",
            "try-the-model", "interpretation", "references"
        ];

        function updateActiveSection() {
            var activeId = SECTION_IDS[0];
            var offset = 110; // roughly the fixed nav bar's height + margin

            for (var i = 0; i < SECTION_IDS.length; i++) {
                var el = window.parent.document.getElementById(SECTION_IDS[i]);
                if (!el) continue;
                var top = el.getBoundingClientRect().top;

                // The last section has no content below it to scroll past,
                // so the strict "just below the nav bar" threshold other
                // sections use may never be reachable for it. Use a much
                // more generous threshold (roughly the upper half of the
                // viewport) only for this final section, based on the
                // same anchor-position approach already working correctly
                // for every other section — not on page/document height,
                // which measures the wrong (non-scrolling) element here.
                var isLast = (i === SECTION_IDS.length - 1);
                var sectionThreshold = isLast ? window.parent.innerHeight * 0.5 : offset;

                if (top - sectionThreshold <= 0) {
                    activeId = SECTION_IDS[i];
                }
            }

            var navItems = window.parent.document.querySelectorAll(".section-nav-item");
            navItems.forEach(function (item) {
                var isActive = item.getAttribute("href") === "#" + activeId;
                item.classList.toggle("active", isActive);
            });
        }

        // Capture phase catches scroll events from any nested scrollable
        // ancestor in the PARENT page, without needing to know exactly
        // which element Streamlit uses to scroll internally.
        function onScroll(eventTarget) {
            toggleNavBar(eventTarget);
            updateActiveSection();
        }

        window.parent.document.addEventListener(
            "scroll",
            function (e) { onScroll(e.target); },
            true
        );
        window.parent.addEventListener(
            "scroll",
            function () { onScroll(window.parent); },
            { passive: true }
        );

        onScroll(window.parent);

        // Smooth-scroll nav clicks: scrollIntoView() correctly handles
        // whichever element actually scrolls (unlike CSS scroll-behavior,
        // which only smooths the specific element it's set on — and
        // that turned out to be the wrong one here, same root cause as
        // the nav bar visibility bug above). preventDefault stops the
        // browser's own instant jump so only the smooth version runs.
        window.parent.document.addEventListener("click", function (e) {
            var link = e.target.closest(".hero-nav-item, .section-nav-item");
            if (!link) return;

            var href = link.getAttribute("href") || "";
            if (href.charAt(0) !== "#") return;

            var targetId = href.slice(1);
            var targetEl = window.parent.document.getElementById(targetId);
            if (!targetEl) return;

            e.preventDefault();
            targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
        });
    })();
    </script>
    """,
    height=0,
)


# ====================================================
# TAB 1 — Project Overview + Why This Matters
# ====================================================

st.html('<div id="overview" class="scroll-anchor"></div>')
if True:

    # ---- Project Overview ----
    st.markdown("### Project Overview")
    st.html(textwrap.dedent("""
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
"""))

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
        ("Rare-Object Detection",
         "Genuine strong lenses are a tiny fraction of all survey objects, "
         "making this a hard imbalanced-classification problem, not a "
         "simple binary split."),
        ("False Positives",
         "Spiral galaxies, ring galaxies, mergers, bright stars, and imaging "
         "artefacts can all visually resemble lensing features."),
        ("Astronomical Survey Scale",
         "Modern surveys produce millions of image cutouts — manual "
         "inspection by expert astronomers cannot scale to this volume."),
        ("Explainable AI",
         "Grad-CAM visualises which image regions drove each prediction, "
         "supporting scrutiny of the model rather than blind trust."),
    ]

    for col, (title, body) in zip([why_col1, why_col2, why_col3, why_col4], why_cards):
        with col:
            st.html(textwrap.dedent(f"""
                <div class="feature-card">
                    <div class="feature-card-title">{title}</div>
                    <div class="feature-card-body">{body}</div>
                </div>
"""))


# ====================================================
# TAB 2 — How the Pipeline Works
# ====================================================

st.html('<div id="how-it-works" class="scroll-anchor"></div>')
if True:
    st.markdown("### How the Pipeline Works")
    st.caption("The same flow runs every time you submit an image in the 'Try the Model' tab.")

    steps = [
        "Image input",
        "Preprocessing (resize, normalise)",
        "CNN prediction",
        "Confidence score",
        "Grad-CAM heatmap",
        "Interpretation",
    ]

    # Single flex row (not separate Streamlit columns) so all step-cards
    # reliably stretch to match the tallest one's height, and arrows
    # center themselves against that — a fixed guessed offset couldn't
    # do this correctly once cards wrap to different numbers of lines.
    _step_html_parts = []
    for i, label in enumerate(steps):
        _step_html_parts.append(f'<div class="pipeline-step"><div>{label}</div></div>')
        if i < len(steps) - 1:
            _step_html_parts.append('<div class="pipeline-arrow">\u2192</div>')

    st.html(f'<div class="pipeline-row">{"".join(_step_html_parts)}</div>')

    st.divider()
    st.html(textwrap.dedent("""
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
"""))


# ====================================================
# TAB 3 — Model Comparison Visualization
# ====================================================

st.html('<div id="model-comparison" class="scroll-anchor"></div>')
if True:
    st.markdown("### Model Performance Comparison")
    st.caption("Quantitative comparison of test metrics across evaluated candidate models.")

    # Hardcoded test metrics dictionary
    model_metrics = {
        "Model Architecture": [
            "Baseline CNN (Simulated Data)",
            "Custom CNN (Real Data)",
            "Brightness-Mitigated CNN",
            "CBAM-Attention CNN",
            "MobileNetV3 (Transfer Learning)",
            "ViT-B/16 (Transfer Learning)",
        ],
        "ROC-AUC": [0.998, 0.983, 0.978, 0.986, 0.981, 0.988],
        "Accuracy": [0.992, 0.944, 0.936, 0.951, 0.942, 0.955],
        "Precision": [0.991, 0.938, 0.929, 0.945, 0.935, 0.950],
        "Recall": [0.994, 0.942, 0.935, 0.950, 0.940, 0.953],
        "F1-Score": [0.992, 0.940, 0.932, 0.947, 0.937, 0.951],
    }

    df_metrics = pd.DataFrame(model_metrics)

    metric_choice = st.selectbox(
        "Select metric to visualize:",
        ["ROC-AUC", "Accuracy", "F1-Score", "Precision", "Recall"],
        index=0
    )

    # Matplotlib bar chart styled to match the dark glass card theme
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    y_pos = np.arange(len(df_metrics["Model Architecture"]))
    values = df_metrics[metric_choice]

    # Uses the same accent tokens defined in the CSS design system above,
    # so the chart palette matches the rest of the UI instead of being a
    # separate, disconnected color set.
    bar_colors = ["#38bdf8", "#818cf8", "#c4b5fd", "#f472b6", "#34d399", "#fbbf24"]

    bars = ax.barh(y_pos, values, color=bar_colors, height=0.55, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_metrics["Model Architecture"], color="#e2e8f0", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(metric_choice, color="#cbd5e1", fontsize=11)
    ax.set_xlim(0.85, 1.01)

    ax.xaxis.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#cbd5e1")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            color="#f8fafc",
            fontsize=9.5,
            fontweight="bold"
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.markdown("#### Detailed Test Metrics")
    df_display = df_metrics.copy()
    for col in ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)


# ====================================================
# TAB 4 — Try the Model (core functionality, UNCHANGED logic)
# ====================================================

st.html('<div id="try-the-model" class="scroll-anchor"></div>')
if True:
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
            "No sample images found in `sample_images/` — you can still upload "
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
            st.html(textwrap.dedent(f"""
                <div class="prediction-lens">
                    <div class="prediction-title">Prediction: {predicted_label}</div>
                    <div class="prediction-subtitle">
                        The model output is above the validation-tuned threshold.
                        Treat this as a candidate flag, not a scientific confirmation.
                    </div>
                </div>
"""))
        else:
            st.html(textwrap.dedent(f"""
                <div class="prediction-nonlens">
                    <div class="prediction-title">Prediction: {predicted_label}</div>
                    <div class="prediction-subtitle">
                        The model output is below the validation-tuned lens threshold.
                        This does not prove the object is not a lens.
                    </div>
                </div>
"""))

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
# TAB 5 — How to Interpret the Result + Model/Dataset Notes + Limitations
# ====================================================

st.html('<div id="interpretation" class="scroll-anchor"></div>')
if True:
    st.markdown("### How to Interpret the Result")
    st.html(textwrap.dedent("""
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
"""))

    st.divider()

    st.markdown("### Model and Dataset Notes")
    st.html(textwrap.dedent("""
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
"""))

    st.divider()

    with st.expander("Known limitations of this model", expanded=False):
        st.html(textwrap.dedent("""
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
"""))

    with st.expander("What we found — project insights", expanded=False):
        st.html(textwrap.dedent("""
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
"""))


# ====================================================
# TAB 6 — References / Research Context
# ====================================================

st.html('<div id="references" class="scroll-anchor"></div>')
if True:
    st.markdown("### References / Research Context")
    st.html(textwrap.dedent("""
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
"""))


# --------------------------------------------------
# 8. Footer
# --------------------------------------------------

st.html(textwrap.dedent("""
    <div class="footer-card">
        Built by Ahana Bhattacharji &amp; Vaishnav Malvankar ·
        Explainable Deep Learning for Gravitational Lens Detection in
        Astronomical Survey Images · Student research prototype, not a
        validated scientific tool.
    </div>
"""))