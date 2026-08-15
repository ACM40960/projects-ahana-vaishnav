# Gravitational Lens Candidate Detector
# Ahana Bhattacharji & Vaishnav Malvankar
# Run: streamlit run app.py
from pathlib import Path
import io
import json
import textwrap
import base64
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import torch
import torch.nn as nn
APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_weights.pt"
CONFIG_PATH = ARTIFACTS_DIR / "preprocessing_config.json"
SAMPLES_DIR = APP_DIR / "sample_images"
HERO_IMAGE_PATH = APP_DIR / "website images" / "main_screen_img.jpg"
FIGURES_DIR = APP_DIR / "website images"
IMAGE_SIZE = 128

# Static result figures exported by the training/eval scripts (results/*/).
# Copy the underlying PNGs into "website images/" so they ship with the
# deployment folder regardless of where results/ lives on disk.
RESULT_FIGURES = {
    "dataset_sample_grid": FIGURES_DIR / "dataset_sample_grid.png",
    "brightness_confound": FIGURES_DIR / "brightness_vs_prediction.png",
    "brightness_confound_mitigated": FIGURES_DIR / "brightness_vs_prediction_mitigated.png",
    "fp_fn_gallery": FIGURES_DIR / "false_positive_negative_gallery.png",
}


def show_result_figure(fig_key: str, caption: str = "", missing_hint: str = ""):
    """Render a pre-generated result figure if present, otherwise a quiet
    hint instead of a hard error (keeps the app usable if a PNG hasn't
    been copied into website images/ yet)."""
    path = RESULT_FIGURES.get(fig_key)
    if path and path.exists():
        st.image(str(path), use_container_width=True)
        if caption:
            st.caption(caption)
    else:
        name = path.name if path else fig_key
        st.caption(
            f"Figure not found: `{name}`. "
            + (missing_hint or f"Copy it into `{FIGURES_DIR.name}/` to display it here.")
        )
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
class RealDataCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding="same"), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding="same"), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x).squeeze(1)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels))
    def forward(self, x):
        avg_pool, max_pool = x.mean(dim=(2, 3)), x.amax(dim=(2, 3))
        return x * torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))[:, :, None, None]
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1))), None
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
    def forward(self, x):
        x = self.channel_attention(x)
        x, _ = self.spatial_attention(x)
        return x
class CBAMAttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
        self.cbam1 = CBAM(16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.cbam2 = CBAM(32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.cbam3 = CBAM(64)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.cbam1(x)
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.cbam2(x)
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.cbam3(x)
        return self.classifier(self.gap(x).flatten(1)).squeeze(1)
class MobileNetTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone
        in_features = backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, 1)
    def forward(self, x):
        return self.backbone(x).squeeze(1)
from torchvision.models import vit_b_16, ViT_B_16_Weights
class ViTTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(768, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1)
        )
    def forward(self, x):
        features = self.backbone._process_input(x)
        batch_size = features.shape[0]
        batch_class_token = self.backbone.class_token.expand(batch_size, -1, -1)
        features = torch.cat([batch_class_token, features], dim=1)
        features = self.backbone.encoder(features)
        cls_token = features[:, 0]
        return self.classifier(cls_token).squeeze(1)
MODEL_REGISTRY = {
    "cnn": {
        "label": "Custom CNN",
        "arch": RealDataCNN,
        "weights": "cnn_weights.pt",
        "config": "cnn_config.json",
        "gradcam_target": lambda m: m.features[6],
        "available": True,
        # Metrics below are derived from results/real_data_cnn/test_metrics.json's
        # confusion_matrix_best_threshold, so they match the confusion matrix exactly.
        "roc_auc": 0.983, "accuracy": 0.944, "threshold": 0.41,
        "precision": 0.940, "recall": 0.943,
        "example_grid": "gradcam_grid.png",
        "dev_note": "This is where everything started — a simple three-block CNN built from scratch on real Legacy Survey imagery. No pretrained weights, no borrowed architecture. Just Conv→Pool×2→GAP→Dense. The fact that it reaches 0.983 ROC-AUC against models 10× more complex is probably the most satisfying result in this whole project.",
        "arch_layers": [
            ("Input\n128×128×3", "#1e3a5f"),
            ("Conv2d\n16 filters", "#164e63"),
            ("MaxPool\n64×64", "#164e63"),
            ("Conv2d\n32 filters", "#164e63"),
            ("MaxPool\n32×32", "#164e63"),
            ("Conv2d\n64 filters", "#164e63"),
            ("GAP\n64-dim", "#0e4429"),
            ("Dense\n32 + Drop", "#0e4429"),
            ("Output\nLogit", "#3b0764"),
        ],
    },
    "cbam": {
        "label": "CBAM-Attention CNN",
        "arch": CBAMAttentionCNN,
        "weights": "cbam_weights.pt",
        "config": "cbam_config.json",
        "gradcam_target": lambda m: m.conv3,
        "available": True,
        # Metrics below are derived from results/cbam_attention_cnn/test_metrics.json's
        # confusion_matrix_best_threshold, so they match the confusion matrix exactly.
        "roc_auc": 0.985, "accuracy": 0.932, "threshold": 0.38,
        "precision": 0.906, "recall": 0.957,
        "example_grid": "cbam_attention_grid.png",
        "dev_note": "Our personal favourite. CBAM inserts channel and spatial attention after each conv block — so the model learns where to look as part of training, not just as a post-hoc explanation. When you look at the Grad-CAM heatmaps here, you are seeing something the network genuinely paid attention to during training. That matters more to us than the extra 0.007 ROC-AUC ViT has over it — though it does trade some precision for that higher recall, so it flags more false positives than CNN or ViT.",
        "arch_layers": [
            ("Input\n128×128×3", "#1e3a5f"),
            ("Conv2d\n16 filters", "#164e63"),
            ("CBAM\nAttention", "#7c2d12"),
            ("MaxPool\n64×64", "#164e63"),
            ("Conv2d\n32 filters", "#164e63"),
            ("CBAM\nAttention", "#7c2d12"),
            ("MaxPool\n32×32", "#164e63"),
            ("Conv2d\n64 filters", "#164e63"),
            ("CBAM\nAttention", "#7c2d12"),
            ("GAP\n64-dim", "#0e4429"),
            ("Dense\n32 + Drop", "#0e4429"),
            ("Output\nLogit", "#3b0764"),
        ],
    },
    "mobilenet": {
        "label": "MobileNetV3 (Transfer Learning)",
        "arch": MobileNetTransfer,
        "weights": "mobilenet_weights.pt",
        "config": "mobilenet_config.json",
        "gradcam_target": lambda m: m.backbone.features[12][0],
        "available": True,
        # Metrics below are derived from results/transfer_learning_cnn/test_metrics.json's
        # confusion_matrix_best_threshold, so they match the confusion matrix exactly.
        "roc_auc": 0.981, "accuracy": 0.947, "threshold": 0.47,
        "precision": 0.947, "recall": 0.943,
        "dev_note": "We froze the entire MobileNetV3 backbone pretrained on ImageNet and only trained a small classification head on top. The fact that ImageNet features — learned on everyday photos of cats and cars — transfer this well to astronomical survey images is genuinely surprising. It still scores slightly below our custom CNN though. Domain gap is real.",
        "arch_layers": [
            ("Input\n128×128×3", "#1e3a5f"),
            ("MobileNetV3\nBackbone\n(frozen)", "#1a3a1a"),
            ("Avg Pool\n576-dim", "#0e4429"),
            ("Dense\n1024", "#0e4429"),
            ("Hardswish\n+ Drop", "#0e4429"),
            ("Dense\n1 (head)\ntrainable", "#3b0764"),
            ("Output\nLogit", "#3b0764"),
        ],
    },
    "vit": {
        "label": "ViT-B/16 (Transfer Learning)",
        "arch": ViTTransferModel,
        "weights": "vit_head_weights.pt",
        "config": None,
        "gradcam_target": lambda m: m.backbone.encoder.layers[-1].ln_1,
        "available": True,
        # Metrics below are derived from results/vit_transfer_learning/test_metrics.json's
        # confusion_matrix_best_threshold, so they match the confusion matrix exactly.
        "roc_auc": 0.992, "accuracy": 0.963, "threshold": 0.65,
        "input_size": 224,
        "precision": 0.985, "recall": 0.936,
        "example_grid": "vit_attention_grid.png",
        "dev_note": "Best model by every metric — 0.992 ROC-AUC and only 4 false positives out of 306 non-lens test images, which is why precision (0.985) is its standout number. ViT sees the whole image at once through self-attention across 196 patches, which suits ring and arc detection better than local conv filters. The full weights are 330MB so we only ship the 98KB classification head and re-download the frozen backbone on first load. Worth the wait.",
        "arch_layers": [
            ("Input\n224×224×3", "#1e3a5f"),
            ("Patch Split\n196 patches\n16×16 each", "#1e3a5f"),
            ("Patch\nEmbedding\n768-dim", "#164e63"),
            ("CLS Token\n+ Pos Embed", "#164e63"),
            ("Transformer\nEncoder\n×12 layers\n(frozen)", "#1a3a1a"),
            ("CLS Token\n768-dim", "#0e4429"),
            ("Dense\n32 + Drop\n(trainable)", "#0e4429"),
            ("Output\nLogit", "#3b0764"),
        ],
    },
}
def draw_architecture(model_key: str, max_cols: int = 4):
    """Draw a snake/boustrophedon flowchart: rows fill left->right, then
    right->left, wrapping until every layer is placed."""
    layers = MODEL_REGISTRY[model_key]["arch_layers"]
    n = len(layers)
    cols = min(max_cols, n)
    rows = -(-n // cols)  # ceil division

    box_w, box_h = 1.55, 1.05
    gap_x, gap_y = 0.55, 0.85
    fig_w = cols * box_w + (cols + 1) * gap_x
    fig_h = rows * box_h + (rows + 1) * gap_y

    fig, ax = plt.subplots(figsize=(fig_w, max(fig_h, 2.2)))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#0a0f1e")
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()  # row 0 at the top
    ax.axis("off")

    # Work out each layer's (x_center, y_center), snaking row by row.
    positions = []
    for i in range(n):
        row = i // cols
        col_in_row = i % cols
        display_col = (cols - 1 - col_in_row) if (row % 2 == 1) else col_in_row
        x_center = gap_x + display_col * (box_w + gap_x) + box_w / 2
        y_center = gap_y + row * (box_h + gap_y) + box_h / 2
        positions.append((x_center, y_center))

    # Boxes + labels
    for (label, color), (x_center, y_center) in zip(layers, positions):
        rect = mpatches.FancyBboxPatch(
            (x_center - box_w / 2, y_center - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.04",
            linewidth=1.2,
            edgecolor="#38bdf8",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x_center, y_center, label,
                 ha="center", va="center",
                 fontsize=7.5, color="#e0f2fe",
                 fontweight="bold", linespacing=1.3)

    # Arrows: horizontal within a row, vertical between rows
    for i in range(n - 1):
        x0, y0 = positions[i]
        x1, y1 = positions[i + 1]
        row0, row1 = i // cols, (i + 1) // cols
        if row0 == row1:
            direction = 1 if (row0 % 2 == 0) else -1
            ax.annotate(
                "",
                xy=(x1 - direction * box_w / 2 * 1.03, y1),
                xytext=(x0 + direction * box_w / 2 * 1.03, y0),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.2),
            )
        else:
            ax.annotate(
                "",
                xy=(x1, y1 - box_h / 2 * 1.05),
                xytext=(x0, y0 + box_h / 2 * 1.05),
                arrowprops=dict(
                    arrowstyle="->", color="#38bdf8", lw=1.2,
                    connectionstyle="arc3,rad=0.0",
                ),
            )

    plt.tight_layout(pad=0.3)
    return fig


def render_active_model_snapshot(reg):
    """Metric cards + snake architecture diagram + developer note for the
    currently selected model. Called from both Overview and Model Comparison."""
    st.html(textwrap.dedent(f"""
<div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin:1rem 0;">
    <div style="flex:1;min-width:140px;max-width:200px;background:rgba(15,23,42,0.78);border:1px solid rgba(148,163,184,0.22);border-radius:16px;padding:1rem;text-align:center;">
        <div style="font-size:0.75rem;color:#94a3b8;letter-spacing:0.05em;margin-bottom:0.4rem;">ROC-AUC</div>
        <div style="font-size:2rem;font-weight:800;color:#38bdf8;">{reg['roc_auc']:.3f}</div>
    </div>
    <div style="flex:1;min-width:140px;max-width:200px;background:rgba(15,23,42,0.78);border:1px solid rgba(148,163,184,0.22);border-radius:16px;padding:1rem;text-align:center;">
        <div style="font-size:0.75rem;color:#94a3b8;letter-spacing:0.05em;margin-bottom:0.4rem;">Accuracy</div>
        <div style="font-size:2rem;font-weight:800;color:#38bdf8;">{reg['accuracy']:.1%}</div>
    </div>
    <div style="flex:1;min-width:140px;max-width:200px;background:rgba(15,23,42,0.78);border:1px solid rgba(148,163,184,0.22);border-radius:16px;padding:1rem;text-align:center;">
        <div style="font-size:0.75rem;color:#94a3b8;letter-spacing:0.05em;margin-bottom:0.4rem;">Precision</div>
        <div style="font-size:2rem;font-weight:800;color:#38bdf8;">{reg['precision']:.1%}</div>
    </div>
    <div style="flex:1;min-width:140px;max-width:200px;background:rgba(15,23,42,0.78);border:1px solid rgba(148,163,184,0.22);border-radius:16px;padding:1rem;text-align:center;">
        <div style="font-size:0.75rem;color:#94a3b8;letter-spacing:0.05em;margin-bottom:0.4rem;">Recall</div>
        <div style="font-size:2rem;font-weight:800;color:#38bdf8;">{reg['recall']:.1%}</div>
    </div>
    <div style="flex:1;min-width:140px;max-width:200px;background:rgba(15,23,42,0.78);border:1px solid rgba(148,163,184,0.22);border-radius:16px;padding:1rem;text-align:center;">
        <div style="font-size:0.75rem;color:#94a3b8;letter-spacing:0.05em;margin-bottom:0.4rem;">Threshold</div>
        <div style="font-size:2rem;font-weight:800;color:#38bdf8;">{reg['threshold']:.2f}</div>
    </div>
</div>
"""))
    with st.expander("View model architecture", expanded=False):
        arch_fig = draw_architecture(st.session_state.selected_model_key)
        st.pyplot(arch_fig, use_container_width=True)
        plt.close(arch_fig)
    example_grid_name = reg.get("example_grid")
    with st.expander("See example predictions (correct & incorrect)", expanded=False):
        if example_grid_name:
            grid_path = FIGURES_DIR / example_grid_name
            if grid_path.exists():
                st.image(str(grid_path), use_container_width=True)
                st.caption(
                    "Sampled test-set predictions with explainability overlays — "
                    "includes both correct and incorrect calls so the failure modes are visible, not just the wins."
                )
            else:
                st.caption(f"Copy `{example_grid_name}` into `{FIGURES_DIR.name}/` to display it here.")
        else:
            st.caption("No example prediction grid was generated for this model yet.")
    st.html(textwrap.dedent(f"""
<div class="dev-note">
<span class="dev-note-label">Developer Note — {reg['label']}</span>
<div class="dev-note-text">{reg['dev_note']}</div>
</div>
"""))
@st.cache_data
def get_base64_image(image_path: Path) -> str | None:
    if not image_path.exists():
        return None
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{encoded}"
hero_image_data_uri = get_base64_image(HERO_IMAGE_PATH)
@st.cache_resource
def load_model(model_key: str):
    reg = MODEL_REGISTRY[model_key]
    if not reg["available"]:
        return None, None, None
    weights_path = ARTIFACTS_DIR / reg["weights"]
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights for {reg['label']} not found: {weights_path.name}\n"
            "Run 08_export_deployment_artifacts.py (or export_vit_head_only.py for ViT) first."
        )
    model = reg["arch"]()
    if model_key == "vit":
        head_state = torch.load(weights_path, map_location="cpu")
        model.classifier.load_state_dict(
            {k.replace("classifier.", ""): v for k, v in head_state.items()}
        )
    else:
        strict = model_key != "mobilenet"
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=strict)
    model.eval()
    config = {
        "mean": [112.99, 109.42, 100.84],
        "std":  [54.21,  53.18,  53.91],
        "best_threshold": reg["threshold"],
        "test_roc_auc": reg["roc_auc"],
        "test_accuracy": reg["accuracy"],
    }
    if reg["config"] is not None:
        config_path = ARTIFACTS_DIR / reg["config"]
        if config_path.exists():
            with open(config_path) as f:
                config.update(json.load(f))
    store = {}
    target_layer = reg["gradcam_target"](model)
    def forward_hook(module, input, output):
        store["activations"] = output
    def backward_hook(module, grad_input, grad_output):
        store["gradients"] = grad_output[0].detach()
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
    return model, config, store
def attention_rollout(model, pil_image, mean, std, input_size=224):
    image = pil_image.convert("RGB").resize((input_size, input_size))
    array = np.asarray(image, dtype="float32")
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - torch.tensor(mean)) / torch.tensor(std)
    attention_maps = []
    def patched_forward(module):
        orig = module.forward
        def new_forward(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True
            out, weights = orig(*args, **kwargs)
            if weights is not None:
                attention_maps.append(weights.detach())
            return out, weights
        return new_forward
    originals = []
    for block in model.backbone.encoder.layers:
        originals.append(block.self_attention.forward)
        block.self_attention.forward = patched_forward(block.self_attention)
    try:
        with torch.no_grad():
            _ = model(tensor)
    finally:
        for block, orig in zip(model.backbone.encoder.layers, originals):
            block.self_attention.forward = orig
    if not attention_maps:
        return None
    result = torch.eye(attention_maps[0].shape[-1])
    for attn in attention_maps:
        attn_avg = attn[0]
        attn_avg = attn_avg + torch.eye(attn_avg.shape[0])
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result = attn_avg @ result
    grid_size = int((attention_maps[0].shape[-1] - 1) ** 0.5)
    mask = result[0, 1:].reshape(grid_size, grid_size).numpy()
    if mask.max() > 0:
        mask = mask / mask.max()
    heatmap = np.asarray(
        Image.fromarray((mask * 255).astype("uint8")).resize(
            (input_size, input_size), resample=Image.BILINEAR
        ), dtype="float32"
    ) / 255.0
    return heatmap
def preprocess_image(pil_image, mean, std, input_size=IMAGE_SIZE):
    image = pil_image.convert("RGB").resize((input_size, input_size))
    array = np.asarray(image, dtype="float32")
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - torch.tensor(mean)) / torch.tensor(std)
    return tensor.unsqueeze(0)
def predict_with_gradcam(pil_image, model, store, mean, std, input_size=IMAGE_SIZE):
    store.clear()
    input_tensor = preprocess_image(pil_image, mean, std, input_size=input_size)
    input_tensor.requires_grad_(True)
    model.zero_grad()
    logit = model(input_tensor)
    probability = torch.sigmoid(logit).item()
    logit.backward()
    if "activations" not in store or "gradients" not in store:
        return probability, None
    acts = store["activations"].detach()
    grads = store["gradients"]
    if acts.dim() == 4:
        acts = acts[0]
        grads = grads[0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = cam.detach().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_image = Image.fromarray((cam * 255).astype("uint8")).resize(
            (input_size, input_size), resample=Image.BILINEAR
        )
        return probability, np.asarray(cam_image, dtype="float32") / 255.0
    else:
        return probability, None
st.set_page_config(page_title="Gravitational Lens Detector", layout="centered")
st.html(textwrap.dedent("""
<style>
html { scroll-behavior: smooth; }
.scroll-anchor { display: block; scroll-margin-top: 90px; }
:root {
    --radius-sm: 10px; --radius-md: 16px; --radius-lg: 24px;
    --space-1: 0.5rem; --space-2: 0.75rem; --space-3: 1rem;
    --space-4: 1.25rem; --space-5: 1.5rem; --space-6: 2rem;
    --shadow-sm: 0 8px 24px rgba(0,0,0,0.22);
    --shadow-lg: 0 20px 60px rgba(0,0,0,0.45);
    --transition: all 0.2s ease;
    --accent-blue: #38bdf8; --accent-indigo: #818cf8; --accent-purple: #c4b5fd;
    --text-primary: #f8fafc; --text-body: #dbeafe; --text-muted: #94a3b8;
    --border-subtle: rgba(148,163,184,0.22); --border-strong: rgba(148,163,184,0.28);
}
.starfield { position:fixed; top:0; left:0; width:100vw; height:100vh; pointer-events:none; z-index:0; overflow:hidden; }
@keyframes twinkle { 0%,100%{opacity:0.20} 50%{opacity:0.65} }
@keyframes slow-drift { from{transform:translateY(0)translateX(0)} to{transform:translateY(-100px)translateX(-35px)} }
.stars-sm { width:1px;height:1px;background:transparent;box-shadow:100px 150px #e0f2fe,350px 420px #93c5fd,700px 800px #c4b5fd,950px 200px #e0f2fe,1250px 650px #93c5fd,180px 780px #e0f2fe,480px 180px #a5f3fc,820px 520px #e0f2fe,1100px 880px #93c5fd,1380px 320px #a5f3fc,280px 920px #e0f2fe,580px 360px #c4b5fd,780px 110px #a5f3fc,1180px 450px #e0f2fe,1420px 810px #93c5fd;animation:twinkle 5s ease-in-out infinite alternate,slow-drift 90s linear infinite;opacity:0.35; }
.stars-md { width:2px;height:2px;background:transparent;border-radius:50%;box-shadow:210px 340px #38bdf8,550px 120px #818cf8,880px 680px #a5f3fc,1150px 250px #38bdf8,410px 850px #818cf8,1010px 420px #a5f3fc,1320px 900px #38bdf8;animation:twinkle 7s ease-in-out infinite alternate,slow-drift 140s linear infinite;opacity:0.28; }
.stars-lg { width:2.5px;height:2.5px;background:transparent;border-radius:50%;box-shadow:300px 530px #e0f2fe,760px 290px #c4b5fd,1050px 750px #93c5fd,500px 210px #e0f2fe;animation:twinkle 4s ease-in-out infinite alternate;opacity:0.35; }
.stApp {
    background:
        radial-gradient(circle at 20% 20%,rgba(60,90,180,0.25),transparent 30%),
        radial-gradient(circle at 80% 10%,rgba(130,80,200,0.22),transparent 25%),
        radial-gradient(circle at 50% 80%,rgba(0,180,220,0.12),transparent 30%),
        linear-gradient(135deg,#030712 0%,#07111f 45%,#020617 100%);
    color:#e5e7eb;
}
.block-container { max-width:1050px; padding-top:0; padding-bottom:var(--space-6); position:relative; z-index:1; }
h1,h2,h3 { color:#f8fafc; letter-spacing:-0.03em; }
p,li,span,div { color:var(--text-body); }
.hero-card {
    position:relative; overflow:hidden;
    width:100vw; margin-left:calc(-50vw + 50%); margin-right:calc(-50vw + 50%);
    padding:var(--space-6) 6vw calc(var(--space-6)*3.5); border-radius:0;
    background: linear-gradient(115deg,rgba(3,7,18,0.88) 0%,rgba(3,7,18,0.55) 45%,rgba(3,7,18,0.82) 100%);
    background-size:cover; background-position:center 22%;
    box-shadow:0 30px 60px rgba(0,0,0,0.5);
    min-height:100vh; display:flex; flex-direction:column; justify-content:flex-end;
}
.hero-nav-list { position:absolute; bottom:var(--space-5); right:2.2rem; text-align:right; display:flex; flex-direction:column; gap:0.35rem; z-index:2; }
.hero-nav-item { font-size:0.72rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:rgba(226,232,240,0.75); text-decoration:none; cursor:pointer; transition:var(--transition); }
.hero-nav-item:hover { color:#fff; text-shadow:0 0 12px rgba(147,197,253,0.6); }
.hero-title { font-size:clamp(2rem,4vw,3.2rem); font-weight:900; line-height:1.05; letter-spacing:-0.02em; margin-bottom:var(--space-2); color:#fff; text-shadow:0 2px 24px rgba(0,0,0,0.6); }
.hero-subtitle { font-size:1.1rem; color:rgba(226,232,240,0.88); max-width:680px; line-height:1.6; font-weight:300; letter-spacing:0.01em; }
.section-label { font-size:0.72rem; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; color:var(--accent-blue); margin-bottom:0.4rem; display:block; }
.section-heading { font-size:2rem; font-weight:800; color:var(--text-primary); letter-spacing:-0.02em; line-height:1.15; margin-bottom:var(--space-4); padding-left:var(--space-3); border-left:3px solid var(--accent-blue); }
.fact-grid { display:grid; grid-template-columns:1fr 1fr; gap:var(--space-4) var(--space-6); margin-bottom:var(--space-5); }
.fact-item { display:grid; grid-template-columns:auto 1fr; gap:var(--space-3); align-items:start; padding:var(--space-3) 0; border-top:1px solid var(--border-subtle); }
.fact-number { font-size:2rem; font-weight:900; color:var(--accent-blue); letter-spacing:-0.04em; line-height:1; min-width:2.5rem; }
.fact-content-title { font-size:0.9rem; font-weight:700; color:var(--text-primary); margin-bottom:0.3rem; letter-spacing:0.01em; }
.fact-content-body { font-size:0.82rem; color:var(--text-body); line-height:1.55; }
.glass-card { padding:var(--space-4) var(--space-5); border-radius:var(--radius-md); background:rgba(15,23,42,0.78); border:1px solid var(--border-subtle); box-shadow:var(--shadow-sm); margin-bottom:var(--space-3); transition:var(--transition); }
.metrics-table-wrap { overflow-x:auto; border-radius:var(--radius-md); border:1px solid var(--border-subtle); background:rgba(15,23,42,0.5); }
.metrics-table-wrap table { margin:0; }
.glass-card:hover { border-color:rgba(147,197,253,0.35); }
.pipeline-row { display:flex; align-items:center; justify-content:center; gap:0.6rem; flex-wrap:wrap; margin-bottom:var(--space-3); }
.pipeline-node { display:flex; flex-direction:column; align-items:center; gap:0.6rem; flex:0 0 auto; }
.pipeline-step { flex:0 0 auto; width:120px; height:120px; border-radius:50%; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:0.75rem; background:rgba(14,26,46,0.85); border:1.5px solid var(--accent-blue); text-align:center; font-size:0.75rem; font-weight:600; color:#e0f2fe; line-height:1.3; transition:var(--transition); box-shadow:0 0 18px rgba(56,189,248,0.12); }
.pipeline-step:hover { background:rgba(56,189,248,0.12); box-shadow:0 0 28px rgba(56,189,248,0.28); border-color:#93c5fd; }
.pipeline-arrow { align-self:flex-start; flex:0 0 auto; font-size:1.2rem; color:var(--accent-blue); opacity:0.7; margin-top:42px; }
.pipeline-node-desc { font-size:0.7rem; color:var(--text-muted); text-align:center; max-width:120px; line-height:1.3; letter-spacing:0.01em; }
.prediction-lens { padding:var(--space-5); border-radius:var(--radius-md); background:linear-gradient(135deg,rgba(88,28,135,0.8),rgba(30,64,175,0.78)); border:1px solid rgba(196,181,253,0.45); box-shadow:var(--shadow-lg); margin-bottom:var(--space-3); }
.prediction-nonlens { padding:var(--space-5); border-radius:var(--radius-md); background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(30,64,175,0.45)); border:1px solid rgba(96,165,250,0.35); box-shadow:var(--shadow-lg); margin-bottom:var(--space-3); }
.prediction-title { font-size:1.8rem; font-weight:800; color:var(--text-primary); margin-bottom:var(--space-1); }
.prediction-subtitle { font-size:0.95rem; color:var(--text-body); }
.section-nav-bar { display:flex; gap:4px; flex-wrap:wrap; justify-content:center; position:fixed; top:52px; left:0; right:0; width:100%; z-index:999; background:transparent; padding:0.6rem 0; opacity:0; visibility:hidden; transform:translateY(-12px); transition:opacity 0.25s ease,transform 0.25s ease,visibility 0.25s; }
.section-nav-bar.visible { opacity:1; visibility:visible; transform:translateY(0); }
.section-nav-spacer { height:64px; }
.section-nav-item { display:inline-block; background:rgba(15,23,42,0.6); border-radius:var(--radius-sm) var(--radius-sm) 0 0; padding:var(--space-2) var(--space-3); font-size:0.92rem; font-weight:600; color:var(--text-body); text-decoration:none; border-bottom:2px solid transparent; transition:var(--transition); }
.section-nav-item:hover { color:#fff; background:rgba(30,64,175,0.35); }
.section-nav-item.active { color:var(--text-primary); border-bottom:2px solid var(--accent-blue); background:rgba(30,64,175,0.25); }
.stButton button { width:100%; border-radius:var(--radius-sm); border:1px solid rgba(147,197,253,0.35); background:rgba(15,23,42,0.82); color:var(--text-body); transition:var(--transition); }
.stButton button:hover { border-color:#93c5fd; color:#fff; background:rgba(30,64,175,0.85); transform:translateY(-1px); box-shadow:var(--shadow-sm); }
[data-testid="stFileUploader"] { background:rgba(15,23,42,0.68); border:1px dashed rgba(147,197,253,0.45); border-radius:var(--radius-md); padding:var(--space-3); transition:var(--transition); }
[data-testid="stFileUploader"]:hover { border-color:var(--accent-blue); }
[data-testid="stExpander"] { background:rgba(15,23,42,0.72); border:1px solid var(--border-subtle); border-radius:var(--radius-md); }
[data-testid="stMetric"] { background:rgba(15,23,42,0.7); border:1px solid var(--border-subtle); border-radius:var(--radius-md); padding:var(--space-2) var(--space-3); transition:var(--transition); }
[data-testid="stMetric"]:hover { border-color:var(--border-subtle); }
.stProgress > div > div > div > div { background-image:linear-gradient(90deg,#38bdf8,#818cf8,#c084fc); }
.warning-card { padding:var(--space-3) var(--space-4); border-radius:var(--radius-md); background:rgba(120,53,15,0.30); border:1px solid rgba(251,191,36,0.35); color:#fde68a; margin-top:var(--space-3); }
.footer-card { padding:var(--space-3) var(--space-5); border-radius:var(--radius-md); background:rgba(15,23,42,0.6); border:1px solid rgba(148,163,184,0.18); color:var(--text-muted); font-size:0.85rem; text-align:center; margin-top:var(--space-5); }
hr { border-color:rgba(148,163,184,0.22); }
img { border-radius:14px; }
footer { visibility:hidden; }
.dev-note {
    padding:var(--space-4) var(--space-5);
    border-radius:var(--radius-md);
    background:rgba(56,189,248,0.06);
    border:1px solid rgba(56,189,248,0.2);
    border-left:3px solid var(--accent-blue);
    margin:var(--space-4) 0;
}
.dev-note-label {
    font-size:0.68rem; font-weight:700; letter-spacing:0.18em;
    text-transform:uppercase; color:var(--accent-blue);
    margin-bottom:0.5rem; display:block;
}
.dev-note-text { font-size:0.88rem; color:var(--text-body); line-height:1.65; font-style:italic; }
</style>
<div class="starfield"><div class="stars-sm"></div><div class="stars-md"></div><div class="stars-lg"></div></div>
"""))
_hero_bg = hero_image_data_uri
if _hero_bg:
    st.html(textwrap.dedent(f"""
<style>
.hero-card {{
    background:
        linear-gradient(115deg,rgba(3,7,18,0.88) 0%,rgba(3,7,18,0.55) 45%,rgba(3,7,18,0.82) 100%),
        url('{_hero_bg}') !important;
    background-size:cover !important;
    background-position:center 22% !important;
}}
</style>
"""))
else:
    # No live external fallback here on purpose — the app should never depend
    # on a network fetch at demo time. If the local file is missing, the
    # hero card just falls back to its plain CSS gradient (no photo).
    st.warning(
        f"Hero background not found at `{HERO_IMAGE_PATH}`. "
        "The page will still work, just without the background photo."
    )
st.html(textwrap.dedent("""
<div class="hero-card">
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
</div>
"""))
if "selected_model_key" not in st.session_state:
    st.session_state.selected_model_key = "cnn"
model = None
config = None
store = None
load_error = None
try:
    model, config, store = load_model(st.session_state.selected_model_key)
except FileNotFoundError as e:
    load_error = str(e)
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
SECTIONS = [
    ("overview",          "Overview"),
    ("how-it-works",      "How It Works"),
    ("model-comparison",  "Model Comparison"),
    ("try-the-model",     "Try the Model"),
    ("interpretation",    "Interpretation & Notes"),
    ("references",        "References"),
]
_nav_items_html = "".join(
    f'<a href="#{key}" class="section-nav-item">{label}</a>'
    for key, label in SECTIONS
)
st.html(f'<div class="section-nav-bar" id="section-nav-bar">{_nav_items_html}</div>')
st.html('<div class="section-nav-spacer"></div>')
components.html("""
    <script>
    (function () {
        function getBar() { return window.parent.document.getElementById("section-nav-bar"); }
        function currentScrollTop(target) {
            if (target && typeof target.scrollTop === "number" && target !== window.parent.document) return target.scrollTop;
            return window.parent.scrollY || window.parent.document.documentElement.scrollTop || window.parent.document.body.scrollTop || 0;
        }
        function toggleNavBar(target) {
            var bar = getBar(); if (!bar) return;
            bar.classList.toggle("visible", currentScrollTop(target) > window.parent.innerHeight * 0.6);
        }
        var SECTION_IDS = ["overview","how-it-works","model-comparison","try-the-model","interpretation","references"];
        function updateActiveSection() {
            var activeId = SECTION_IDS[0], offset = 110;
            for (var i = 0; i < SECTION_IDS.length; i++) {
                var el = window.parent.document.getElementById(SECTION_IDS[i]);
                if (!el) continue;
                var threshold = (i === SECTION_IDS.length - 1) ? window.parent.innerHeight * 0.5 : offset;
                if (el.getBoundingClientRect().top - threshold <= 0) activeId = SECTION_IDS[i];
            }
            window.parent.document.querySelectorAll(".section-nav-item").forEach(function(item) {
                item.classList.toggle("active", item.getAttribute("href") === "#" + activeId);
            });
        }
        function onScroll(target) { toggleNavBar(target); updateActiveSection(); }
        window.parent.document.addEventListener("scroll", function(e) { onScroll(e.target); }, true);
        window.parent.addEventListener("scroll", function() { onScroll(window.parent); }, { passive: true });
        onScroll(window.parent);
        window.parent.document.addEventListener("click", function(e) {
            var link = e.target.closest(".hero-nav-item, .section-nav-item");
            if (!link) return;
            var href = link.getAttribute("href") || "";
            if (href.charAt(0) !== "#") return;
            var target = window.parent.document.getElementById(href.slice(1));
            if (!target) return;
            e.preventDefault();
            target.scrollIntoView({ behavior: "smooth", block: "start" });
        });
    })();
    </script>
    """, height=0)
# Overview
st.html('<div id="overview" class="scroll-anchor"></div>')
if True:
    st.html(textwrap.dedent("""
<span class="section-label">Introduction</span>
<div class="section-heading">Project Overview</div>
"""))
    st.html(textwrap.dedent("""
<div class="glass-card">
<b>Gravitational lensing</b> is a direct consequence of Einstein's general
relativity: mass curves spacetime, and light travelling through that
curved spacetime bends off its straight-line path. When a massive
<b>foreground galaxy</b> (the "lens") sits almost exactly between us and a
much more distant <b>background galaxy</b> (the "source"), its gravity
bends the source's light around it from multiple directions at once.
<br><br>
Because the light rays bend rather than travel straight, the background
galaxy doesn't just look dimmer or displaced — its image gets smeared into
curved <b>arcs</b>, or in the most perfectly aligned cases, a complete
<b>Einstein ring</b> encircling the lens galaxy. This is <b>strong
lensing</b>: the regime where the distortion is dramatic enough to see
directly in an image, as opposed to <i>weak lensing</i>, where the effect
is a subtle statistical stretching averaged over many galaxies.
</div>
"""))
    lens_diagram_path = FIGURES_DIR / "gravitational_lensing_diagram.png"
    if lens_diagram_path.exists():
        st.image(str(lens_diagram_path), use_container_width=True)
        st.caption(
            "Light from a distant background galaxy bends around a closer "
            "foreground lens galaxy on its way to the observer, producing "
            "the arcs and Einstein rings this project detects."
        )
    else:
        st.warning(
            f"Diagram not found at: `{lens_diagram_path}`\n\n"
            "Check the filename matches exactly (including capitalisation) "
            "and that the app was restarted after adding the file."
        )
    st.html(textwrap.dedent("""
<div class="glass-card">
Strong lenses matter scientifically because the amount and shape of the
distortion depends on how much mass — including invisible <b>dark
matter</b> — sits in the foreground galaxy. They act as natural
gravitational telescopes, magnifying faint, extremely distant background
galaxies that would otherwise be too faint to study. But genuine strong
lenses are also <b>rare</b>, and modern sky surveys capture millions to
billions of objects — far too many for manual expert inspection to scale.
This app demonstrates a <b>CNN-based classification prototype</b> that
flags likely lens candidates automatically, paired with <b>Grad-CAM
explainability</b> so predictions can be visually inspected rather than
trusted blindly.
</div>
"""))
    st.html(textwrap.dedent("""
<span class="section-label">Context</span>
<div class="section-heading">Why This Matters</div>
<div class="fact-grid">
<div class="fact-item"><div class="fact-number">01</div><div>
<div class="fact-content-title">Rare-Object Detection</div>
<div class="fact-content-body">Genuine strong lenses are a tiny fraction of all survey objects. To learn discriminative features, we trained on a <b>class-balanced</b> curated set — but the model is deployed against surveys where positives are vanishingly rare, which is why precision matters as much as recall in practice.</div>
</div></div>
<div class="fact-item"><div class="fact-number">02</div><div>
<div class="fact-content-title">False Positives</div>
<div class="fact-content-body">Spiral galaxies, ring galaxies, mergers, bright stars, and imaging artefacts can all visually resemble lensing features — precision matters as much as recall.</div>
</div></div>
<div class="fact-item"><div class="fact-number">03</div><div>
<div class="fact-content-title">Survey Scale</div>
<div class="fact-content-body">Modern astronomical surveys capture millions to billions of objects — far too many for manual expert inspection to scale to this volume.</div>
</div></div>
<div class="fact-item"><div class="fact-number">04</div><div>
<div class="fact-content-title">Explainable AI</div>
<div class="fact-content-body">Grad-CAM visualises which image regions drove each prediction, supporting human scrutiny of the model rather than opaque black-box trust.</div>
</div></div>
</div>
"""))
    with st.expander("See sample training images (lens vs. non-lens)", expanded=False):
        show_result_figure(
            "dataset_sample_grid",
            caption="Confident lens systems (top) vs. presumed non-lens galaxies (bottom) from the training set — "
                    "same galaxies matched pairwise where possible, to keep brightness and morphology comparable.",
        )
    st.html(textwrap.dedent("""
<span class="section-label">Data</span>
<div class="section-heading">Dataset at a Glance</div>
<div class="fact-grid">
<div class="fact-item"><div class="fact-number">01</div><div>
<div class="fact-content-title">4,076 labelled images</div>
<div class="fact-content-body">2,038 confident lens systems (catalogue-listed) paired with 2,038 presumed non-lens galaxies, drawn from real Legacy Survey / Galaxy10 imagery — not simulations.</div>
</div></div>
<div class="fact-item"><div class="fact-number">02</div><div>
<div class="fact-content-title">2,744 / 588 / 588 split</div>
<div class="fact-content-body">Train / validation / test split, held fixed across all four architectures so their metrics are directly comparable.</div>
</div></div>
<div class="fact-item"><div class="fact-number">03</div><div>
<div class="fact-content-title">Balanced by design</div>
<div class="fact-content-body">The curated dataset is exactly 50/50 lens vs. non-lens. This is a deliberate training choice, not a claim that real surveys are balanced — see Rare-Object Detection above.</div>
</div></div>
<div class="fact-item"><div class="fact-number">04</div><div>
<div class="fact-content-title">"Presumed" non-lens, not proven</div>
<div class="fact-content-body">Negative examples are galaxies without a catalogued lens detection, not objects verified impossible to lens — a limitation carried through to model evaluation.</div>
</div></div>
</div>
"""))
# How the Pipeline Works
st.html('<div id="how-it-works" class="scroll-anchor"></div>')
if True:
    st.html(textwrap.dedent("""
<span class="section-label">Methodology</span>
<div class="section-heading">How the Pipeline Works</div>
"""))
    st.caption("The same flow runs every time you submit an image.")
    steps = [
        ("Image Input",      "Uploaded JPG/PNG cutout"),
        ("Preprocessing",    "Resize to 128×128, normalise"),
        ("CNN Prediction",   "Forward pass, raw probability"),
        ("Confidence Score", "Threshold-tuned decision"),
        ("Grad-CAM",         "Gradient heatmap of influence"),
        ("Interpretation",   "Candidate flag for review"),
    ]
    _parts = []
    for i, (label, desc) in enumerate(steps):
        _parts.append(
            f'<div class="pipeline-node">'
            f'<div class="pipeline-step">{label}</div>'
            f'<div class="pipeline-node-desc">{desc}</div>'
            f'</div>'
        )
        if i < len(steps) - 1:
            _parts.append('<div class="pipeline-arrow">→</div>')
    st.html(f'<div class="pipeline-row">{"".join(_parts)}</div>')
# Model Comparison
st.html('<div id="model-comparison" class="scroll-anchor"></div>')
if True:
    st.html(textwrap.dedent("""
<span class="section-label">Evaluation</span>
<div class="section-heading">Model Performance Comparison</div>
"""))
    st.caption("Quantitative comparison of test metrics across all evaluated models.")
    st.html(textwrap.dedent("""
<div class="dev-note">
<span class="dev-note-label">Developer Note — Which model would we use?</span>
<div class="dev-note-text">
Honestly? It depends on what you care about. <b>ViT-B/16 is our best performer</b> on every
metric. But if you care about <em>understanding why</em> a prediction was made, <b>CBAM is our
personal favourite</b> — attention is baked into the architecture, not bolted on afterwards.
Custom CNN is the scrappiest and still remarkably competitive. MobileNetV3 showed us how far
pretrained ImageNet features stretch into domains they were never designed for.
</div>
</div>
"""))
    st.html(textwrap.dedent("""
<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:var(--accent-blue);margin-bottom:0.4rem;display:block;">Active Model</div>
"""))
    selector_cols = st.columns(4)
    for i, (key, reg) in enumerate(MODEL_REGISTRY.items()):
        with selector_cols[i]:
            is_active = st.session_state.selected_model_key == key
            label = f"{'✓ ' if is_active else ''}{reg['label']}"
            if st.button(label, key=f"model_btn_{key}",
                         type="primary" if is_active else "secondary",
                         use_container_width=True):
                st.session_state.selected_model_key = key
                st.rerun()
    render_active_model_snapshot(MODEL_REGISTRY[st.session_state.selected_model_key])
    st.divider()
    # Built directly from MODEL_REGISTRY (the single source of truth) instead
    # of a second hardcoded copy, so the chart/table can never drift out of
    # sync with the metric cards or test_metrics.json.
    _display_names = {
        "cnn": "Custom CNN (Real Data)",
        "cbam": "CBAM-Attention CNN",
        "mobilenet": "MobileNetV3 (Transfer Learning)",
        "vit": "ViT-B/16 (Transfer Learning)",
    }
    def _f1(reg):
        p, r = reg["precision"], reg["recall"]
        return 2 * p * r / (p + r) if (p + r) else 0.0
    model_metrics = {
        "Model Architecture": [_display_names[k] for k in MODEL_REGISTRY],
        "ROC-AUC":  [MODEL_REGISTRY[k]["roc_auc"] for k in MODEL_REGISTRY],
        "Accuracy": [MODEL_REGISTRY[k]["accuracy"] for k in MODEL_REGISTRY],
        "Precision":[MODEL_REGISTRY[k]["precision"] for k in MODEL_REGISTRY],
        "Recall":   [MODEL_REGISTRY[k]["recall"] for k in MODEL_REGISTRY],
        "F1-Score": [round(_f1(MODEL_REGISTRY[k]), 3) for k in MODEL_REGISTRY],
    }
    df_metrics = pd.DataFrame(model_metrics)
    metric_choice = st.selectbox(
        "Select metric to visualize:",
        ["ROC-AUC", "Accuracy", "F1-Score", "Precision", "Recall"],
        index=0,
    )
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    y_pos = np.arange(len(df_metrics))
    values = df_metrics[metric_choice]
    best_idx = list(values).index(max(values))
    bar_colors = ["#38bdf8" if i == best_idx else "#0e3a52" for i in range(len(values))]
    ax.barh(y_pos, values, color=bar_colors, height=0.55, edgecolor="none")
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
    for bar, val in zip(ax.patches, values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="left", va="center", color="#f8fafc", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.divider()
    st.html(textwrap.dedent("""
<div style="font-size:0.95rem;font-weight:700;color:var(--text-primary);letter-spacing:0.01em;margin:var(--space-4) 0 var(--space-2);">Detailed Test Metrics</div>
"""))
    metric_cols = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]
    def highlight_best(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in metric_cols:
            if col in df.columns:
                styles.loc[df[col].idxmax(), col] = (
                    "background-color:rgba(56,189,248,0.18);color:#38bdf8;font-weight:700;"
                )
        return styles
    styled = (
        df_metrics.style
        .apply(highlight_best, axis=None)
        .format({col: "{:.3f}" for col in metric_cols})
        .set_properties(**{
            "background-color": "rgba(15,23,42,0.0)",
            "color": "#dbeafe",
            "border-color": "rgba(148,163,184,0.15)",
            "font-size": "0.88rem",
            "text-align": "center",
        })
        .set_properties(subset=["Model Architecture"], **{"text-align": "left"})
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "rgba(15,23,42,0.9)"),
                ("color", "#94a3b8"),
                ("font-size", "0.75rem"),
                ("letter-spacing", "0.08em"),
                ("text-transform", "uppercase"),
                ("border-bottom", "1px solid rgba(148,163,184,0.25)"),
                ("padding", "0.6rem 0.75rem"),
                ("text-align", "center"),
            ]},
            {"selector": "tbody tr:hover td", "props": [("background-color", "rgba(56,189,248,0.06)")]},
            {"selector": "td", "props": [
                ("padding", "0.55rem 0.75rem"),
                ("border-bottom", "1px solid rgba(148,163,184,0.1)"),
                ("text-align", "center"),
            ]},
            {"selector": "td:first-child", "props": [("text-align", "left")]},
            {"selector": "thead th:first-child", "props": [("text-align", "left")]},
        ])
        .set_table_attributes('style="width:100%;border-collapse:collapse;"')
        .hide(axis="index")
    )
    # st.dataframe() renders its own interactive grid and ignores a Styler's
    # text-align CSS (numbers default to right-aligned regardless of what the
    # Styler sets) — rendering the Styler's own HTML respects it exactly.
    st.markdown(f'<div class="metrics-table-wrap">{styled.to_html()}</div>', unsafe_allow_html=True)
# Try the Model
st.html('<div id="try-the-model" class="scroll-anchor"></div>')
if True:
    active_label = MODEL_REGISTRY[st.session_state.selected_model_key]["label"]
    st.html(textwrap.dedent(f"""
<span class="section-label">Demo</span>
<div class="section-heading">Try the Model</div>
<div style="font-size:0.85rem;color:var(--text-muted);margin-top:-1rem;margin-bottom:1rem;">
Active model: <b style="color:var(--accent-blue);">{active_label}</b> —
switch models in the Evaluation section above.
</div>
"""))
    if "selected_image_bytes" not in st.session_state:
        st.session_state.selected_image_bytes = None
        st.session_state.selected_image_name = None
    sample_files = sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
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
        st.caption("No sample images found in `sample_images/` — upload your own below.")
    st.html(textwrap.dedent("""
<div style="font-size:0.95rem;font-weight:700;color:var(--text-primary);letter-spacing:0.01em;margin:var(--space-4) 0 var(--space-2);">Or upload your own</div>
"""))
    st.caption(
        "Upload a square galaxy-scale cutout in JPG or PNG format. "
        "Best results come from survey-style cutouts similar to the Legacy Survey training data."
    )
    uploaded_file = st.file_uploader(
        "Upload an astronomical image cutout (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is not None:
        st.session_state.selected_image_bytes = uploaded_file.read()
        st.session_state.selected_image_name = uploaded_file.name
    if st.session_state.selected_image_bytes is not None:
        try:
            pil_image = Image.open(io.BytesIO(st.session_state.selected_image_bytes))
        except Exception as error:
            st.error(f"Could not read this image file: `{error}`")
            st.stop()
        active_input_size = MODEL_REGISTRY[st.session_state.selected_model_key].get("input_size", IMAGE_SIZE)
        with st.spinner("Running model..."):
            probability, cam = predict_with_gradcam(pil_image, model, store, mean, std, input_size=active_input_size)
        predicted_label = "Lens candidate" if probability >= best_threshold else "Non-lens"
        st.divider()
        st.caption(f"Analysing: **{st.session_state.selected_image_name}**")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image")
            st.image(pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), use_container_width=True)
        with col2:
            resize_to = MODEL_REGISTRY[st.session_state.selected_model_key].get("input_size", IMAGE_SIZE)
            display_image = np.asarray(pil_image.convert("RGB").resize((resize_to, resize_to)))
            if st.session_state.selected_model_key == "vit":
                st.subheader("Attention Rollout")
                with st.spinner("Computing attention rollout..."):
                    rollout = attention_rollout(model, pil_image, mean, std, input_size=resize_to)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(display_image)
                if rollout is not None:
                    ax.imshow(rollout, cmap="magma", alpha=0.48)
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "Attention Rollout (Abnar & Zuidema, 2020) — multiplies attention "
                    "weights across all 12 ViT encoder layers to show which image patches "
                    "most influenced the classification output."
                )
            else:
                st.subheader("Grad-CAM heatmap")
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(display_image)
                if cam is not None:
                    ax.imshow(cam, cmap="magma", alpha=0.48)
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Bright regions indicate stronger influence on the prediction.")
        st.divider()
        if predicted_label == "Lens candidate":
            st.html(textwrap.dedent(f"""
<div class="prediction-lens">
<div class="prediction-title">Prediction: {predicted_label}</div>
<div class="prediction-subtitle">The model output is above the validation-tuned threshold. Treat this as a candidate flag, not a scientific confirmation.</div>
</div>
"""))
        else:
            st.html(textwrap.dedent(f"""
<div class="prediction-nonlens">
<div class="prediction-title">Prediction: {predicted_label}</div>
<div class="prediction-subtitle">The model output is below the validation-tuned lens threshold. This does not prove the object is not a lens.</div>
</div>
"""))
        conf_col1, conf_col2 = st.columns([2, 1])
        with conf_col1:
            st.write("**Lens confidence**")
            st.progress(min(max(probability, 0.0), 1.0), text=f"{probability:.1%}")
            st.caption(
                f"Decision threshold: {best_threshold:.2f} — tuned on validation-set F1 rather than "
                "the default 0.5. Each model has its own threshold, so the same raw score can cross the "
                "line differently across models — this one is tuned for the real-world imbalance, "
                "not the class-balanced training set."
            )
        with conf_col2:
            st.metric("Lens probability", f"{probability:.1%}")
    else:
        st.info("Click a sample above, or upload your own image, to get a prediction.")
# Interpretation & Notes
st.html('<div id="interpretation" class="scroll-anchor"></div>')
if True:
    st.html(textwrap.dedent("""
<span class="section-label">Guidance</span>
<div class="section-heading">How to Interpret the Result</div>
"""))
    st.html(textwrap.dedent("""
<div class="fact-grid">
<div class="fact-item"><div class="fact-number">01</div><div>
<div class="fact-content-title">High confidence does not equal a true lens</div>
<div class="fact-content-body">The score reflects the model's learned pattern-matching, not physical confirmation of gravitational lensing. Treat every result as a candidate flag, not a scientific conclusion.</div>
</div></div>
<div class="fact-item"><div class="fact-number">02</div><div>
<div class="fact-content-title">Grad-CAM shows influence, not proof</div>
<div class="fact-content-body">It highlights image regions that most affected the prediction — it does not verify the model has learned the actual physics of lensing.</div>
</div></div>
<div class="fact-item"><div class="fact-number">03</div><div>
<div class="fact-content-title">Arc & ring structure is the signal</div>
<div class="fact-content-body">A credible lens prediction should concentrate on curved, elongated features. If the heatmap highlights a point source or edge artefact instead, treat the result with extra caution.</div>
</div></div>
<div class="fact-item"><div class="fact-number">04</div><div>
<div class="fact-content-title">False positives are expected</div>
<div class="fact-content-body">Spiral arms, ring galaxies, mergers, bright stars, and imaging artefacts can all visually resemble lensing features. Expert review is always required before drawing astrophysical conclusions.</div>
</div></div>
</div>
"""))
    with st.expander("See real false positive / false negative examples", expanded=False):
        show_result_figure(
            "fp_fn_gallery",
            caption="Real false positives and false negatives from the Custom CNN's test set, with the model's predicted probability. "
                    "Several false positives are ordinary bright ellipticals — exactly the confound described below.",
        )
    st.divider()
    st.html(textwrap.dedent("""
<span class="section-label">Limitations</span>
<div class="section-heading">Model and Dataset Notes</div>
"""))
    st.html(textwrap.dedent("""
<div class="glass-card">
<ul>
<li>This is a <b>student-scale prototype</b>, built for a final-year academic project — not a production astronomical pipeline.</li>
<li>The model is <b>not a validated astronomical discovery tool</b> and should not be used to draw real scientific conclusions.</li>
<li><b>Dataset quality and provenance</b> directly affect performance: of the 4,076 labelled images (2,744/588/588 train/val/test), positive examples come from catalogue-listed lens systems of varying confidence, and negative examples are <i>presumed</i> non-lens galaxies, not objects proven impossible to lens.</li>
<li>Results should be treated as <b>candidate screening</b> to prioritise objects for expert review — not as final scientific confirmation.</li>
</ul>
</div>
"""))
    st.divider()
    st.html(textwrap.dedent("""
<div class="fact-grid">
<div class="fact-item"><div class="fact-number">01</div><div>
<div class="fact-content-title">Student-scale dataset</div>
<div class="fact-content-body">Trained on catalogue-listed lens candidates and presumed non-lens galaxies — not a fully expert-verified benchmark. Some lensing features may be faint or not clearly visible in the Legacy Survey JPEG cutouts.</div>
</div></div>
<div class="fact-item"><div class="fact-number">02</div><div>
<div class="fact-content-title">Brightness confound (r = -0.81)</div>
<div class="fact-content-body">Model confidence correlates with image brightness/compactness — likely a genuine structural difference between classes rather than a simple normalisation artefact.</div>
</div></div>
<div class="fact-item"><div class="fact-number">03</div><div>
<div class="fact-content-title">All four architectures converge</div>
<div class="fact-content-body">Custom CNN, CBAM-Attention CNN, MobileNetV3, and ViT-B/16 all land in the 0.98–0.99 ROC-AUC range, suggesting a practical ceiling for this dataset rather than one model outperforming by chance.</div>
</div></div>
<div class="fact-item"><div class="fact-number">04</div><div>
<div class="fact-content-title">ViT achieves highest precision</div>
<div class="fact-content-body">ViT-B/16 reaches ROC-AUC 0.992 with only 4 false positives out of 306 non-lens test images. Global self-attention may capture ring and arc morphology more effectively than local convolutional filters.</div>
</div></div>
</div>
"""))
    with st.expander("See the brightness confound evidence, before and after mitigation", expanded=False):
        bright_col1, bright_col2 = st.columns(2)
        with bright_col1:
            show_result_figure(
                "brightness_confound",
                caption="Predicted probability vs. centre/edge brightness ratio (r = -0.81) — before any mitigation attempt.",
            )
        with bright_col2:
            show_result_figure(
                "brightness_confound_mitigated",
                caption="Same test images after per-image normalisation (r = -0.81, unchanged) — the confound survives the fix, "
                        "suggesting it reflects a genuine structural difference between classes rather than a lighting artefact.",
            )
# References
st.html('<div id="references" class="scroll-anchor"></div>')
if True:
    st.html(textwrap.dedent("""
<span class="section-label">Literature</span>
<div class="section-heading">References & Research Context</div>
"""))
    st.html(textwrap.dedent("""
<div class="glass-card">
<ul>
<li><b>CNN-based lens detection:</b> Lanusse et al. (2017), CMU DeepLens — early deep learning methods for automatic galaxy-galaxy strong lens finding.</li>
<li><b>Strong Gravitational Lens Finding Challenge:</b> Metcalf et al. (2019) and Bom et al. (2022) — machine learning challenges framing lens detection as rare-object classification and highlighting the importance of controlling false positives.</li>
<li><b>Sim-to-real gap:</b> Pearce-Casey et al. (2024) — Euclid strong lens searches, showing that models trained on simulated/controlled data can be harder to apply to real survey images.</li>
<li><b>Grad-CAM:</b> Selvaraju et al. (2017) — gradient-based visual explanations used throughout this app's explainability features.</li>
<li><b>Attention Rollout:</b> Abnar & Zuidema (2020) — used here for ViT explainability, propagating attention weights across all encoder layers.</li>
<li><b>Simulated data tooling:</b> lenstronomy / deeplenstronomy (Birrer &amp; Amara, 2018) — used for this project's simulated baseline model.</li>
</ul>
</div>
"""))
# Footer
st.html(textwrap.dedent("""
<div class="footer-card">
    Built by Ahana Bhattacharji &amp; Vaishnav Malvankar ·
    Explainable Deep Learning for Gravitational Lens Detection in Astronomical Survey Images ·
    Student research prototype, not a validated scientific tool.
</div>
"""))