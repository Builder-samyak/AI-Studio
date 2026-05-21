"""
app/streamlit_app.py
Destroyers | 42174 AI Studio Autumn 2026

Streamlit web UI for breast cancer detection.
Loads trained EfficientNet-B0 checkpoint and runs inference.
Displays: prediction, probability, Grad-CAM heatmap, NLP explanation.

Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/sagemaker-user/user-default-efs/destroyers_model/efficientnet_b0_best.pth"
)
MODEL_VERSION    = "EfficientNet-B0 v0.2"
FULL_DATASET_ACC = 91.97
FULL_DATASET_AUC = 0.9699
DEFAULT_THRESHOLD = 0.5

CLASS_NAMES   = {0: "Non-Cancer", 1: "Cancer"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Detection — Destroyers",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI-Based Breast Cancer Detection")
st.caption(
    f"Model: {MODEL_VERSION}  ·  Accuracy: {FULL_DATASET_ACC}%  ·  "
    f"AUC: {FULL_DATASET_AUC}  ·  Team Destroyers — 42174 AI Studio"
)
st.markdown("---")


@st.cache_resource
def load_model(model_path: str):
    """Load EfficientNet-B0 checkpoint. Cached so it loads only once."""
    import torch
    from src.models.train import build_efficientnet_b0

    if not os.path.exists(model_path):
        return None, f"Model not found at: {model_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_efficientnet_b0(num_classes=2)
    ckpt   = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, None


def preprocess_image(image: Image.Image):
    """Apply EfficientNet preprocessing transforms."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def run_inference(model, image_tensor, threshold: float):
    """Run model inference and return prediction, probability."""
    import torch
    device = next(model.parameters()).device
    with torch.no_grad():
        out  = model(image_tensor.to(device))
        prob = torch.softmax(out, dim=1)[0, 1].item()
    pred_class = "Cancer" if prob >= threshold else "Non-Cancer"
    return pred_class, prob


def generate_gradcam(model, image_tensor) -> np.ndarray | None:
    """Generate Grad-CAM heatmap overlay. Returns overlay as numpy array or None."""
    try:
        import torch, cv2

        device = next(model.parameters()).device
        grads, acts = [], []

        def hook_fn(module, inp, out):
            acts.append(out.detach())
            out.register_hook(lambda g: grads.append(g.detach()))

        hook = model.features[-1].register_forward_hook(hook_fn)
        img  = image_tensor.to(device)
        out  = model(img)
        pred = out.argmax(1).item()
        model.zero_grad()
        out[0, pred].backward()
        hook.remove()

        if not grads or not acts:
            return None

        g = grads[0].squeeze().cpu().numpy()
        a = acts[0].squeeze().cpu().numpy()
        w = g.mean(axis=(1, 2))
        cam = np.zeros(a.shape[1:], dtype=np.float32)
        for wi, ai in zip(w, a):
            cam += wi * ai
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (128, 128))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        mean = np.array(IMAGENET_MEAN)
        std  = np.array(IMAGENET_STD)
        img_np = image_tensor.squeeze().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np * std + mean, 0, 1)
        img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    except Exception:
        return None


# ── Load model ────────────────────────────────────────────────────────────────
model, model_error = load_model(MODEL_PATH)

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload image")
    uploaded = st.file_uploader(
        "Upload a breast histopathology image (.png or .jpg)",
        type=["png", "jpg", "jpeg"],
        help="50×50 or larger histopathology patch image"
    )
    threshold = st.slider(
        "Malignant probability threshold",
        min_value=0.30, max_value=0.90,
        value=DEFAULT_THRESHOLD, step=0.05,
        help="Predictions above this value are classified as Cancer"
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image (original size)", use_column_width=True)

with col2:
    st.subheader("Prediction result")

    if model_error:
        st.error(f"Model load error: {model_error}")
        st.info("Set the MODEL_PATH environment variable to point to your checkpoint.")

    elif uploaded and st.button("▶ Run classification", type="primary"):
        with st.spinner("Running EfficientNet-B0 inference..."):
            image_tensor = preprocess_image(image)
            pred_class, prob = run_inference(model, image_tensor, threshold)

        # Result display
        if pred_class == "Cancer":
            st.error(f"### ⚠ Prediction: {pred_class}")
        else:
            st.success(f"### ✓ Prediction: {pred_class}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Malignant probability", f"{prob*100:.1f}%")
        c2.metric("Threshold", f"{threshold*100:.0f}%")

        from src.nlp.explainer import get_confidence_label, get_flag_status
        c3.metric("Confidence", get_confidence_label(prob, threshold))

        if get_flag_status(pred_class, prob, threshold):
            st.warning("🚩 Case flagged for clinical review.")

        # Grad-CAM
        st.markdown("**Grad-CAM heatmap** — highlighted regions contributed most to the prediction")
        overlay = generate_gradcam(model, image_tensor)
        if overlay is not None:
            st.image(overlay, caption="Grad-CAM heatmap overlay", use_column_width=True)
        else:
            st.info("Grad-CAM not available for this image.")

        # NLP explanation
        st.markdown("---")
        st.subheader("Model explanation")
        from src.nlp.explainer import generate_explanation
        explanation = generate_explanation(pred_class, prob, threshold)
        st.markdown(explanation)

        # Disclaimer
        st.markdown("---")
        st.caption(
            "⚠ **Disclaimer:** This system is a research prototype and "
            "decision-support tool only. It is not a clinical diagnostic device "
            "and must not replace professional medical diagnosis."
        )

    elif not uploaded:
        st.info("Upload an image on the left to begin.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model information")
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| Architecture | EfficientNet-B0 |
| Training images | 194,266 |
| Test accuracy | **{FULL_DATASET_ACC}%** |
| AUC | **{FULL_DATASET_AUC}** |
| Recall (malignant) | **87.03%** |
| Precision | **85.14%** |
| Version | v0.2 |
| PRD acc target ≥ 90% | ✓ Met |
""")
    st.markdown("---")
    st.markdown("**Team Destroyers**")
    st.markdown("Harshitha · Aagusthya · Samyak")
    st.markdown("42174 AI Studio — Autumn 2026")
    st.markdown("University of Technology Sydney")
