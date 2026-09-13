"""
NeuroScan AI v3.0 — Streamlit Web Application
==============================================

Research/demo interface for brain MRI tumour segmentation,
four-class tumour classification, uncertainty estimation and explainability.

Usage:
    streamlit run app.py

Environment variables:
    NEUROSCAN_ARTIFACT_DIR  Override artifact directory path
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

# ── Package setup ──────────────────────────────────────────────────────────
# Ensure the neuroscan package is importable regardless of working directory
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from neuroscan.models import AttentionUNet, NeuroScanClassifier
from neuroscan.inference import (
    NeuroScanInference, AnalysisResult,
    CLASS_NAMES, RISK_BY_CLASS,
    ENSEMBLE_WEIGHTS, DEFAULT_MC_SAMPLES, DEFAULT_TTA_STEPS,
)
from neuroscan.explainability import (
    build_unified_explanation, build_probability_figure,
    overlay_heatmap, RISK_COLOUR,
)
from neuroscan.reporting import generate_report, REPORTLAB_AVAILABLE
from neuroscan.utils import (
    find_artifact_dir, check_artifacts, load_checkpoint,
    get_device, device_display_name, pil_to_bytes, numpy_to_png, load_metrics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroscan_app")

# ── App config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NeuroScan AI v3.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "NeuroScan AI v3.0 — Explainable Brain MRI Tumour Analysis",
    },
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero header */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #42a5f5, #7e57c2, #26c6da);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #90a4ae;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a237e22, #283593aa);
    border: 1px solid #42a5f544;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(66,165,245,0.15);
}
.metric-card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #42a5f5;
    line-height: 1.1;
}
.metric-card-label {
    font-size: 0.8rem;
    color: #90a4ae;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* Prediction result card */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    border-left: 6px solid;
    background: linear-gradient(135deg, #0d1b2a, #1a237e22);
}
.result-card.HIGH  { border-color: #E53935; }
.result-card.MEDIUM { border-color: #FB8C00; }
.result-card.LOW   { border-color: #43A047; }

.prediction-label {
    font-size: 0.85rem;
    color: #90a4ae;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 500;
}
.prediction-value {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1;
    margin: 0.3rem 0 1rem 0;
}
.prediction-value.HIGH   { color: #EF5350; }
.prediction-value.MEDIUM { color: #FFA726; }
.prediction-value.LOW    { color: #66BB6A; }

/* Status badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.badge-high   { background: #E5393533; color: #EF5350; border: 1px solid #E5393555; }
.badge-medium { background: #FB8C0033; color: #FFA726; border: 1px solid #FB8C0055; }
.badge-low    { background: #43A04733; color: #66BB6A; border: 1px solid #43A04755; }
.badge-ok     { background: #43A04733; color: #66BB6A; border: 1px solid #43A04755; }
.badge-err    { background: #E5393533; color: #EF5350; border: 1px solid #E5393555; }

/* Step indicator */
.step-box {
    background: #1a237e22;
    border: 1px solid #42a5f533;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.92rem;
    color: #cfd8dc;
}
.step-num {
    background: linear-gradient(135deg, #42a5f5, #7e57c2);
    color: white;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* Disclaimer box */
.disclaimer {
    background: #1a237e11;
    border: 1px solid #42a5f522;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.8rem;
    color: #90a4ae;
    margin: 1rem 0;
    line-height: 1.6;
}

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e3f2fd;
    border-bottom: 2px solid #42a5f544;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* Sidebar brand */
.sidebar-brand {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #42a5f5, #7e57c2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Uncertainty level pill */
.unc-low    { color: #66BB6A; font-weight: 600; }
.unc-mod    { color: #FFA726; font-weight: 600; }
.unc-high   { color: #EF5350; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED RESOURCE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_models() -> Tuple[Optional[NeuroScanInference], Dict, str, torch.device]:
    """
    Load segmenter and classifier from the artifact directory.
    Returns (inference_engine or None, artifact_status, artifact_dir_str, device).
    """
    device = get_device()
    artifact_dir = find_artifact_dir()
    status = check_artifacts(artifact_dir)

    if artifact_dir is None:
        return None, status, "", device

    # Segmenter
    try:
        segmenter = AttentionUNet(in_ch=3, filters=(32, 64, 128, 256, 512))
        seg_path = artifact_dir / "checkpoints" / "segmenter.pt"
        load_checkpoint(seg_path, segmenter, device)
        segmenter = segmenter.to(device)
        segmenter.eval()
        logger.info("Segmenter loaded from %s", seg_path)
    except Exception as exc:
        logger.error("Segmenter load failed: %s", exc)
        return None, status, str(artifact_dir), device

    # Classifier (full_image deployment arm)
    try:
        classifier = NeuroScanClassifier(
            num_classes=4, hidden=256, dropout=(0.40, 0.30), pretrained=False
        )
        clf_path = artifact_dir / "checkpoints" / "classifier_full_image.pt"
        load_checkpoint(clf_path, classifier, device)
        classifier = classifier.to(device)
        classifier.eval()
        logger.info("Classifier loaded from %s", clf_path)
    except Exception as exc:
        logger.error("Classifier load failed: %s", exc)
        return None, status, str(artifact_dir), device

    engine = NeuroScanInference(
        segmenter=segmenter,
        classifier=classifier,
        device=device,
        mc_samples=DEFAULT_MC_SAMPLES,
        tta_steps=DEFAULT_TTA_STEPS,
    )
    return engine, status, str(artifact_dir), device


@st.cache_data(show_spinner=False)
def _load_metrics_cached(artifact_dir_str: str) -> Dict:
    if not artifact_dir_str:
        return {}
    return load_metrics(Path(artifact_dir_str))


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(engine, status: Dict, artifact_dir: str, device: torch.device) -> Dict:
    """Render the sidebar and return UI settings."""
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🧠 NeuroScan AI v3.0</div>', unsafe_allow_html=True)
        st.markdown("---")

        # Model info
        st.markdown("**Model**")
        st.markdown("""
- 🔷 **Classifier:** EfficientNet-B0
- 🔷 **Segmenter:** Attention U-Net
- 🔷 **Input:** 224 × 224
- 🔷 **Classes:** 4 (Glioma, Meningioma, No Tumor, Pituitary)
        """)

        st.markdown("**Inference**")
        mc_samples_ui = st.slider("MC Dropout samples", 5, 30, DEFAULT_MC_SAMPLES, step=5)
        tta_steps_ui = st.slider("TTA transforms", 1, 10, DEFAULT_TTA_STEPS, step=1)

        fast_mode = st.checkbox(
            "⚡ Fast Demo Mode", value=False,
            help="Uses fewer MC/TTA passes. Faster but less stable uncertainty estimates.",
        )
        if fast_mode:
            st.info("Fast Demo Mode — reduced stochastic inference (10 MC, 5 TTA)")

        st.markdown("---")
        st.markdown("**System**")

        # Device
        dev_name = device_display_name(device)
        dev_icon = "🟢" if device.type == "cuda" else "⚪"
        st.markdown(f"{dev_icon} **Device:** {dev_name}")

        # PyTorch version
        st.markdown(f"🔧 **PyTorch:** {torch.__version__}")

        # Artifact health
        st.markdown("**Artifact Health**")
        checks = {
            "Segmenter": status.get("checkpoints/segmenter.pt", False),
            "Classifier": status.get("checkpoints/classifier_full_image.pt", False),
            "Metrics": status.get("metrics/run_summary.json", False),
            "Figures": status.get("figures/arm_comparison.png", False),
            "XAI panels": status.get("xai/unified_glioma.png", False),
        }
        for name, ok in checks.items():
            icon = "✅" if ok else "❌"
            st.markdown(f"{icon} {name}")

        if artifact_dir:
            st.caption(f"📁 `{artifact_dir}`")

        st.markdown("---")
        st.markdown(
            "⚠️ **Research use only.**  \nNot a clinical diagnostic tool.",
            unsafe_allow_html=False,
        )

    return {
        "mc_samples": mc_samples_ui,
        "tta_steps": tta_steps_ui,
        "fast_mode": fast_mode,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard(artifact_dir: str) -> None:
    st.markdown('<div class="hero-title">NeuroScan AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Explainable Brain MRI Tumour Analysis</div>',
        unsafe_allow_html=True,
    )

    # Quick-stats cards
    cols = st.columns(4)
    stats = [
        ("4", "Tumour Classes"),
        ("224×224", "Input Resolution"),
        ("30", "MC Dropout Passes"),
        ("10", "TTA Transforms"),
    ]
    for col, (val, lbl) in zip(cols, stats):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-card-value">{val}</div>'
                f'<div class="metric-card-label">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="section-header">🔬 Analysis Pipeline</div>', unsafe_allow_html=True)
        steps = [
            "Upload brain MRI (JPEG / PNG / BMP / TIFF)",
            "Brain extraction (Otsu + largest contour)",
            "Attention U-Net tumour segmentation",
            "EfficientNet-B0 classification (full-image domain)",
            "MC Dropout uncertainty estimation (30 passes)",
            "Deterministic TTA (10 transforms)",
            "Fitted ensemble (0.75 MC + 0.25 TTA)",
            "Grad-CAM explainability",
            "Research PDF report generation",
        ]
        for i, step in enumerate(steps, 1):
            st.markdown(
                f'<div class="step-box">'
                f'<div class="step-num">{i}</div>{step}'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown('<div class="section-header">📊 Tumour Classes</div>', unsafe_allow_html=True)

        class_info = {
            "🔴 Glioma":      ("HIGH risk",   "Most common primary brain tumour (glial cells)"),
            "🟠 Meningioma":  ("MEDIUM risk",  "Typically benign, arises from meninges"),
            "🟢 No Tumor":    ("LOW risk",     "No tumour features detected in this image"),
            "🔵 Pituitary":   ("MEDIUM risk",  "Benign pituitary gland adenoma"),
        }
        for cls, (risk, desc) in class_info.items():
            st.markdown(f"**{cls}** — {risk}")
            st.caption(desc)
            st.divider()

        st.markdown('<div class="section-header">⚡ Ensemble Strategy</div>', unsafe_allow_html=True)
        st.markdown("""
| Component | Weight |
|---|---|
| Standard (single pass) | 0.00 |
| MC Dropout (30 passes) | **0.75** |
| Deterministic TTA (10 transforms) | **0.25** |

Weights selected by grid search on the validation split.
The held-out test split took no part in this selection.
        """)

        # Disclaimer
        st.markdown(
            '<div class="disclaimer">'
            '⚠️ <b>Research &amp; Demonstration Use Only</b><br>'
            'NeuroScan AI is an experimental research system. Its outputs are '
            'not a medical diagnosis and must not be used as a substitute for '
            'evaluation by a qualified radiologist or clinician. The system '
            'analyses a single 2D MRI image and may produce incorrect or '
            'uncertain predictions.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Pre-generated XAI panels
    if artifact_dir:
        st.markdown("---")
        st.markdown('<div class="section-header">🖼️ Pre-Generated Explanations (Test Set)</div>', unsafe_allow_html=True)
        xai_cols = st.columns(4)
        for col, cls in zip(xai_cols, CLASS_NAMES):
            xai_path = Path(artifact_dir) / "xai" / f"unified_{cls}.png"
            if xai_path.exists():
                with col:
                    st.image(str(xai_path), caption=cls.replace("_", " ").title(), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: ANALYZE MRI
# ══════════════════════════════════════════════════════════════════════════════

def render_analyze_tab(engine: Optional[NeuroScanInference], ui_settings: Dict) -> None:
    st.markdown('<div class="section-header">🔬 Upload & Analyze Brain MRI</div>', unsafe_allow_html=True)

    if engine is None:
        st.error(
            "**NeuroScan models are not loaded.**\n\n"
            "Please verify that the artifact directory contains:\n"
            "- `checkpoints/segmenter.pt`\n"
            "- `checkpoints/classifier_full_image.pt`\n\n"
            "Set the `NEUROSCAN_ARTIFACT_DIR` environment variable to your artifact path, "
            "or place the `neuroscan_v3_artifacts/` folder next to `app.py`."
        )
        return

    uploaded = st.file_uploader(
        "Upload a brain MRI image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key="mri_upload",
        help="Upload a 2D brain MRI scan for AI analysis.",
    )

    if uploaded is not None:
        image_bytes = uploaded.read()

        # Preview: original + brain crop
        col_orig, col_crop = st.columns(2)
        with col_orig:
            st.image(image_bytes, caption="📁 Uploaded MRI", width='stretch')
        with col_crop:
            try:
                arr = np.frombuffer(image_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    from neuroscan.preprocessing import brain_crop_from_array
                    brain_rgb = brain_crop_from_array(bgr)
                    st.image(brain_rgb, caption="🧠 Brain Crop (224×224)", width='stretch')
            except Exception as e:
                st.warning(f"Preview failed: {e}")

        st.markdown("---")

        # Analyze button
        run_analysis = st.button(
            "🚀 Analyze MRI",
            type="primary",
            width='stretch',
        )

        if run_analysis:
            _run_full_analysis(engine, image_bytes, ui_settings, uploaded.name)

    # Show cached result
    elif "analysis_result" in st.session_state:
        st.info("📋 Showing previous analysis result. Upload a new image to reanalyze.")
        _render_results_from_session()


def _run_full_analysis(
    engine: NeuroScanInference,
    image_bytes: bytes,
    ui_settings: Dict,
    filename: str,
) -> None:
    """Run the full inference pipeline with progress feedback."""

    steps_container = st.empty()
    progress = st.progress(0)

    def update(msg: str, pct: int) -> None:
        steps_container.info(f"⏳ {msg}")
        progress.progress(pct)

    try:
        update("Loading MRI...", 5)
        time.sleep(0.05)

        update("Extracting brain region...", 15)
        time.sleep(0.05)

        # Update MC/TTA based on UI or fast mode
        fast = ui_settings["fast_mode"]
        engine.mc_samples = 10 if fast else ui_settings["mc_samples"]
        engine.tta_steps = 5 if fast else ui_settings["tta_steps"]

        update("Running Attention U-Net segmentation...", 30)
        result: AnalysisResult = engine.analyze(image_bytes, fast_mode=fast)

        update("Estimating uncertainty (MC Dropout)...", 60)
        time.sleep(0.05)

        update("Running deterministic TTA...", 75)
        time.sleep(0.05)

        update("Computing Grad-CAM explanation...", 88)
        time.sleep(0.05)

        update("Preparing results...", 95)

        # Build unified explanation PNG
        geom_contours = result.geometry["contours"] if result.geometry else None
        unified_arr = build_unified_explanation(
            brain_rgb=result.brain_crop,
            seg_prob=result.segmentation_probability,
            roi_overlay=result.roi_overlay,
            gradcam_overlay=result.gradcam_overlay,
            gradcam_heatmap=result.gradcam,
            prediction=result.prediction,
            confidence=result.confidence,
            uncertainty=result.uncertainty,
            uncertainty_lvl=result.uncertainty_level,
            probabilities=result.probabilities,
            per_class_sigma=result.per_class_sigma,
            geometry_contours=geom_contours,
        )

        # Build probability figure
        prob_fig_arr = build_probability_figure(
            result.probabilities, result.per_class_sigma, result.prediction
        )

        # Generate PDF
        gradcam_overlay_png = pil_to_bytes(result.gradcam_overlay)
        pdf_bytes = b""
        if REPORTLAB_AVAILABLE:
            try:
                pdf_bytes = generate_report(result, gradcam_overlay_png)
            except Exception as exc:
                logger.warning("PDF generation failed: %s", exc)

        # Store in session state
        st.session_state["analysis_result"] = result
        st.session_state["unified_png"] = pil_to_bytes(unified_arr)
        st.session_state["prob_fig_png"] = pil_to_bytes(prob_fig_arr)
        st.session_state["gradcam_overlay_png"] = gradcam_overlay_png
        st.session_state["pdf_bytes"] = pdf_bytes
        st.session_state["analysis_filename"] = filename

        steps_container.success("✅ Analysis complete!")
        progress.progress(100)
        time.sleep(0.3)
        steps_container.empty()
        progress.empty()

        _render_results_from_session()

    except MemoryError:
        steps_container.empty()
        progress.empty()
        st.error(
            "❌ **Out of memory.** Try Fast Demo Mode or use a CPU-only session. "
            "Make sure no other applications are using the GPU."
        )
    except Exception as exc:
        steps_container.empty()
        progress.empty()
        st.error(f"❌ **Analysis failed:** {exc}")
        with st.expander("🛠️ Technical details"):
            st.code(traceback.format_exc())


def _render_results_from_session() -> None:
    """Render the analysis results from st.session_state."""
    result: AnalysisResult = st.session_state.get("analysis_result")
    if result is None:
        return

    unified_png: bytes = st.session_state.get("unified_png", b"")
    prob_fig_png: bytes = st.session_state.get("prob_fig_png", b"")
    gradcam_overlay_png: bytes = st.session_state.get("gradcam_overlay_png", b"")
    pdf_bytes: bytes = st.session_state.get("pdf_bytes", b"")
    filename: str = st.session_state.get("analysis_filename", "mri")

    # ── RESULT HEADER CARD ──────────────────────────────────────────────────
    risk = result.risk
    risk_cls = risk.lower() if risk.lower() in ("high", "medium", "low") else "low"
    pred_display = result.prediction.upper().replace("_", " ")
    unc_css = {"Low": "unc-low", "Moderate": "unc-mod", "High": "unc-high"}.get(
        result.uncertainty_level, "unc-low"
    )

    st.markdown(
        f'<div class="result-card {risk}">'
        f'<div class="prediction-label">AI Model Prediction</div>'
        f'<div class="prediction-value {risk}">{pred_display}</div>'
        f'<table style="width:100%; border-collapse:collapse;">'
        f'<tr>'
        f'<td style="padding:0.5rem; min-width:130px;">'
        f'<div style="color:#90a4ae; font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;">Confidence</div>'
        f'<div style="font-size:1.8rem; font-weight:700; color:#e3f2fd;">{result.confidence*100:.1f}%</div>'
        f'</td>'
        f'<td style="padding:0.5rem; min-width:130px;">'
        f'<div style="color:#90a4ae; font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;">Uncertainty (σ)</div>'
        f'<div style="font-size:1.8rem; font-weight:700; color:#e3f2fd;">{result.uncertainty:.4f}</div>'
        f'</td>'
        f'<td style="padding:0.5rem; min-width:130px;">'
        f'<div style="color:#90a4ae; font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;">Uncertainty Level</div>'
        f'<div style="font-size:1.4rem; font-weight:700;" class="{unc_css}">{result.uncertainty_level.upper()}</div>'
        f'</td>'
        f'<td style="padding:0.5rem;">'
        f'<div style="color:#90a4ae; font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;">Model Risk Category</div>'
        f'<div style="font-size:1.4rem; font-weight:700; color:{RISK_COLOUR[risk]};">{risk}</div>'
        f'</td>'
        f'</tr>'
        f'</table>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="disclaimer">'
        '⚠️ <b>Research result only.</b> This AI output is not a clinical diagnosis '
        'and must not be used to direct patient care. Correlate with full clinical '
        'history and radiological assessment.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── PROBABILITY BAR CHART ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Class Probabilities</div>', unsafe_allow_html=True)
    if prob_fig_png:
        st.image(prob_fig_png, width='stretch')

    # ── SEGMENTATION VIEW ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Tumour Segmentation</div>', unsafe_allow_html=True)
    seg_cols = st.columns(4)
    with seg_cols[0]:
        st.image(result.brain_crop, caption="Brain Crop", width='stretch')
    with seg_cols[1]:
        prob_img = (result.segmentation_probability * 255).astype(np.uint8)
        prob_rgb = cv2.applyColorMap(prob_img, cv2.COLORMAP_HOT)[:, :, ::-1]
        st.image(prob_rgb, caption="Tumour Probability (U-Net)", width='stretch')
    with seg_cols[2]:
        mask_disp = (result.segmentation_mask * 255).astype(np.uint8)
        st.image(mask_disp, caption="Thresholded Mask (t=0.5)", width='stretch')
    with seg_cols[3]:
        st.image(result.roi_overlay, caption="ROI Contour + Box", width='stretch')

    if result.geometry is None:
        st.info("ℹ️ No reliable tumour region was segmented in this image.")

    # ── TUMOUR GEOMETRY ─────────────────────────────────────────────────────
    if result.geometry:
        g = result.geometry
        st.markdown('<div class="section-header">📐 Tumour Region Geometry</div>', unsafe_allow_html=True)
        st.caption("All measurements are in image-space pixels (224×224). "
                   "Physical dimensions cannot be determined without calibration.")
        gcols = st.columns(4)
        geom_metrics = [
            (f"{g['area_px']:.0f} px²",             "Area"),
            (f"{g['circularity']:.3f}",              "Circularity"),
            (g["quadrant"],                           "Quadrant"),
            (f"{g['area_fraction']*100:.1f}%",       "Area Fraction"),
        ]
        for col, (val, lbl) in zip(gcols, geom_metrics):
            with col:
                st.metric(lbl, val)

        gcols2 = st.columns(3)
        with gcols2[0]:
            st.metric("Centroid (x, y)", f"({g['centroid'][0]}, {g['centroid'][1]})")
        with gcols2[1]:
            st.metric("Perimeter", f"{g['perimeter_px']:.1f} px")
        with gcols2[2]:
            st.metric("Threshold", "0.50")

    # ── GRAD-CAM VIEW ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔥 Grad-CAM Explainability</div>', unsafe_allow_html=True)
    st.caption(
        "Highlighted regions indicate areas that contributed strongly to the model's "
        "classification decision. This does not confirm the presence of a tumour. "
        "Grad-CAM is computed on the full-image domain (matching the deployment training domain)."
    )
    gc_cols = st.columns(3)
    with gc_cols[0]:
        st.image(result.brain_crop, caption="Classifier Input (Full Image)", width='stretch')
    with gc_cols[1]:
        hm_img = (result.gradcam * 255).astype(np.uint8)
        hm_rgb = cv2.applyColorMap(hm_img, cv2.COLORMAP_JET)[:, :, ::-1]
        st.image(hm_rgb, caption="Grad-CAM Heatmap", width='stretch')
    with gc_cols[2]:
        st.image(result.gradcam_overlay, caption="Overlay", width='stretch')

    # ── UNIFIED EXPLANATION ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧬 Unified Explanation Panel</div>', unsafe_allow_html=True)
    if unified_png:
        st.image(unified_png, width='stretch')

    # ── DOWNLOAD BUTTONS ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)
    dl_cols = st.columns(4)

    with dl_cols[0]:
        if pdf_bytes:
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"neuroscan_report_{result.prediction}.pdf",
                mime="application/pdf",
                width='stretch',
            )
        else:
            st.button("📄 PDF (reportlab unavailable)", disabled=True, width='stretch')

    with dl_cols[1]:
        if unified_png:
            st.download_button(
                "🖼️ Download Explanation PNG",
                data=unified_png,
                file_name=f"neuroscan_explanation_{result.prediction}.png",
                mime="image/png",
                width='stretch',
            )

    with dl_cols[2]:
        seg_png = numpy_to_png(result.segmentation_probability)
        st.download_button(
            "🎯 Download Segmentation PNG",
            data=seg_png,
            file_name=f"neuroscan_segmentation_{result.prediction}.png",
            mime="image/png",
            width='stretch',
        )

    with dl_cols[3]:
        result_json = json.dumps(result.to_dict(), indent=2)
        st.download_button(
            "📊 Download JSON Result",
            data=result_json.encode(),
            file_name=f"neuroscan_result_{result.prediction}.json",
            mime="application/json",
            width='stretch',
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def render_explainability_tab(artifact_dir: str) -> None:
    st.markdown('<div class="section-header">🧬 Explainability & Model Interpretation</div>', unsafe_allow_html=True)

    st.markdown("""
Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions
of the input image the model attends to when making its prediction.

**Key properties of this implementation:**
- Target layer: last `Conv2d` of EfficientNet-B0 feature extractor
- Input explicitly set `requires_grad=True` (fixes v2 bug with frozen backbone)
- Uses `register_full_backward_hook` (fixes v2 deprecation issue)
- Computed on the **full-image domain** — same domain the classifier was trained on
- Hooks are registered and removed per-request (no cross-request leakage)
    """)

    # Show result if available
    result: Optional[AnalysisResult] = st.session_state.get("analysis_result")
    if result:
        st.markdown('<div class="section-header">Current Image Analysis</div>', unsafe_allow_html=True)

        gc_cols = st.columns(3)
        with gc_cols[0]:
            st.image(result.brain_crop, caption="Full-Image Input", width='stretch')
        with gc_cols[1]:
            hm_u8 = (result.gradcam * 255).astype(np.uint8)
            hm_rgb = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)[:, :, ::-1]
            st.image(hm_rgb, caption="Grad-CAM Heatmap", width='stretch')
        with gc_cols[2]:
            st.image(result.gradcam_overlay, caption="Heatmap Overlay (α=0.40)", width='stretch')

        # Per-class sigma table
        st.markdown("**Per-Class MC Dropout Uncertainty (σ)**")
        sigma_data = {
            "Class": CLASS_NAMES,
            "Ensemble Prob (%)": [f"{p*100:.2f}" for p in result.probabilities],
            "MC Prob (%)": [f"{p*100:.2f}" for p in result.mc_probabilities],
            "Sigma (σ)": [f"{s:.4f}" for s in result.per_class_sigma],
            "Selected": ["✅" if i == result.class_index else "" for i in range(4)],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(sigma_data), width='stretch', hide_index=True)

    else:
        st.info("Upload and analyze an MRI in the **Analyze MRI** tab to see explanations here.")

    # Pre-generated panels
    if artifact_dir:
        st.markdown("---")
        st.markdown('<div class="section-header">Pre-Generated Explanations (Test Set)</div>', unsafe_allow_html=True)
        st.caption("These panels were generated from the official BRISC2025 test set during training.")
        xai_tabs = st.tabs([c.replace("_", " ").title() for c in CLASS_NAMES])
        for tab, cls in zip(xai_tabs, CLASS_NAMES):
            with tab:
                xai_path = Path(artifact_dir) / "xai" / f"unified_{cls}.png"
                if xai_path.exists():
                    st.image(str(xai_path), width='stretch')
                else:
                    st.warning(f"Not found: {xai_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def render_performance_tab(metrics: Dict, artifact_dir: str) -> None:
    st.markdown('<div class="section-header">📈 Model Performance (Benchmark)</div>', unsafe_allow_html=True)
    st.caption(
        "All metrics are computed on the **official BRISC2025 held-out test set**, "
        "which was never used during training or hyperparameter selection."
    )
    st.info("📌 These are benchmark metrics from the trained model evaluation, "
            "not the result for your uploaded image.")

    run = metrics.get("run_summary", {})

    if not run:
        st.warning("Metrics JSON files not found. Place `metrics/run_summary.json` in the artifact directory.")
        return

    # Dataset stats
    st.markdown("### Dataset")
    splits = run.get("splits", {})
    dedup = run.get("deduplication", {})
    ds_cols = st.columns(5)
    ds_stats = [
        (f"{splits.get('train', 'N/A'):,}", "Train"),
        (f"{splits.get('val', 'N/A'):,}", "Validation"),
        (f"{splits.get('test', 'N/A'):,}", "Test"),
        (f"{dedup.get('duplicate', 'N/A')}", "Duplicates removed"),
        (f"{dedup.get('cross_split_duplicate', 'N/A')}", "Cross-split dups"),
    ]
    for col, (val, lbl) in zip(ds_cols, ds_stats):
        with col:
            st.metric(lbl, val)

    st.markdown("---")

    # Segmentation
    st.markdown("### Segmentation — Attention U-Net")
    seg_data = run.get("segmentation", {})
    seg_cols = st.columns(4)
    seg_metrics = [
        (f"{seg_data.get('val_dice', 0):.4f}",   "Val Dice"),
        (f"{seg_data.get('val_iou', 0):.4f}",    "Val IoU"),
        (f"{seg_data.get('test_dice', 0):.4f}",  "Test Dice"),
        (f"{seg_data.get('test_iou', 0):.4f}",   "Test IoU"),
    ]
    for col, (val, lbl) in zip(seg_cols, seg_metrics):
        with col:
            st.metric(lbl, val)

    st.markdown("---")

    # Classification arms comparison
    st.markdown("### Classification — Full Image vs ROI-Guided")

    arms = run.get("arms", {})
    fi = arms.get("full_image", {})
    roi = arms.get("roi_guided", {})

    import pandas as pd
    clf_table = {
        "Metric":             ["Accuracy", "Macro F1", "Balanced Acc", "Precision", "Recall", "Cohen κ", "Macro AUC"],
        "Full Image (✅ deployed)": [
            f"{fi.get('accuracy', 0):.4f}",
            f"{fi.get('macro_f1', 0):.4f}",
            f"{fi.get('balanced_accuracy', 0):.4f}",
            f"{fi.get('macro_precision', 0):.4f}",
            f"{fi.get('macro_recall', 0):.4f}",
            f"{fi.get('cohen_kappa', 0):.4f}",
            f"{fi.get('macro_auc', 0):.4f}",
        ],
        "ROI-Guided": [
            f"{roi.get('accuracy', 0):.4f}",
            f"{roi.get('macro_f1', 0):.4f}",
            f"{roi.get('balanced_accuracy', 0):.4f}",
            f"{roi.get('macro_precision', 0):.4f}",
            f"{roi.get('macro_recall', 0):.4f}",
            f"{roi.get('cohen_kappa', 0):.4f}",
            f"{roi.get('macro_auc', 0):.4f}",
        ],
    }
    st.dataframe(pd.DataFrame(clf_table), width='stretch', hide_index=True)

    mcnemar = run.get("mcnemar", {})
    if mcnemar:
        p_val = mcnemar.get("p_value", 1.0)
        st.caption(
            f"McNemar test: p = {p_val:.4f} — the full-image arm was selected "
            f"as statistically significantly better (p < 0.05)."
        )

    st.markdown("---")

    # Ensemble metrics
    ens = run.get("ensemble_test_metrics", {})
    if ens:
        st.markdown("### Ensemble (Final Deployed Configuration)")
        ens_cols = st.columns(4)
        ens_stats = [
            (f"{ens.get('accuracy', 0):.4f}",    "Accuracy"),
            (f"{ens.get('macro_f1', 0):.4f}",    "Macro F1"),
            (f"{ens.get('cohen_kappa', 0):.4f}", "Cohen κ"),
            (f"{ens.get('macro_auc', 0):.4f}",   "Macro AUC"),
        ]
        for col, (val, lbl) in zip(ens_cols, ens_stats):
            with col:
                st.metric(lbl, val)

        ew = run.get("ensemble_weights", {})
        st.caption(
            f"Ensemble weights — standard: {ew.get('standard', 0):.2f} | "
            f"MC Dropout: {ew.get('mc_dropout', 0):.2f} | "
            f"TTA: {ew.get('tta', 0):.2f}"
        )

    # Figures
    if artifact_dir:
        st.markdown("---")
        st.markdown("### Training Curves & Comparison")
        fig_tabs = st.tabs(["Segmentation Curves", "ARM Curves", "ARM Comparison", "Test Panels"])
        fig_names = [
            "figures/segmentation_curves.png",
            "figures/arm_curves.png",
            "figures/arm_comparison.png",
            "figures/test_panels.png",
        ]
        for tab, fname in zip(fig_tabs, fig_names):
            with tab:
                p = Path(artifact_dir) / fname
                if p.exists():
                    st.image(str(p), width='stretch')
                else:
                    st.warning(f"Not found: {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: TECHNICAL DETAILS
# ══════════════════════════════════════════════════════════════════════════════

def render_technical_tab(device: torch.device) -> None:
    st.markdown('<div class="section-header">⚙️ Technical Details</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Classifier**")
        st.json({
            "Architecture": "EfficientNet-B0",
            "Head": "Dropout(0.40) → Linear → BatchNorm1d → ReLU → Dropout(0.30) → Linear",
            "Hidden dim": 256,
            "Output": "4 logits (softmax at inference)",
            "Deployed arm": "full_image",
            "Classes": CLASS_NAMES,
        })

        st.markdown("**Segmenter**")
        st.json({
            "Architecture": "Attention U-Net",
            "Encoder": "5 levels (32, 64, 128, 256, 512 filters)",
            "Decoder": "4 levels with attention gates",
            "Input": "224 × 224 × 3",
            "Output": "1 channel segmentation logits (sigmoid applied once)",
            "Params": "~7.98M",
        })

    with col_r:
        st.markdown("**Inference Configuration**")
        st.json({
            "MC Dropout samples": DEFAULT_MC_SAMPLES,
            "TTA transforms": DEFAULT_TTA_STEPS,
            "Ensemble weights": {
                "standard": ENSEMBLE_WEIGHTS[0],
                "mc_dropout": ENSEMBLE_WEIGHTS[1],
                "tta": ENSEMBLE_WEIGHTS[2],
            },
            "Uncertainty bands": {
                "Low":      "σ < 0.05",
                "Moderate": "0.05 ≤ σ < 0.12",
                "High":     "σ ≥ 0.12",
            },
        })

        st.markdown("**Preprocessing**")
        st.json({
            "Brain extraction": "Otsu thresholding + largest external contour",
            "Morphology": "Erosion (2 iterations) + Dilation (2 iterations)",
            "Resize": "224 × 224 (INTER_AREA)",
            "Normalization": "ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
            "ROI threshold": 0.5,
            "ROI padding": "15%",
            "ROI min fraction": "10%",
        })

    st.markdown("---")
    st.markdown("**System**")
    sys_cols = st.columns(3)
    with sys_cols[0]:
        st.metric("Device", device_display_name(device))
    with sys_cols[1]:
        st.metric("PyTorch", torch.__version__)
    with sys_cols[2]:
        cuda_str = torch.version.cuda if torch.cuda.is_available() else "N/A"
        st.metric("CUDA", cuda_str)

    st.markdown("**TTA Transform Bank**")
    tta_data = {
        "Index": list(range(1, 11)),
        "Transform": [
            "Identity", "Horizontal Flip",
            "Rotation +8°", "Rotation −8°",
            "Rotation +15°", "Rotation −15°",
            "HFlip + Rotation +8°", "HFlip + Rotation −8°",
            "Scale × 0.92 (center crop)", "Scale × 1.08 (center crop)",
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(tta_data), width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def render_about_tab() -> None:
    st.markdown('<div class="hero-title">NeuroScan AI v3.0</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Explainable Brain MRI Tumour Analysis Research System</div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
## About

**NeuroScan AI v3.0** is an experimental AI research system for brain MRI tumour
analysis. It combines deep learning segmentation, classification, uncertainty
estimation, and explainability into a unified inference pipeline.

### Pipeline

1. **Brain Extraction** — Otsu thresholding + largest external contour crop
2. **Attention U-Net Segmentation** — tumour probability map + ROI extraction
3. **EfficientNet-B0 Classification** — full-image domain (deployed arm)
4. **MC Dropout Uncertainty** — 30 stochastic passes with BatchNorm frozen
5. **Deterministic TTA** — 10 geometric transforms
6. **Fitted Ensemble** — 75% MC Dropout + 25% TTA (selected by validation grid search)
7. **Grad-CAM Explanation** — last backbone Conv2d, same domain as training
8. **Research Report** — ReportLab PDF with provenance and disclaimers

### Dataset

**BRISC2025** (briscdataset/brisc2025) — 5,950 usable images after deduplication.

- Train: 4,211 | Validation: 744 | Test: 995

### Classes

| Class | Risk | Description |
|---|---|---|
| Glioma | HIGH | Primary brain tumour, glial cells |
| Meningioma | MEDIUM | Benign, meningeal origin |
| No Tumor | LOW | No tumour features detected |
| Pituitary | MEDIUM | Benign pituitary adenoma |

### Key Results (Test Set)

| Configuration | Accuracy | Macro F1 | Cohen κ |
|---|---|---|---|
| Full Image (single pass) | 98.59% | 98.62% | 0.9808 |
| ROI-Guided (single pass) | 97.49% | 97.56% | 0.9657 |
| Ensemble (deployed) | **98.79%** | **98.84%** | **0.9835** |

Segmentation: Test Dice 0.8400 | Test IoU 0.7756
        """)

    with col_r:
        st.markdown("""
## Deployment Decision

The **full-image arm** was selected for deployment over ROI-guided based on:

- Higher accuracy (+1.1%) and macro F1 (+1.1%)
- McNemar test p = 0.0433 (statistically significant)
- Segmentation still runs for visualization

## Grad-CAM v2 Bug Fixes

Both bugs from NeuroScan v2 are fixed:

1. Input tensor explicitly requires `requires_grad=True`
   (necessary when backbone is frozen)
2. `register_full_backward_hook` replaces deprecated
   `register_backward_hook`

## Safety Notice
        """)
        st.error(
            "**Research & Demonstration Only**\n\n"
            "NeuroScan AI is an experimental research system. "
            "Its outputs are NOT a medical diagnosis and must not be used "
            "as a substitute for evaluation by a qualified radiologist or clinician. "
            "The system analyses a single 2D MRI image and may produce "
            "incorrect or uncertain predictions.",
            icon="⚠️",
        )

        st.markdown("""
## Running This App

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Expose via ngrok (for mobile demo)
ngrok http 8501
```

Set `NEUROSCAN_ARTIFACT_DIR` to point to your artifact directory if it's
not in the default location next to `app.py`.
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Load models (cached)
    with st.spinner("🔄 Loading NeuroScan AI models..."):
        engine, status, artifact_dir, device = _load_models()

    # Load metrics (cached)
    metrics = _load_metrics_cached(artifact_dir)

    # Sidebar
    ui_settings = render_sidebar(engine, status, artifact_dir, device)

    # Navigation tabs
    tab_names = ["🏠 Dashboard", "🔬 Analyze MRI", "🧬 Explainability", "📈 Performance", "⚙️ Technical", "ℹ️ About"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_dashboard(artifact_dir)

    with tabs[1]:
        render_analyze_tab(engine, ui_settings)

    with tabs[2]:
        render_explainability_tab(artifact_dir)

    with tabs[3]:
        render_performance_tab(metrics, artifact_dir)

    with tabs[4]:
        render_technical_tab(device)

    with tabs[5]:
        render_about_tab()


if __name__ == "__main__":
    main()
