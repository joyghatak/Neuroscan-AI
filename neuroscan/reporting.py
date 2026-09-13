"""
NeuroScan AI v3.0 — PDF Report generation
Adapted from NeuroScan_V3.ipynb cell 34.
"""
from __future__ import annotations

import datetime
import io
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .inference import CLASS_NAMES, RISK_BY_CLASS, ENSEMBLE_WEIGHTS, AnalysisResult

# ReportLab imports (guarded for graceful failure)
try:
    from reportlab.lib import colors as rl
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ── Colour palette ──────────────────────────────────────────────────────────
if REPORTLAB_AVAILABLE:
    DARK  = rl.HexColor("#1a237e")
    BLUE  = rl.HexColor("#1565c0")
    MED   = rl.HexColor("#42a5f5")
    MGRAY = rl.HexColor("#bdbdbd")
    LGRAY = rl.HexColor("#f5f5f5")
    BODY  = rl.HexColor("#212121")

    RISK_RL = {
        "HIGH":   rl.HexColor("#E53935"),
        "MEDIUM": rl.HexColor("#FB8C00"),
        "LOW":    rl.HexColor("#43A047"),
    }

# ── Interpretation text ─────────────────────────────────────────────────────
INTERPRETATION: Dict[str, str] = {
    "glioma": (
        "Gliomas are the most common primary brain tumours, arising from glial cells. "
        "They are classified by grade (I–IV), with high-grade glioblastoma multiforme "
        "(GBM, WHO grade IV) being the most aggressive. MRI findings typically include "
        "irregular enhancement, central necrosis, perilesional oedema, and mass effect. "
        "The AI model detected features in this image consistent with glioma morphology."
    ),
    "meningioma": (
        "Meningiomas arise from the meninges and are typically benign (WHO grade I), "
        "though higher-grade variants exist. They commonly appear as well-circumscribed, "
        "homogeneously enhancing extra-axial masses with a dural tail sign. The AI model "
        "identified features consistent with this pattern."
    ),
    "no_tumor": (
        "The AI model did not detect features associated with the tumour classes present "
        "in the training data (glioma, meningioma, pituitary adenoma). This result "
        "should be interpreted in conjunction with clinical findings and complete "
        "radiological assessment. A negative AI result does not exclude pathology."
    ),
    "pituitary": (
        "Pituitary adenomas are benign tumours of the pituitary gland. They are "
        "classified as microadenomas (<10 mm) or macroadenomas (≥10 mm). Common "
        "MRI findings include a sellar or suprasellar mass, heterogeneous enhancement, "
        "and deviation of the pituitary stalk. The AI model identified features "
        "consistent with pituitary region pathology."
    ),
}

RECOMMENDATION: Dict[str, str] = {
    "glioma": (
        "Urgent multidisciplinary neuro-oncology review is recommended. "
        "Advanced MRI sequences (perfusion, spectroscopy, diffusion) and neurosurgical "
        "consultation should be considered."
    ),
    "meningioma": (
        "Neurosurgical and neuro-radiological assessment is recommended. "
        "Management depends on tumour size, location, and clinical symptoms. "
        "Regular imaging follow-up may be appropriate for incidental small meningiomas."
    ),
    "no_tumor": (
        "Clinical correlation is required. If clinical suspicion remains, "
        "further imaging with contrast enhancement, advanced sequences, or "
        "specialist radiological review is recommended."
    ),
    "pituitary": (
        "Endocrinological assessment and specialist neuro-radiological evaluation "
        "are recommended. Hormonal profiling and visual field testing should be "
        "considered as clinically indicated."
    ),
}


# ── Table builder ───────────────────────────────────────────────────────────

def _table(data, widths, header_fill=None):
    if header_fill is None:
        header_fill = BLUE
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  header_fill),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  rl.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LGRAY, rl.white]),
        ("GRID",         (0, 0), (-1, -1), 0.5, MGRAY),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TEXTCOLOR",    (0, 1), (-1, -1), BODY),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ── Report generator ────────────────────────────────────────────────────────

def generate_report(result: AnalysisResult, gradcam_overlay_png: Optional[bytes] = None) -> bytes:
    """
    Generate a ReportLab PDF for the analysis result.

    Args:
        result: AnalysisResult from the inference engine.
        gradcam_overlay_png: Optional PNG bytes of the Grad-CAM overlay image.

    Returns:
        PDF file as bytes.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "reportlab is not installed. Run: pip install reportlab"
        )

    buf = io.BytesIO()
    ss = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "T", parent=ss["Normal"], fontName="Helvetica-Bold",
        fontSize=16, textColor=DARK, alignment=TA_CENTER, spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "S", parent=ss["Normal"], fontName="Helvetica",
        fontSize=9, textColor=BLUE, alignment=TA_CENTER, spaceAfter=4,
    )
    h1_style = ParagraphStyle(
        "H1", parent=ss["Normal"], fontName="Helvetica-Bold",
        fontSize=10.5, textColor=DARK, spaceBefore=9, spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "B", parent=ss["Normal"], fontName="Helvetica", fontSize=8.5,
        textColor=BODY, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=3,
    )
    small_style = ParagraphStyle(
        "D", parent=ss["Normal"], fontName="Helvetica-Oblique",
        fontSize=7.5, textColor=BLUE, leading=10.5, alignment=TA_JUSTIFY,
    )

    prediction_key = result.prediction
    risk = result.risk
    now = datetime.datetime.now()
    rid = "NS3-" + now.strftime("%Y%m%d-%H%M%S")

    elements = []

    # Header
    elements.append(Paragraph("NeuroScan AI v3.0 — Research Analysis Report", title_style))
    elements.append(Paragraph("AI-assisted neuro-radiological screening (research use only)", sub_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=MED, spaceAfter=7))

    # Meta table
    elements.append(_table(
        [
            ["Report ID", rid, "Generated", now.strftime("%d %b %Y %H:%M")],
            ["Deployed arm", "full_image", "Backbone", "EfficientNet-B0"],
            ["Segmenter", "Attention U-Net", "Input size", "224×224"],
        ],
        [2.4 * cm, 6.4 * cm, 2.4 * cm, 5.8 * cm],
    ))
    elements.append(Spacer(1, 8))

    # Primary AI impression
    elements.append(Paragraph("PRIMARY AI IMPRESSION", h1_style))
    diag = _table(
        [
            ["AI impression",        prediction_key.upper()],
            ["Ensemble confidence",  f"{result.confidence * 100:.2f}%"],
            ["Risk stratum",         risk],
            ["Predictive uncertainty", f"σ = {result.uncertainty:.4f} ({result.uncertainty_level})"],
        ],
        [4.6 * cm, 12.4 * cm],
    )
    diag.setStyle(TableStyle([
        ("TEXTCOLOR", (1, 2), (1, 2), RISK_RL[risk]),
        ("FONTNAME",  (1, 0), (1, 0), "Helvetica-Bold"),
        ("FONTSIZE",  (1, 0), (1, 0), 11),
    ]))
    elements.append(diag)
    elements.append(Spacer(1, 8))

    # Class posterior
    elements.append(Paragraph("CLASS POSTERIOR", h1_style))
    rows = [["Class", "Ensemble", "Standard", "MC Dropout", "TTA", "MC σ", ""]]
    for i, c in enumerate(CLASS_NAMES):
        rows.append([
            c,
            f"{result.probabilities[i] * 100:.2f}%",
            f"{result.standard_probabilities[i] * 100:.2f}%",
            f"{result.mc_probabilities[i] * 100:.2f}%",
            f"{result.tta_probabilities[i] * 100:.2f}%",
            f"{result.per_class_sigma[i]:.4f}",
            "SELECTED" if i == result.class_index else "",
        ])
    elements.append(_table(rows, [3.2*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.2*cm, 2.2*cm, 2.2*cm]))
    elements.append(Spacer(1, 4))

    w = result.ensemble_weights
    elements.append(Paragraph(
        f"Fusion weights: standard {w[0]:.2f} / MC Dropout {w[1]:.2f} / TTA {w[2]:.2f}. "
        f"Weights were fitted by grid search on the validation split; "
        f"the held-out test split took no part in selecting them.",
        small_style,
    ))
    elements.append(Spacer(1, 8))

    # Geometry
    if result.geometry:
        g = result.geometry
        elements.append(Paragraph("LESION GEOMETRY (Attention U-Net)", h1_style))
        elements.append(_table(
            [
                ["Area",      f"{g['area_px']:.0f} px² ({g['area_fraction']*100:.2f}% of crop)",
                 "Circularity", f"{g['circularity']:.3f}"],
                ["Centroid",  f"({g['centroid'][0]}, {g['centroid'][1]})",
                 "Quadrant",    g["quadrant"]],
                ["Perimeter", f"{g['perimeter_px']:.1f} px", "Threshold", "0.5"],
            ],
            [2.6*cm, 7.0*cm, 2.6*cm, 4.8*cm],
        ))
        elements.append(Paragraph(
            "Geometry is computed in image-space pixels (224×224). "
            "Physical dimensions cannot be determined without calibration data.",
            small_style,
        ))
        elements.append(Spacer(1, 8))

    # Grad-CAM image
    if gradcam_overlay_png is not None:
        elements.append(Paragraph("GRAD-CAM EXPLANATION", h1_style))
        elements.append(Paragraph(
            "The heatmap highlights regions that contributed most strongly to the "
            "model's classification decision. Warmer colours indicate higher importance.",
            body_style,
        ))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(gradcam_overlay_png)
            tmp_path = tmp.name
        img_elem = RLImage(tmp_path, width=10*cm, height=10*cm, kind="proportional")
        elements.append(img_elem)
        elements.append(Spacer(1, 8))

    # Interpretation
    elements.append(Paragraph("INTERPRETATION", h1_style))
    elements.append(Paragraph(INTERPRETATION.get(prediction_key, ""), body_style))
    elements.append(Paragraph(
        f"<b>Recommendation:</b> {RECOMMENDATION.get(prediction_key, '')}",
        body_style,
    ))
    elements.append(Spacer(1, 8))

    # Model provenance
    elements.append(Paragraph("MODEL PROVENANCE", h1_style))
    elements.append(_table(
        [
            ["Dataset",       "BRISC2025 (briscdataset/brisc2025)"],
            ["Test split",    "Official BRISC2025 test folder, untouched during training"],
            ["Segmentation",  "Attention U-Net — test Dice 0.8400, IoU 0.7756"],
            ["Deployed arm",  "full_image — test accuracy 98.59%, macro F1 98.62%"],
            ["With ensemble", "test accuracy 98.79%, macro F1 98.84%, κ 0.9835"],
            ["Uncertainty",   f"MC Dropout, {result.mc_samples_used} stochastic passes (BatchNorm frozen)"],
            ["Explainability","Grad-CAM on last backbone Conv2d, same input domain as training"],
        ],
        [4.0*cm, 13.0*cm],
        header_fill=DARK,
    ))
    elements.append(Spacer(1, 10))

    # Disclaimer
    elements.append(HRFlowable(width="100%", thickness=2, color=MED, spaceAfter=5))
    elements.append(Paragraph(
        "IMPORTANT: This is an experimental research system. This report is not a medical "
        "diagnosis, has not been reviewed by a licensed clinician, and must not be used to "
        "direct patient care. Model outputs are derived from a single 2D slice and must be "
        "correlated with the full study, clinical history, and expert radiological "
        "interpretation. The authors accept no liability for decisions based on this output.",
        small_style,
    ))

    SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        leftMargin=2.0*cm, rightMargin=2.0*cm,
    ).build(elements)

    return buf.getvalue()
