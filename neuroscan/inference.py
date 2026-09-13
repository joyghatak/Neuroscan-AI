"""
NeuroScan AI v3.0 — Inference engine
MC Dropout, deterministic TTA, ensemble, segmentation, ROI extraction.
Extracted from NeuroScan_V3.ipynb cells 24 and 30.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from .preprocessing import (
    IMG_SIZE,
    brain_crop_from_array,
    to_tensor_imagenet,
    to_tensor_unit,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

CLASS_NAMES: List[str] = ["glioma", "meningioma", "no_tumor", "pituitary"]

RISK_BY_CLASS: Dict[str, str] = {
    "glioma": "HIGH",
    "meningioma": "MEDIUM",
    "pituitary": "MEDIUM",
    "no_tumor": "LOW",
}

UNCERTAINTY_BANDS: Tuple[float, float] = (0.05, 0.12)

# Fitted ensemble weights (from notebook grid search on validation split)
ENSEMBLE_WEIGHTS: Tuple[float, float, float] = (0.00, 0.75, 0.25)

# Default inference config
DEFAULT_MC_SAMPLES: int = 30
DEFAULT_TTA_STEPS: int = 10

# Segmentation config (matching notebook)
ROI_THRESHOLD: float = 0.5
ROI_PAD_FRAC: float = 0.15
ROI_MIN_FRAC: float = 0.10

S = IMG_SIZE

# ── Deterministic TTA bank (exactly matching notebook) ────────────────────

TTA_BANK: List[Tuple[str, callable]] = [
    ("identity",     lambda t: t),
    ("hflip",        lambda t: torch.flip(t, dims=[-1])),
    ("rot+8",        lambda t: TF.rotate(t, 8.0)),
    ("rot-8",        lambda t: TF.rotate(t, -8.0)),
    ("rot+15",       lambda t: TF.rotate(t, 15.0)),
    ("rot-15",       lambda t: TF.rotate(t, -15.0)),
    ("hflip+rot+8",  lambda t: TF.rotate(torch.flip(t, dims=[-1]), 8.0)),
    ("hflip+rot-8",  lambda t: TF.rotate(torch.flip(t, dims=[-1]), -8.0)),
    ("scale0.92",    lambda t: TF.center_crop(TF.resize(t, [int(S * 0.92)] * 2, antialias=True), [S, S])),
    ("scale1.08",    lambda t: TF.center_crop(TF.resize(t, [int(S * 1.08)] * 2, antialias=True), [S, S])),
]


# ── Helper functions ────────────────────────────────────────────────────────

def uncertainty_level(sigma: float) -> str:
    """Map MC Dropout sigma to Low / Moderate / High."""
    if sigma < UNCERTAINTY_BANDS[0]:
        return "Low"
    elif sigma < UNCERTAINTY_BANDS[1]:
        return "Moderate"
    return "High"


def enable_mc_dropout(model: nn.Module) -> int:
    """Set eval() everywhere, then re-enable Dropout layers only (BN frozen)."""
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
            n += 1
    return n


def mask_to_box(
    mask: np.ndarray,
    roi_threshold: float = ROI_THRESHOLD,
    roi_pad_frac: float = ROI_PAD_FRAC,
    roi_min_frac: float = ROI_MIN_FRAC,
    img_size: int = S,
) -> Tuple[Optional[List[int]], float]:
    """Largest connected component → padded bounding box [x1,y1,x2,y2] or None."""
    binary = (mask > roi_threshold).astype(np.uint8)
    area_frac = float(binary.mean())
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, area_frac
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    px, py = int(w * roi_pad_frac), int(h * roi_pad_frac)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(img_size, x + w + px)
    y2 = min(img_size, y + h + py)
    if (x2 - x1) < roi_min_frac * img_size or (y2 - y1) < roi_min_frac * img_size:
        return None, area_frac
    return [x1, y1, x2, y2], area_frac


def crop_to_roi(img_u8: np.ndarray, box: List[int]) -> np.ndarray:
    """Crop image to ROI box and resize back to IMG_SIZE × IMG_SIZE."""
    x1, y1, x2, y2 = box
    crop = img_u8[y1:y2, x1:x2]
    return cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Single-image analysis result."""
    # Classification
    prediction: str
    class_index: int
    confidence: float
    probabilities: np.ndarray          # (4,) ensemble
    standard_probabilities: np.ndarray # (4,)
    mc_probabilities: np.ndarray       # (4,)
    tta_probabilities: np.ndarray      # (4,)
    per_class_sigma: np.ndarray        # (4,) MC std per class
    uncertainty: float                 # mean sigma
    uncertainty_level: str
    risk: str

    # Images (uint8 RGB)
    brain_crop: np.ndarray
    segmentation_probability: np.ndarray   # (S, S) float32
    segmentation_mask: np.ndarray          # (S, S) uint8 binary
    roi_box: Optional[List[int]]
    roi_overlay: np.ndarray               # brain crop with contour+box

    # Grad-CAM
    gradcam: np.ndarray                   # (S, S) float32 heatmap
    gradcam_overlay: np.ndarray           # uint8 RGB overlay

    # Geometry
    geometry: Optional[Dict]

    # Meta
    ensemble_weights: Tuple[float, float, float] = field(default=ENSEMBLE_WEIGHTS)
    mc_samples_used: int = DEFAULT_MC_SAMPLES
    tta_steps_used: int = DEFAULT_TTA_STEPS

    def to_dict(self) -> Dict:
        return {
            "project": "NeuroScan AI v3.0",
            "prediction": self.prediction,
            "class_index": self.class_index,
            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
            "uncertainty_level": self.uncertainty_level,
            "risk_category": self.risk,
            "probabilities": {
                CLASS_NAMES[i]: float(self.probabilities[i]) for i in range(4)
            },
            "per_class_sigma": {
                CLASS_NAMES[i]: float(self.per_class_sigma[i]) for i in range(4)
            },
            "model": {
                "classifier": "EfficientNet-B0",
                "segmenter": "Attention U-Net",
                "input_size": "224x224",
                "domain": "full_image",
            },
            "inference": {
                "mc_samples": self.mc_samples_used,
                "tta_steps": self.tta_steps_used,
                "ensemble_weights": {
                    "standard": self.ensemble_weights[0],
                    "mc_dropout": self.ensemble_weights[1],
                    "tta": self.ensemble_weights[2],
                },
            },
            # 'contours' is a list of np.ndarrays — excluded for JSON serialisability
            "geometry": {
                k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v)
                for k, v in self.geometry.items()
                if k != "contours"
            } if self.geometry else None,
        }


# ── Inference engine ────────────────────────────────────────────────────────

class NeuroScanInference:
    """Encapsulates full NeuroScan AI inference pipeline."""

    def __init__(
        self,
        segmenter: nn.Module,
        classifier: nn.Module,
        device: torch.device,
        mc_samples: int = DEFAULT_MC_SAMPLES,
        tta_steps: int = DEFAULT_TTA_STEPS,
    ):
        self.segmenter = segmenter
        self.classifier = classifier
        self.device = device
        self.mc_samples = mc_samples
        self.tta_steps = tta_steps

    def _run_segmentation(self, brain_rgb: np.ndarray) -> np.ndarray:
        """Run Attention U-Net on brain crop. Returns (S, S) probability map."""
        self.segmenter.eval()
        with torch.inference_mode():
            t = to_tensor_unit(brain_rgb).unsqueeze(0).to(self.device)
            logits = self.segmenter(t)
            prob = torch.sigmoid(logits.float()).cpu().numpy()[0, 0]
        return prob.astype(np.float32)

    def _run_standard(self, x: torch.Tensor) -> np.ndarray:
        """Single deterministic forward pass. Returns (4,) softmax probabilities."""
        self.classifier.eval()
        with torch.inference_mode():
            logits = self.classifier(x)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
        return probs

    def _run_mc_dropout(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """30 stochastic passes with BatchNorm frozen. Returns (mean, std)."""
        enable_mc_dropout(self.classifier)
        draws = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.classifier(x)
                p = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
                draws.append(p)
        self.classifier.eval()
        draws_arr = np.stack(draws)
        return draws_arr.mean(axis=0), draws_arr.std(axis=0)

    def _run_tta(self, x: torch.Tensor) -> np.ndarray:
        """Deterministic TTA over 10 transforms. Returns (4,) mean probabilities."""
        self.classifier.eval()
        bank = TTA_BANK[:max(1, min(self.tta_steps, len(TTA_BANK)))]
        acc = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        with torch.no_grad():
            for _, fn in bank:
                xb = fn(x)
                logits = self.classifier(xb)
                p = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
                acc += p
        return (acc / len(bank)).astype(np.float32)

    def _compute_geometry(self, mask_prob: np.ndarray) -> Optional[Dict]:
        """Compute tumour geometry metrics from segmentation probability map."""
        binary = (mask_prob > ROI_THRESHOLD).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < 1.0:
            return None
        perim = float(cv2.arcLength(c, True))
        m = cv2.moments(c)
        cx = int(m["m10"] / m["m00"]) if m["m00"] else S // 2
        cy = int(m["m01"] / m["m00"]) if m["m00"] else S // 2
        circularity = float(4 * math.pi * area / perim ** 2) if perim > 0 else 0.0
        return {
            "area_px": area,
            "perimeter_px": perim,
            "centroid": (cx, cy),
            "circularity": circularity,
            "area_fraction": float(binary.mean()),
            "quadrant": f"{'Upper' if cy < S / 2 else 'Lower'}-{'Left' if cx < S / 2 else 'Right'}",
            "contours": contours,
        }

    def _build_roi_overlay(
        self, brain_rgb: np.ndarray, geometry: Optional[Dict]
    ) -> np.ndarray:
        """Draw contour and bounding box on brain crop."""
        overlay = brain_rgb.copy()
        if geometry is not None:
            cv2.drawContours(overlay, geometry["contours"], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(max(geometry["contours"], key=cv2.contourArea))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 165, 0), 2)
        return overlay

    def analyze(
        self,
        image_bytes: bytes,
        fast_mode: bool = False,
    ) -> AnalysisResult:
        """
        Full single-image pipeline.

        Args:
            image_bytes: Raw image file bytes (JPEG/PNG/BMP/TIFF).
            fast_mode: If True, use reduced MC (10) and TTA (5) passes.

        Returns:
            AnalysisResult with all inference outputs.
        """
        mc_samples = 10 if fast_mode else self.mc_samples
        tta_steps = 5 if fast_mode else self.tta_steps

        # 1. Decode and brain-crop
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode uploaded image.")
        brain_rgb = brain_crop_from_array(bgr, size=S)

        # 2. Segmentation (for visualization; classifier uses full brain crop)
        seg_prob = self._run_segmentation(brain_rgb)
        seg_mask = (seg_prob > ROI_THRESHOLD).astype(np.uint8)

        # 3. ROI box (for visualization only — classifier uses full_image domain)
        roi_box, _area_frac = mask_to_box(seg_prob)

        # 4. Build classifier input tensor (FULL IMAGE — matching deployment arm)
        x = to_tensor_imagenet(brain_rgb).unsqueeze(0).to(self.device)

        # 5. Standard prediction
        std_p = self._run_standard(x)

        # 6. MC Dropout
        mc_p, mc_sigma = self._run_mc_dropout(x)

        # 7. Deterministic TTA
        tta_p = self._run_tta(x)

        # 8. Ensemble (fitted weights: 0.00 std / 0.75 MC / 0.25 TTA)
        w = ENSEMBLE_WEIGHTS
        ens_p = w[0] * std_p + w[1] * mc_p + w[2] * tta_p
        cls_idx = int(ens_p.argmax())
        sigma = float(mc_sigma.mean())

        # 9. Grad-CAM
        from .explainability import GradCAM, overlay_heatmap
        with GradCAM(self.classifier) as cam:
            gradcam_heatmap, _, _ = cam(x, class_idx=cls_idx)
        gradcam_ov = overlay_heatmap(brain_rgb, gradcam_heatmap)

        # 10. Geometry
        geometry = self._compute_geometry(seg_prob)
        roi_overlay = self._build_roi_overlay(brain_rgb, geometry)

        return AnalysisResult(
            prediction=CLASS_NAMES[cls_idx],
            class_index=cls_idx,
            confidence=float(ens_p[cls_idx]),
            probabilities=ens_p.astype(np.float32),
            standard_probabilities=std_p.astype(np.float32),
            mc_probabilities=mc_p.astype(np.float32),
            tta_probabilities=tta_p.astype(np.float32),
            per_class_sigma=mc_sigma.astype(np.float32),
            uncertainty=sigma,
            uncertainty_level=uncertainty_level(sigma),
            risk=RISK_BY_CLASS[CLASS_NAMES[cls_idx]],
            brain_crop=brain_rgb,
            segmentation_probability=seg_prob,
            segmentation_mask=seg_mask,
            roi_box=roi_box,
            roi_overlay=roi_overlay,
            gradcam=gradcam_heatmap,
            gradcam_overlay=gradcam_ov,
            geometry=geometry,
            ensemble_weights=ENSEMBLE_WEIGHTS,
            mc_samples_used=mc_samples,
            tta_steps_used=tta_steps,
        )
