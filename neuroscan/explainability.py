"""
NeuroScan AI v3.0 — Grad-CAM and visualization
Extracted from NeuroScan_V3.ipynb cells 18 and 32.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io

from .inference import CLASS_NAMES, RISK_BY_CLASS


RISK_COLOUR: dict = {"HIGH": "#E53935", "MEDIUM": "#FB8C00", "LOW": "#43A047"}


class GradCAM:
    """
    Grad-CAM against the last convolution of the classifier backbone.
    
    Fixes v2 bugs:
    1. Input tensor explicitly requires_grad=True (needed when backbone is frozen).
    2. register_full_backward_hook instead of deprecated register_backward_hook.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.layer = target_layer if target_layer is not None else model.gradcam_layer
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._handles = [
            self.layer.register_forward_hook(self._save_acts),
            self.layer.register_full_backward_hook(self._save_grads),
        ]

    def _save_acts(self, _m, _inp, out) -> None:
        self._acts = out

    def _save_grads(self, _m, _gin, gout) -> None:
        self._grads = gout[0]

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Args:
            x: (1, 3, H, W) ImageNet-normalised tensor.
            class_idx: Target class (defaults to predicted class).

        Returns:
            (heatmap HxW in [0,1], class_idx, probabilities (4,))
        """
        was_training = self.model.training
        self.model.eval()
        self._acts = self._grads = None

        with torch.enable_grad():  # survives an outer no_grad context
            inp = x.clone().detach().requires_grad_(True)  # THE FIX from v2
            logits = self.model(inp)
            probs = torch.softmax(logits, dim=1)
            idx = int(logits.detach().argmax(1).item()) if class_idx is None else int(class_idx)
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()

        if self._acts is None or self._grads is None:
            raise RuntimeError(
                "Grad-CAM captured no activations/gradients. "
                "The target layer is not on the backward path for this input."
            )

        acts = self._acts[0].detach()   # (C, h, w)
        grads = self._grads[0].detach() # (C, h, w)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * acts).sum(0))
        cam = cam - cam.min()
        denom = float(cam.max())
        cam = cam / denom if denom > 1e-8 else torch.zeros_like(cam)
        cam = F.interpolate(
            cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]

        if was_training:
            self.model.train()

        return (
            cam.detach().cpu().numpy().astype(np.float32),
            idx,
            probs.detach().cpu().numpy()[0],
        )


def overlay_heatmap(
    rgb_u8: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    """Overlay a [0,1] heatmap on an RGB image using JET colormap."""
    hm = cv2.resize(heatmap, (rgb_u8.shape[1], rgb_u8.shape[0]))
    colour = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    return cv2.addWeighted(rgb_u8, 1 - alpha, colour, alpha, 0)


def build_unified_explanation(
    brain_rgb: np.ndarray,
    seg_prob: np.ndarray,
    roi_overlay: np.ndarray,
    gradcam_overlay: np.ndarray,
    gradcam_heatmap: np.ndarray,
    prediction: str,
    confidence: float,
    uncertainty: float,
    uncertainty_lvl: str,
    probabilities: np.ndarray,
    per_class_sigma: np.ndarray,
    geometry_contours,
    S: int = 224,
) -> np.ndarray:
    """
    Build the 6-panel unified explanation figure (as in notebook cell 32).
    Returns the figure as a uint8 RGB numpy array.
    """
    risk = RISK_BY_CLASS[prediction]

    # Combined panel: green mask tint + Grad-CAM blend + contour
    tint = np.zeros_like(brain_rgb, dtype=np.float32)
    tint[..., 1] = (seg_prob > 0.5) * 255
    heat_full = cv2.resize(gradcam_heatmap, (S, S))
    combined = cv2.addWeighted(brain_rgb.astype(np.float32), 0.70, tint, 0.30, 0).astype(np.uint8)
    gradcam_full = overlay_heatmap(brain_rgb, heat_full, alpha=1.0)
    combined = cv2.addWeighted(combined, 0.65, gradcam_full, 0.35, 0)
    if geometry_contours is not None:
        cv2.drawContours(combined, geometry_contours, -1, (255, 255, 255), 1)

    fig = plt.figure(figsize=(22, 5.4))
    fig.patch.set_facecolor("#0d1b2a")
    fig.suptitle(
        f"NeuroScan AI v3.0 — Unified Explanation   |   {prediction.upper()}   "
        f"conf {confidence * 100:.1f}%   sigma {uncertainty:.3f} "
        f"({uncertainty_lvl})   risk {risk}",
        color="white", fontsize=12, fontweight="bold", y=1.03,
    )

    panels = [
        (brain_rgb,       None,  "1 · Brain crop"),
        (seg_prob,        "hot", "2 · Tumour probability (U-Net)"),
        (roi_overlay,     None,  "3 · ROI contour + box"),
        (gradcam_overlay, None,  "4 · Grad-CAM (full_image domain)"),
        (combined,        None,  "5 · Combined"),
    ]
    for i, (panel, cmap, title) in enumerate(panels):
        ax = fig.add_subplot(1, 6, i + 1)
        ax.imshow(panel, cmap=cmap, vmin=0, vmax=1 if cmap else None)
        ax.set_title(title, color="white", fontsize=9, fontweight="bold")
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor(RISK_COLOUR[risk] if i == 4 else "#4a7aaa")

    ax = fig.add_subplot(1, 6, 6)
    colours = ["#E53935", "#FB8C00", "#43A047", "#1E88E5"]
    ax.barh(
        CLASS_NAMES,
        probabilities * 100,
        xerr=per_class_sigma * 100,
        color=colours, edgecolor="white", linewidth=0.6,
    )
    ax.set_facecolor("#0d1b2a")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlim(0, 108)
    ax.set_xlabel("probability (%) ± MC sigma", color="white", fontsize=8)
    ax.set_title("6 · Class posterior", color="white", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.2, color="white")
    for sp in ax.spines.values():
        sp.set_color("#4a7aaa")

    plt.tight_layout()

    # Render to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_arr = np.frombuffer(buf.read(), dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR→RGB


def build_probability_figure(
    probabilities: np.ndarray,
    per_class_sigma: np.ndarray,
    prediction: str,
) -> np.ndarray:
    """Bar chart of class probabilities with uncertainty. Returns uint8 RGB array."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    colours = ["#E53935", "#FB8C00", "#43A047", "#1E88E5"]
    bars = ax.barh(
        CLASS_NAMES,
        probabilities * 100,
        xerr=per_class_sigma * 100,
        color=colours, edgecolor="white", linewidth=0.6,
        capsize=4,
    )
    # Highlight predicted class
    pred_idx = CLASS_NAMES.index(prediction)
    bars[pred_idx].set_edgecolor("white")
    bars[pred_idx].set_linewidth(2.5)

    ax.tick_params(colors="white", labelsize=10)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Probability (%)", color="white", fontsize=10)
    ax.set_title("Class Posterior Probabilities ± MC σ", color="white", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2, color="white")
    for sp in ax.spines.values():
        sp.set_color("#4a7aaa")

    # Value labels
    for i, (p, s) in enumerate(zip(probabilities, per_class_sigma)):
        ax.text(p * 100 + 1, i, f"{p*100:.1f}% ±{s*100:.1f}%",
                va="center", ha="left", color="white", fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)[:, :, ::-1]
