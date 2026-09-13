"""
NeuroScan AI v3.0 — Artifact discovery, checkpoint loading, utilities.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Artifact discovery ──────────────────────────────────────────────────────

CANDIDATE_DIRS: List[str] = [
    "neuroscan_v3_artifacts",
    "neuroscan_v3_artifacts/neuroscan_v3_artifacts",
    "artifacts",
]

REQUIRED_CHECKPOINTS: List[str] = [
    "checkpoints/segmenter.pt",
    "checkpoints/classifier_full_image.pt",
]

OPTIONAL_METRICS: List[str] = [
    "metrics/run_summary.json",
    "metrics/comparison.json",
    "metrics/segmentation.json",
]


def find_artifact_dir() -> Optional[Path]:
    """
    Search for the NeuroScan artifact directory.
    Priority:
    1. NEUROSCAN_ARTIFACT_DIR environment variable
    2. Candidate subdirectories relative to this file
    3. Candidate subdirectories relative to cwd
    """
    env_path = os.environ.get("NEUROSCAN_ARTIFACT_DIR")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        logger.warning("NEUROSCAN_ARTIFACT_DIR=%s does not exist", env_path)

    base_dirs = [Path(__file__).parent.parent, Path.cwd()]
    for base in base_dirs:
        for cand in CANDIDATE_DIRS:
            p = base / cand
            if p.exists() and (p / "checkpoints").exists():
                return p

    return None


def check_artifacts(artifact_dir: Optional[Path]) -> Dict[str, bool]:
    """Return a dict of artifact_name → exists."""
    status: Dict[str, bool] = {}
    if artifact_dir is None:
        for name in REQUIRED_CHECKPOINTS + OPTIONAL_METRICS:
            status[name] = False
        return status

    for name in REQUIRED_CHECKPOINTS:
        status[name] = (artifact_dir / name).exists()

    for name in OPTIONAL_METRICS:
        status[name] = (artifact_dir / name).exists()

    # Check figures
    figures = ["figures/arm_comparison.png", "figures/arm_curves.png",
               "figures/segmentation_curves.png", "figures/test_panels.png"]
    for name in figures:
        status[name] = (artifact_dir / name).exists()

    # Check XAI
    for cls in ["glioma", "meningioma", "no_tumor", "pituitary"]:
        name = f"xai/unified_{cls}.png"
        status[name] = (artifact_dir / name).exists()

    return status


# ── Checkpoint loading ─────────────────────────────────────────────────────

def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> nn.Module:
    """
    Load a checkpoint into model, handling both raw state_dict and wrapped formats.
    Reports missing/unexpected keys as warnings, not crashes.
    """
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)

    # Determine the actual state dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        else:
            # Assume it IS the state dict
            state = checkpoint
    else:
        state = checkpoint

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Checkpoint %s — missing keys: %s", path.name, missing[:5])
    if unexpected:
        logger.warning("Checkpoint %s — unexpected keys: %s", path.name, unexpected[:5])

    return model


# ── Metrics loading ────────────────────────────────────────────────────────

def load_json_safe(path: Path) -> Optional[Dict]:
    """Load JSON file, returning None on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def load_metrics(artifact_dir: Path) -> Dict:
    """Load all metrics JSONs, returning empty dicts for missing files."""
    metrics: Dict = {}
    metrics["run_summary"] = load_json_safe(artifact_dir / "metrics" / "run_summary.json") or {}
    metrics["comparison"] = load_json_safe(artifact_dir / "metrics" / "comparison.json") or {}
    metrics["segmentation"] = load_json_safe(artifact_dir / "metrics" / "segmentation.json") or {}
    return metrics


# ── Device detection ───────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_display_name(device: torch.device) -> str:
    """Human-readable device name for UI display."""
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            return f"NVIDIA {name}"
        except Exception:
            return "CUDA GPU"
    return "CPU"


# ── Image encoding helpers ─────────────────────────────────────────────────

import cv2
import numpy as np
from PIL import Image
import io as _io


def pil_to_bytes(img: np.ndarray, fmt: str = "PNG") -> bytes:
    """Convert uint8 RGB numpy array to PNG bytes."""
    pil = Image.fromarray(img.astype(np.uint8))
    buf = _io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


def numpy_to_png(arr: np.ndarray) -> bytes:
    """Convert a float32 [0,1] or uint8 [0,255] array to PNG bytes."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return pil_to_bytes(arr)
