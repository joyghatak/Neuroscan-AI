"""
NeuroScan AI v3.0 — Image preprocessing
Extracted verbatim from NeuroScan_V3.ipynb cell 10.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
import cv2
import torch

IMG_SIZE: int = 224

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def brain_bbox(bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """Otsu + largest external contour → (x, y, w, h) in original coords."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.erode(thr, None, iterations=2)
    thr = cv2.dilate(thr, None, iterations=2)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if cw > 20 and ch > 20:
            return x, y, cw, ch
    return 0, 0, w, h


def brain_crop_from_array(
    bgr: np.ndarray,
    return_bbox: bool = False,
    size: int = IMG_SIZE,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Crop brain region from a BGR numpy array, resize to size×size. Returns uint8 RGB."""
    if bgr is None or bgr.size == 0:
        raise IOError("Empty image array supplied to brain_crop_from_array")
    x, y, w, h = brain_bbox(bgr)
    crop = bgr[y:y + h, x:x + w]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    out = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return (out, (x, y, w, h)) if return_bbox else out


def brain_crop(
    path: str,
    return_bbox: bool = False,
    size: int = IMG_SIZE,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Read an MRI file, crop to brain, resize to size×size. Returns uint8 RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise IOError(f"Unreadable image: {path}")
    return brain_crop_from_array(bgr, return_bbox=return_bbox, size=size)


def to_tensor_imagenet(rgb_u8: np.ndarray) -> torch.Tensor:
    """Convert uint8 RGB HWC → float32 CHW ImageNet-normalised tensor."""
    t = torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1).float().div_(255.0)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def to_tensor_unit(rgb_u8: np.ndarray) -> torch.Tensor:
    """Convert uint8 RGB HWC → float32 CHW tensor in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(rgb_u8)).permute(2, 0, 1).float().div_(255.0)
