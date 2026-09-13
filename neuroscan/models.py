"""
NeuroScan AI v3.0 — Model architectures
Extracted verbatim from NeuroScan_V3.ipynb cell 14.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ── Attention U-Net ────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """(Conv → BN → ReLU) x2, with optional spatial dropout between them."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Additive attention gate, canonical formulation (Oktay et al., 2018).
    Gating signal is the COARSE decoder feature (half the skip's resolution).
    """

    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int):
        super().__init__()
        self.theta = nn.Conv2d(skip_ch, inter_ch, 1, stride=2, bias=False)
        self.phi = nn.Conv2d(gate_ch, inter_ch, 1, bias=True)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)
        self.out = nn.Sequential(nn.Conv2d(skip_ch, skip_ch, 1), nn.BatchNorm2d(skip_ch))

    def forward(self, skip: torch.Tensor, gate: torch.Tensor):
        theta = self.theta(skip)
        phi = self.phi(gate)
        if theta.shape[-2:] != phi.shape[-2:]:
            phi = F.interpolate(phi, size=theta.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.relu(theta + phi, inplace=True)))
        alpha = F.interpolate(alpha, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(skip * alpha), alpha


class AttentionUNet(nn.Module):
    """5-level encoder / 4-level decoder, attention gate on every skip."""

    def __init__(self, in_ch: int = 3, filters: Sequence[int] = (32, 64, 128, 256, 512)):
        super().__init__()
        f = list(filters)
        drops = [0.0, 0.05, 0.10, 0.15, 0.25]
        self.enc = nn.ModuleList([
            ConvBlock(in_ch if i == 0 else f[i - 1], f[i], drops[i]) for i in range(4)
        ])
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(f[3], f[4], drops[4])
        self.up = nn.ModuleList([nn.ConvTranspose2d(f[i + 1], f[i], 2, stride=2) for i in range(3, -1, -1)])
        self.att = nn.ModuleList([AttentionGate(f[i], f[i + 1], max(f[i] // 2, 1)) for i in range(3, -1, -1)])
        self.dec = nn.ModuleList([ConvBlock(f[i] * 2, f[i], drops[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(f[0], 1, 1)
        self.last_alphas: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor, keep_attention: bool = False) -> torch.Tensor:
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bridge(x)
        alphas = []
        for j in range(4):
            skip = skips[3 - j]
            attended, alpha = self.att[j](skip, x)
            x = self.up[j](x)
            if x.shape[-2:] != attended.shape[-2:]:
                x = F.interpolate(x, size=attended.shape[-2:], mode="bilinear", align_corners=False)
            x = self.dec[j](torch.cat([x, attended], dim=1))
            if keep_attention:
                alphas.append(alpha.detach())
        self.last_alphas = alphas
        return self.head(x)  # LOGITS, not sigmoid


# ── NeuroScan Classifier (EfficientNet-B0) ─────────────────────────────────

class NeuroScanClassifier(nn.Module):
    """
    EfficientNet-B0 backbone + classification head.
    Head order: Dropout → Linear → BatchNorm1d → ReLU → Dropout → Linear
    """

    def __init__(self, num_classes: int = 4, hidden: int = 256,
                 dropout: Tuple[float, float] = (0.40, 0.30),
                 pretrained: bool = False):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout[1]),
            nn.Linear(hidden, num_classes),
        )
        self.set_backbone_trainable(False)

    def set_backbone_trainable(self, flag: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: BatchNorm1d in the head requires eval() mode for batch_size=1.
        # The inference engine always calls model.eval() before forward.
        return self.head(self.backbone(x))  # LOGITS

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last Conv2d of the feature extractor — the Grad-CAM target."""
        last = None
        for m in self.backbone.features.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("No Conv2d found in the backbone")
        return last
