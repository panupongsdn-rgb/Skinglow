"""
zone_model.py — The per-zone skin classifier (shared by training and the API)
=============================================================================
One CNN looks at a zone crop (forehead, cheek, nose, under-eye, chin, or a
close-up "patch") and answers six yes/no questions at once — is there acne,
black spots, eyebags, oiliness, redness, wrinkles in this zone? (multi-label
classification, one sigmoid per class).

The zone's identity is fed in as a small learned embedding next to the image
features, because "normal" looks different per zone: fine lines under the
eyes are not the same signal as lines on the forehead.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from face_zones import CLASSES, ZONE_NAMES, class_allowed

ZONES = ZONE_NAMES + ["patch"]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def allowed_matrix() -> torch.Tensor:
    """(n_zones, n_classes) 1/0 mask of face_zones.ALLOWED_ZONES."""
    m = torch.zeros(len(ZONES), len(CLASSES))
    for j, c in enumerate(CLASSES):
        for i, z in enumerate(ZONES):
            m[i, j] = 1.0 if class_allowed(c, z) else 0.0
    return m


def _backbone(arch: str, pretrained: bool):
    import torchvision.models as tvm
    if arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        feat = m.classifier[1].in_features
        m.classifier = nn.Identity()
    elif arch == "efficientnet_b2":
        m = tvm.efficientnet_b2(weights=tvm.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
        feat = m.classifier[1].in_features
        m.classifier = nn.Identity()
    elif arch == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        feat = m.classifier[0].in_features
        m.classifier = nn.Identity()
    elif arch == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat = m.fc.in_features
        m.fc = nn.Identity()
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return m, feat


class ZoneClassifier(nn.Module):
    def __init__(self, arch: str = "efficientnet_b0", pretrained: bool = True,
                 zone_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.arch = arch
        self.backbone, feat = _backbone(arch, pretrained)
        self.zone_emb = nn.Embedding(len(ZONES), zone_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat + zone_dim, len(CLASSES)))

    def forward(self, x: torch.Tensor, zone_idx: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        return self.head(torch.cat([f, self.zone_emb(zone_idx)], dim=1))


# ---------------------------------------------------------------------------
# Inference wrapper used by ai-service/main.py
# ---------------------------------------------------------------------------
class ZonePredictor:
    """Loads best.pt written by zone_train/train_zone_classifier.py."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        self.meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        self.model = ZoneClassifier(ckpt["arch"], pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(device)
        self.device = device
        self.img_size = int(ckpt.get("img_size", 224))
        self.thresholds = np.array([ckpt["thresholds"][c] for c in CLASSES], dtype=np.float32)
        self.allowed = allowed_matrix().numpy()
        self.mean = np.array(MEAN, np.float32).reshape(1, 1, 3)
        self.std = np.array(STD, np.float32).reshape(1, 1, 3)

    def _to_tensor(self, crops_bgr: List[np.ndarray]) -> torch.Tensor:
        batch = []
        for c in crops_bgr:
            rgb = c[:, :, ::-1].astype(np.float32) / 255.0
            batch.append(((rgb - self.mean) / self.std).transpose(2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(np.stack(batch)))

    @torch.inference_mode()
    def predict(self, crops_bgr: List[np.ndarray], zone_names: List[str]) -> List[Dict[str, dict]]:
        """Return, per crop, {class: {"prob": p, "present": bool}}."""
        if not crops_bgr:
            return []
        x = self._to_tensor(crops_bgr).to(self.device)
        zi = torch.tensor([ZONES.index(z) for z in zone_names], dtype=torch.long, device=self.device)
        probs = torch.sigmoid(self.model(x, zi)).cpu().numpy()
        probs = probs * self.allowed[zi.cpu().numpy()]  # impossible zone/class pairs -> 0
        out = []
        for p in probs:
            out.append({c: {"prob": round(float(p[j]), 3), "present": bool(p[j] >= self.thresholds[j])}
                        for j, c in enumerate(CLASSES)})
        return out


def load_predictor(path: str) -> Optional[ZonePredictor]:
    try:
        return ZonePredictor(path)
    except Exception as exc:  # missing file, version mismatch...
        print(f"[zone_model] could not load {path}: {exc!r}")
        return None
