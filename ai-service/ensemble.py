"""
ensemble.py — Combine predictions from multiple YOLOv8 models trained on
different datasets, using Weighted Boxes Fusion (WBF).

WHY WBF AND NOT SIMPLE NMS OR WEIGHT-AVERAGING:
  - Averaging model *weights* (state_dict averaging / "model soup") only
    makes sense when models share the same architecture AND were fine-tuned
    from closely related starting points. Models trained independently on
    different datasets can end up in very different regions of weight
    space — averaging them directly tends to produce a WORSE model than
    either input, not a better one. Not used here.
  - Plain NMS across combined detections just picks one box per cluster and
    throws the rest away. WBF instead produces a fused box that's a
    confidence-weighted average of all overlapping boxes — it actually
    uses the agreement between models rather than discarding it.

HOW TO USE:
  1. Put your trained .pt files in ai-service/models/, one per dataset,
     e.g.:
       models/skin_analysis_farmasi.pt   (FarmasiSkinCare Skin_Analysis)
       models/skin_problem_clean3.pt     (Skin-Problem-Detection-Relabel-Clean3)
       models/face_skin_condition.pt     (face_skin_condition)
  2. Edit ENSEMBLE_CONFIG below: for each file, list the LABEL_MAP that
     translates *that model's* class names to this project's canonical
     6 labels (acne, black_spot, eyebag, oiliness, redness, wrinkle).
     Every model almost certainly uses different exact class names/casing
     — inspect with `YOLO('models/yourfile.pt').names` to see what to map.
  3. Classes that don't correspond to anything in our taxonomy should map
     to None and get dropped (e.g. if a source dataset has a "pore" class
     we don't track).
  4. Call `ensemble_predict(bgr_image)` instead of a single model's
     `.predict()`.

HONEST LIMITATIONS:
  - This costs roughly N x the inference time and memory of N models. On
    a free-tier CPU host (e.g. Render free web service), running 3 YOLOv8
    models per request will be noticeably slower than one — factor this
    into your cold-start/UX expectations before choosing this over
    training one model on a merged dataset (see train/README.md).
  - WBF helps most when models genuinely disagree in informative ways
    (e.g. one is better at redness, another at wrinkle). If all 3 models
    are trained on very similar/overlapping data, gains may be small for
    the cost.
"""

import os
from typing import Dict, List, Optional

import numpy as np
from wbf import weighted_boxes_fusion  # vendored, see wbf.py

# A member runs from <name>.onnx (onnxruntime, no PyTorch — fits a 512 MB host)
# when that file exists next to the .pt; otherwise from the .pt via ultralytics.
# Set SKINGLOW_RUNTIME=torch to force the .pt files.
RUNTIME = os.getenv("SKINGLOW_RUNTIME", "auto").lower()

CANONICAL_LABELS = ["acne", "black_spot", "eyebag", "oiliness", "redness", "wrinkle"]
LABEL_TO_IDX = {name: i for i, name in enumerate(CANONICAL_LABELS)}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ----------------------------------------------------------------------
# EDIT THIS: one entry per model file you drop into models/.
# `label_map` keys are that model's own class names (from YOLO(...).names)
# mapped to one of CANONICAL_LABELS, or None to drop that class entirely.
# `weight` lets you trust one model more than another in the fusion (1.0 = equal).
# ----------------------------------------------------------------------
ENSEMBLE_CONFIG = [
    {
        "file": "skin_analysis_farmasi.pt",
        "weight": 1.0,
        "label_map": {
            "Acne": "acne",
            "Black Spot": "black_spot",
            "Eyebag": "eyebag",
            "Oilness": "oiliness",
            "Redness": "redness",
            "Wrinkle": "wrinkle",
        },
    },
    # {
    #     "file": "skin_problem_clean3.pt",
    #     "weight": 1.0,
    #     "label_map": {
    #         "Acne": "acne",
    #         "Blackheads": "black_spot",     # <- confirm actual names via model.names before training finishes
    #         "Dark-Spots": "black_spot",
    #         "Dry-Skin": None,               # not in our taxonomy -> dropped
    #         "Enlarged-Pores": None,
    #     },
    # },
    # {
    #     "file": "face_skin_condition.pt",
    #     "weight": 1.0,
    #     "label_map": {
    #         # fill in after inspecting this model's .names
    #     },
    # },
]


class EnsembleMember:
    def __init__(self, file: str, weight: float, label_map: Dict[str, Optional[str]]):
        self.path = os.path.join(MODELS_DIR, file)
        self.weight = weight
        self.label_map = label_map
        self.model = None
        self.runtime: Optional[str] = None
        self.load_error: Optional[str] = None

        onnx_path = os.path.splitext(self.path)[0] + ".onnx"
        if RUNTIME != "torch" and os.path.exists(onnx_path):
            try:
                from onnx_models import OnnxYolo
                self.model, self.runtime, self.path = OnnxYolo(onnx_path), "onnxruntime", onnx_path
                return
            except Exception as exc:
                self.load_error = f"onnx: {exc}"
        if not os.path.exists(self.path):
            self.load_error = self.load_error or f"file not found: {self.path}"
            return
        try:
            from ultralytics import YOLO
            self.model, self.runtime, self.load_error = YOLO(self.path), "torch", None
        except Exception as exc:
            self.load_error = str(exc)

    def detect(self, bgr: np.ndarray, conf: float):
        """(xyxy pixel boxes, confidences, class ids) from either runtime."""
        if self.runtime == "onnxruntime":
            return self.model.detect(bgr, conf)
        r = self.model.predict(bgr, conf=conf, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)

    @property
    def available(self) -> bool:
        return self.model is not None


def load_ensemble() -> List[EnsembleMember]:
    members = []
    for cfg in ENSEMBLE_CONFIG:
        member = EnsembleMember(cfg["file"], cfg["weight"], cfg["label_map"])
        if member.available:
            print(f"[ensemble] loaded {os.path.basename(member.path)} ({member.runtime}) — classes: {member.model.names}")
            unmapped = [n for n in member.model.names.values() if resolve_label(member, n) is None]
            if unmapped:
                print(f"[ensemble] WARNING {cfg['file']}: these classes will be IGNORED "
                      f"(no mapping to {CANONICAL_LABELS}): {unmapped}")
        else:
            print(f"[ensemble] SKIPPED {cfg['file']}: {member.load_error}")
        members.append(member)
    return members


# Spelling variants seen across datasets/retrains, mapped to canonical labels.
# e.g. the original FarmasiSkinCare export used "Oilness"/"Black Spot", while a
# model retrained later emits "oiliness"/"black_spot".
NAME_ALIASES = {
    "oilness": "oiliness",
    "oily": "oiliness",
    "oily_skin": "oiliness",
    "blackspot": "black_spot",
    "dark_spot": "black_spot",
    "dark_spots": "black_spot",
    "eye_bag": "eyebag",
    "eyebags": "eyebag",
    "skin_redness": "redness",
    "wrinkles": "wrinkle",
}


def normalize_label(raw_name: str):
    """'Black Spot' / 'black-spot' / 'BLACK_SPOT' -> 'black_spot'; returns a
    canonical label or None if it isn't part of this project's taxonomy."""
    key = str(raw_name).strip().lower().replace(" ", "_").replace("-", "_")
    key = NAME_ALIASES.get(key, key)
    return key if key in LABEL_TO_IDX else None


def resolve_label(member: "EnsembleMember", raw_name: str):
    """Explicit label_map entry wins (including an explicit None = drop);
    otherwise fall back to normalized matching."""
    if raw_name in member.label_map:
        return member.label_map[raw_name]
    return normalize_label(raw_name)


def _predict_one(member: EnsembleMember, bgr: np.ndarray, conf: float, img_w: int, img_h: int):
    """Run one model and return (boxes_norm, scores, canonical_label_indices)
    filtered to only classes present in this project's taxonomy."""
    xyxy, confs, classes = member.detect(bgr, conf)

    boxes, scores, labels = [], [], []
    for (x0, y0, x1, y1), score, cls in zip(xyxy.tolist(), confs.tolist(), classes.tolist()):
        raw_name = member.model.names[int(cls)]
        canonical = resolve_label(member, raw_name)
        if canonical is None:
            continue  # class not in our taxonomy, or explicitly dropped
        if canonical not in LABEL_TO_IDX:
            continue  # safety: typo in label_map pointing to an unknown canonical name

        # WBF expects normalized [0,1] coordinates
        boxes.append([x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h])
        scores.append(float(score))
        labels.append(LABEL_TO_IDX[canonical])

    return boxes, scores, labels


def ensemble_predict(bgr: np.ndarray, members: List[EnsembleMember], conf: float = 0.25,
                      wbf_iou_thr: float = 0.5, skip_box_thr: float = 0.0):
    """Runs every available model in `members` and fuses their detections
    with Weighted Boxes Fusion. Returns a list of dicts matching this
    project's Detection schema (box in pixel xyxy, label, confidence)."""
    img_h, img_w = bgr.shape[:2]

    all_boxes, all_scores, all_labels, weights = [], [], [], []
    for member in members:
        if not member.available:
            continue
        boxes, scores, labels = _predict_one(member, bgr, conf, img_w, img_h)
        if not boxes:
            continue
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        weights.append(member.weight)

    if not all_boxes:
        return []

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=weights, iou_thr=wbf_iou_thr, skip_box_thr=skip_box_thr,
    )

    detections = []
    for box, score, label_idx in zip(fused_boxes, fused_scores, fused_labels):
        x0, y0, x1, y1 = box
        detections.append({
            "box": [int(x0 * img_w), int(y0 * img_h), int(x1 * img_w), int(y1 * img_h)],
            "label": CANONICAL_LABELS[int(label_idx)],
            "confidence": round(float(score), 2),
        })

    return sorted(detections, key=lambda d: d["confidence"], reverse=True)
