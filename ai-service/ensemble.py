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
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO

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
        self.model: Optional[YOLO] = None
        self.load_error: Optional[str] = None

        if not os.path.exists(self.path):
            self.load_error = f"file not found: {self.path}"
            return
        try:
            self.model = YOLO(self.path)
        except Exception as exc:
            self.load_error = str(exc)

    @property
    def available(self) -> bool:
        return self.model is not None


def load_ensemble() -> List[EnsembleMember]:
    members = []
    for cfg in ENSEMBLE_CONFIG:
        member = EnsembleMember(cfg["file"], cfg["weight"], cfg["label_map"])
        if member.available:
            print(f"[ensemble] loaded {cfg['file']} — classes: {member.model.names}")
        else:
            print(f"[ensemble] SKIPPED {cfg['file']}: {member.load_error}")
        members.append(member)
    return members


def _predict_one(member: EnsembleMember, bgr: np.ndarray, conf: float, img_w: int, img_h: int):
    """Run one model and return (boxes_norm, scores, canonical_label_indices)
    filtered to only classes present in this project's taxonomy."""
    results = member.model.predict(bgr, conf=conf, verbose=False)[0]

    boxes, scores, labels = [], [], []
    for box in results.boxes:
        raw_name = member.model.names[int(box.cls[0])]
        canonical = member.label_map.get(raw_name)
        if canonical is None:
            continue  # class not in our taxonomy, or explicitly dropped
        if canonical not in LABEL_TO_IDX:
            continue  # safety: typo in label_map pointing to an unknown canonical name

        x0, y0, x1, y1 = box.xyxy[0].tolist()
        # WBF expects normalized [0,1] coordinates
        boxes.append([x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h])
        scores.append(float(box.conf[0]))
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
