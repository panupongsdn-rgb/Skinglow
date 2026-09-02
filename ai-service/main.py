"""
Skinglow AI Service — v4
--------------------------
Now supports 1..N trained YOLOv8 models via ai-service/ensemble.py, fused
with Weighted Boxes Fusion when more than one is available. Falls back
automatically to the v2 heuristic CV pipeline (OpenCV color thresholding +
contour detection) if zero models are present or all fail to load — so the
service never hard-crashes just because a model file didn't make it into
a deployment.

Drop trained .pt files into ai-service/models/ and configure their label
mappings in ensemble.py's ENSEMBLE_CONFIG. See ensemble.py's docstring for
the full explanation of why WBF (not weight-averaging) is used to combine
multiple models trained on different datasets.

Class taxonomy (all paths use the same 6 labels):
acne, black_spot, eyebag, oiliness, redness, wrinkle
See ai-service/train/data.yaml.

IMPORTANT — accuracy notes:
  See ai-service/train/README.md for training/evaluation history. This
  service reflects whatever models actually exist in models/ at deploy
  time — check the /health endpoint's model_type and model_count fields
  to see what's actually active in a given deployment.

  The heuristic fallback (`analyze_face()`) is rule-based, not learned,
  and is kept only as a safety net for when zero trained models are present.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image, ImageOps

import ensemble

app = FastAPI(title="Skinglow AI Service", version="4.0.0")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ---------------------------------------------------------------------
# Load whatever trained models are present. Scales automatically:
#   0 models available -> heuristic_cv_v2
#   1 model available  -> yolov8_trained (still goes through ensemble.py,
#                         WBF with a single input is a no-op passthrough —
#                         verified in testing)
#   2+ models available -> yolov8_ensemble (WBF-fused)
# ---------------------------------------------------------------------
ENSEMBLE_MEMBERS = ensemble.load_ensemble()
_available_count = sum(1 for m in ENSEMBLE_MEMBERS if m.available)

if _available_count == 0:
    MODEL_TYPE = "heuristic_cv_v2"
elif _available_count == 1:
    MODEL_TYPE = "yolov8_trained"
else:
    MODEL_TYPE = "yolov8_ensemble"

print(f"[startup] {_available_count} model(s) loaded -> model_type={MODEL_TYPE}")


class Detection(BaseModel):
    box: List[int]          # [x_min, y_min, x_max, y_max]
    label: str
    confidence: float


class AnalyzeResponse(BaseModel):
    skin_score: float
    detections: List[Detection]
    model_type: str  # "yolov8_trained" or "heuristic_cv_v2" — tells the frontend/PHP which pipeline produced this
    face_detected: bool = True   # False if no face was located at all (image quality/framing issue)
    skin_status: str = "issues_found"  # "clear" | "issues_found" | "no_face_detected"


# Confidence threshold below which a detection isn't trusted enough to
# count toward "issues found" — same value used at inference time in
# YOLO_MODEL.predict(conf=...) below, kept as one constant so the two
# stay in sync.
CLEAR_CONFIDENCE_THRESHOLD = 0.35


def determine_skin_status(face_found: bool, detections: List[Detection]) -> str:
    """Implements the 'Clear' decision from the architecture discussion:
    Clear is NOT a YOLO class — it's inferred at the application layer from
    (a) a face actually being present and (b) no detection clearing the
    confidence bar. See ai-service/train/README.md 'Clear class' section
    for the full reasoning."""
    if not face_found:
        return "no_face_detected"
    if not any(d.confidence >= CLEAR_CONFIDENCE_THRESHOLD for d in detections):
        return "clear"
    return "issues_found"


def face_present(bgr: np.ndarray) -> bool:
    """Honest face-presence check (unlike locate_face, which silently falls
    back to a centered region and never reports 'no face found' to callers).
    Used only to set the face_detected/skin_status fields in the response."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return len(faces) > 0


# ----------------------------------------------------------------------
# Face localization
# ----------------------------------------------------------------------
def locate_face(bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """Returns (x, y, w, h) of the largest detected face, or a centered
    fallback region if no face is found (e.g. a close-up crop)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        h, w = bgr.shape[:2]
        fw, fh = int(w * 0.7), int(h * 0.8)
        return ((w - fw) // 2, (h - fh) // 2, fw, fh)

    return tuple(max(faces, key=lambda f: f[2] * f[3]))


def locate_eyes(gray_face: np.ndarray) -> List[Tuple[int, int, int, int]]:
    eyes = EYE_CASCADE.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=6, minSize=(15, 15))
    return list(eyes)


# ----------------------------------------------------------------------
# Blob helpers
# ----------------------------------------------------------------------
def _find_blobs(mask: np.ndarray, min_area: float, max_area: float):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        blobs.append((x, y, w, h, area))
    return blobs


def _in_any(box, regions) -> bool:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    for (rx, ry, rw, rh) in regions:
        if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
            return True
    return False


# ----------------------------------------------------------------------
# Core analysis
# ----------------------------------------------------------------------
def analyze_face(bgr: np.ndarray) -> AnalyzeResponse:
    fx, fy, fw, fh = locate_face(bgr)
    face = bgr[fy:fy + fh, fx:fx + fw]
    if face.size == 0:
        raise ValueError("Empty face region after crop.")

    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    eyes = locate_eyes(gray_face)  # (x,y,w,h) relative to face crop, used as exclusion zones

    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    l_chan, a_chan, b_chan = cv2.split(lab)

    face_area = fw * fh
    min_area = max(6, face_area * 0.00025)
    max_area = face_area * 0.02

    detections: List[Detection] = []

    # ---- Redness / acne: high "a" channel (red-green axis in LAB) ----
    a_norm = cv2.normalize(a_chan, None, 0, 255, cv2.NORM_MINMAX)
    _, red_mask = cv2.threshold(a_norm, 160, 255, cv2.THRESH_BINARY)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    for (x, y, w, blob_h, area) in _find_blobs(red_mask, min_area, max_area):
        if _in_any((x, y, w, blob_h), eyes):
            continue
        aspect = w / max(blob_h, 1)
        intensity = float(np.mean(a_norm[y:y + blob_h, x:x + w])) / 255.0
        label = "acne" if (0.6 < aspect < 1.6 and area < face_area * 0.004) else "redness"
        confidence = round(min(0.95, 0.5 + intensity * 0.5), 2)
        detections.append(Detection(
            box=[fx + x, fy + y, fx + x + w, fy + y + blob_h],
            label=label, confidence=confidence,
        ))

    # ---- Dark spots: locally low luminance vs. surrounding skin ----
    blur_l = cv2.GaussianBlur(l_chan, (25, 25), 0)
    diff = cv2.subtract(blur_l, l_chan)
    _, dark_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    for (x, y, w, blob_h, area) in _find_blobs(dark_mask, min_area, max_area):
        if _in_any((x, y, w, blob_h), eyes):
            continue
        intensity = float(np.mean(diff[y:y + blob_h, x:x + w])) / 255.0
        confidence = round(min(0.92, 0.5 + intensity * 1.5), 2)
        detections.append(Detection(
            box=[fx + x, fy + y, fx + x + w, fy + y + blob_h],
            label="black_spot", confidence=confidence,
        ))

    # ---- Oiliness: specular highlights (very bright, low saturation) ----
    _, bright_mask = cv2.threshold(v, 235, 255, cv2.THRESH_BINARY)
    low_sat = cv2.inRange(s, 0, 60)
    shine_mask = cv2.bitwise_and(bright_mask, low_sat)
    shine_mask = cv2.morphologyEx(shine_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    shine_ratio = float(np.count_nonzero(shine_mask)) / face_area
    if shine_ratio > 0.004:
        tz_x, tz_y, tz_w, tz_h = int(fw * 0.3), int(fh * 0.15), int(fw * 0.4), int(fh * 0.55)
        confidence = round(min(0.9, 0.5 + shine_ratio * 20), 2)
        detections.append(Detection(
            box=[fx + tz_x, fy + tz_y, fx + tz_x + tz_w, fy + tz_y + tz_h],
            label="oiliness", confidence=confidence,
        ))

    # ---- Wrinkle proxy: edge density in forehead strip ----
    edges = cv2.Canny(gray_face, 40, 120)
    fh_x, fh_y, fh_w, fh_h = int(fw * 0.15), int(fh * 0.05), int(fw * 0.7), int(fh * 0.18)
    roi = edges[fh_y:fh_y + fh_h, fh_x:fh_x + fh_w]
    if roi.size > 0:
        density = float(np.count_nonzero(roi)) / roi.size
        if density > 0.09:
            confidence = round(min(0.85, 0.4 + density * 3), 2)
            detections.append(Detection(
                box=[fx + fh_x, fy + fh_y, fx + fh_x + fh_w, fy + fh_y + fh_h],
                label="wrinkle", confidence=confidence,
            ))

    # ---- Eyebag: localized darkness/puffiness in the region just below each eye ----
    for (ex, ey, ew, eh) in eyes:
        under_y0 = ey + eh
        under_h = int(eh * 0.7)
        under_x0 = max(0, ex - int(ew * 0.05))
        under_w = int(ew * 1.1)
        under_y1 = min(fh, under_y0 + under_h)
        under_x1 = min(fw, under_x0 + under_w)
        if under_y1 <= under_y0 or under_x1 <= under_x0:
            continue
        region_diff = diff[under_y0:under_y1, under_x0:under_x1]
        if region_diff.size == 0:
            continue
        avg_dark = float(np.mean(region_diff))
        if avg_dark > 6:
            confidence = round(min(0.85, 0.4 + avg_dark / 40), 2)
            detections.append(Detection(
                box=[fx + under_x0, fy + under_y0, fx + under_x1, fy + under_y1],
                label="eyebag", confidence=confidence,
            ))

    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[:12]

    penalty = sum(d.confidence for d in detections) * 6
    skin_score = round(max(0.0, 100 - penalty), 1)

    found = face_present(bgr)
    status = determine_skin_status(found, detections)
    if status == "clear":
        skin_score = max(skin_score, 90.0)  # a clear result shouldn't show a middling score

    return AnalyzeResponse(
        skin_score=skin_score, detections=detections, model_type="heuristic_cv_v2",
        face_detected=found, skin_status=status,
    )


def analyze_face_ml(bgr: np.ndarray) -> AnalyzeResponse:
    """Run whatever trained model(s) are loaded, via ensemble.py. With a
    single model this is a passthrough (WBF over one input is a no-op);
    with 2+ models it fuses their outputs with Weighted Boxes Fusion."""
    raw_detections = ensemble.ensemble_predict(bgr, ENSEMBLE_MEMBERS, conf=CLEAR_CONFIDENCE_THRESHOLD)

    detections = [
        Detection(box=d["box"], label=d["label"], confidence=d["confidence"])
        for d in raw_detections
    ][:12]

    penalty = sum(d.confidence for d in detections) * 6
    skin_score = round(max(0.0, 100 - penalty), 1)

    found = face_present(bgr)
    status = determine_skin_status(found, detections)
    if status == "clear":
        skin_score = max(skin_score, 90.0)

    return AnalyzeResponse(
        skin_score=skin_score, detections=detections, model_type=MODEL_TYPE,
        face_detected=found, skin_status=status,
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "model_count": _available_count,
        "models": [
            {"file": cfg["file"], "loaded": m.available, "error": m.load_error}
            for cfg, m in zip(ensemble.ENSEMBLE_CONFIG, ENSEMBLE_MEMBERS)
        ],
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type.")

    raw_bytes = await file.read()

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)  # fix sideways/upside-down phone photos
        pil_img = pil_img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    try:
        if _available_count > 0:
            result = analyze_face_ml(bgr)
        else:
            result = analyze_face(bgr)
    except Exception as exc:
        # if the trained model(s) error on a specific image, fall back to
        # the heuristic rather than failing the request outright
        if _available_count > 0:
            try:
                result = analyze_face(bgr)
            except Exception as fallback_exc:
                raise HTTPException(status_code=422, detail=f"Analysis failed: {fallback_exc}")
        else:
            raise HTTPException(status_code=422, detail=f"Analysis failed: {exc}")

    return result
