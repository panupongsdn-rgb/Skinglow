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

import os

# Render's free instance gets ~0.1 CPU, but PyTorch/OpenMP see every core of
# the host machine and start that many threads, which then fight over the
# tiny CPU quota. One thread each is the right setting for this instance.
# (Must run before numpy / cv2 / torch are imported.)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import io
import threading
from typing import List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image, ImageOps

import ensemble

cv2.setNumThreads(1)
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

app = FastAPI(title="Skinglow AI Service", version="4.0.0")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
MP_FACE_MESH = mp.solutions.face_mesh
FACE_MESH = MP_FACE_MESH.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.60,)
FACE_OVAL_INDICES = sorted({
    index
    for connection in MP_FACE_MESH.FACEMESH_FACE_OVAL
    for index in connection
})
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
def locate_face(
    bgr: np.ndarray
) -> Optional[Tuple[int, int, int, int]]:

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    return tuple(
        max(faces, key=lambda f: f[2] * f[3])
    )


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
def create_strict_face_region(bgr: np.ndarray):
    """
    Create a strict facial ROI using MediaPipe Face Mesh.

    Returns:
        masked_face
        face_mask
        face_box
    """

    image_height, image_width = bgr.shape[:2]

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )

    result = FACE_MESH.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    points = []

    for index in FACE_OVAL_INDICES:

        landmark = landmarks[index]

        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)

        x = max(
            0,
            min(x, image_width - 1)
        )

        y = max(
            0,
            min(y, image_height - 1)
        )

        points.append([x, y])

    points = np.asarray(
        points,
        dtype=np.int32
    )

    if len(points) < 3:
        return None

    face_hull = cv2.convexHull(points)

    face_mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    cv2.fillConvexPoly(
        face_mask,
        face_hull,
        255
    )

    x, y, w, h = cv2.boundingRect(
        face_hull
    )

    face_crop = bgr[
        y:y+h,
        x:x+w
    ]

    crop_mask = face_mask[
        y:y+h,
        x:x+w
    ]

    masked_face = cv2.bitwise_and(
        face_crop,
        face_crop,
        mask=crop_mask
    )

    return (
        masked_face,
        crop_mask,
        (x, y, w, h)
    )

def calculate_mask_overlap(
    mask: np.ndarray,
    box: List[int]
) -> float:

    image_height, image_width = mask.shape

    x1, y1, x2, y2 = box

    x1 = max(
        0,
        min(x1, image_width - 1)
    )

    y1 = max(
        0,
        min(y1, image_height - 1)
    )

    x2 = max(
        x1 + 1,
        min(x2, image_width)
    )

    y2 = max(
        y1 + 1,
        min(y2, image_height)
    )

    box_mask = mask[
        y1:y2,
        x1:x2
    ]

    box_area = (
        (x2 - x1) *
        (y2 - y1)
    )

    if box_area <= 0:
        return 0.0

    face_pixels = cv2.countNonZero(
        box_mask
    )

    return face_pixels / box_area

def analyze_face_ml(
    bgr: np.ndarray
) -> AnalyzeResponse:

    # ==========================================================
    # 1. Detect actual face
    # ==========================================================

    face_region = create_strict_face_region(
        bgr
    )

    if face_region is None:

        return AnalyzeResponse(
            skin_score=100.0,
            detections=[],
            model_type=MODEL_TYPE,
            face_detected=False,
            skin_status="no_face_detected",
        )

    (
        masked_face,
        face_mask,
        face_box
    ) = face_region

    face_x, face_y, face_w, face_h = face_box


    # ==========================================================
    # 2. Run YOLO ONLY on face ROI
    # ==========================================================

    raw_detections = ensemble.ensemble_predict(
        masked_face,
        ENSEMBLE_MEMBERS,
        conf=CLEAR_CONFIDENCE_THRESHOLD,
    )


    detections = []


    # ==========================================================
    # 3. Strictly validate every detection
    # ==========================================================

    for raw in raw_detections:

        x1, y1, x2, y2 = raw["box"]


        # ------------------------------------------------------
        # Clamp coordinates
        # ------------------------------------------------------

        x1 = max(
            0,
            min(x1, face_w - 1)
        )

        y1 = max(
            0,
            min(y1, face_h - 1)
        )

        x2 = max(
            x1 + 1,
            min(x2, face_w)
        )

        y2 = max(
            y1 + 1,
            min(y2, face_h)
        )


        local_box = [
            x1,
            y1,
            x2,
            y2
        ]


        # ------------------------------------------------------
        # 4. Check CENTER
        # ------------------------------------------------------

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )


        if (
            center_x < 0
            or center_y < 0
            or center_x >= face_mask.shape[1]
            or center_y >= face_mask.shape[0]
        ):
            continue


        if face_mask[
            center_y,
            center_x
        ] == 0:
            continue


        # ------------------------------------------------------
        # 5. Check FACE MASK OVERLAP
        # ------------------------------------------------------

        overlap = calculate_mask_overlap(
            face_mask,
            local_box
        )


        # ต้องอยู่ในหน้าอย่างน้อย 90%
        if overlap < 0.90:
            continue


        # ------------------------------------------------------
        # 6. Convert back to original image coordinates
        # ------------------------------------------------------

        global_box = [
            int(x1 + face_x),
            int(y1 + face_y),
            int(x2 + face_x),
            int(y2 + face_y)
        ]


        detections.append(
            Detection(
                box=global_box,
                label=raw["label"],
                confidence=raw["confidence"]
            )
        )


    # ==========================================================
    # 7. Sort detections
    # ==========================================================

    detections = sorted(
        detections,
        key=lambda d: d.confidence,
        reverse=True
    )[:12]


    # ==========================================================
    # 8. Skin score
    # ==========================================================

    penalty = sum(
        d.confidence
        for d in detections
    ) * 6

    skin_score = round(
        max(
            0.0,
            100 - penalty
        ),
        1
    )


    status = determine_skin_status(
        True,
        detections
    )


    if status == "clear":

        skin_score = max(
            skin_score,
            90.0
        )


    return AnalyzeResponse(
        skin_score=skin_score,
        detections=detections,
        model_type=MODEL_TYPE,
        face_detected=True,
        skin_status=status,
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


# Phone/DSLR photos routinely come in at 3000-4000px on a side, which is far
# more resolution than face-analysis needs and is a real OOM risk on a
# resource-capped host (e.g. Render free tier: 0.1 CPU / 512MB RAM shared
# with the loaded model weights). Downscale before any processing; detection
# boxes are rescaled back to the ORIGINAL image's coordinate space before
# returning, so the caller (PHP, drawing on the full-res original with GD)
# never has to know this happened.
MAX_PROCESSING_DIMENSION = 1280


def resize_for_processing(pil_img: Image.Image):
    """Returns (resized_image, scale_factor). scale_factor is how much the
    image was shrunk (< 1.0 if resized, 1.0 if already small enough) — divide
    detection box coordinates by this to map them back to the original size."""
    width, height = pil_img.size
    longest_side = max(width, height)
    if longest_side <= MAX_PROCESSING_DIMENSION:
        return pil_img, 1.0

    scale = MAX_PROCESSING_DIMENSION / longest_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return pil_img.resize(new_size, Image.LANCZOS), scale


# One analysis at a time: the MediaPipe FaceMesh object is shared and not
# thread-safe, and with ~0.1 CPU running two at once would only make both slower.
_ANALYSIS_LOCK = threading.Lock()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type.")

    raw_bytes = await file.read()

    # The analysis is slow, blocking CPU work. Running it directly inside this
    # async function froze the whole server, so Render's /health check (5 s
    # limit) went unanswered and Render killed the instance mid-analysis
    # ("Instance failed: HTTP health check failed"). In a worker thread, the
    # event loop stays free to answer /health while the analysis runs.
    return await run_in_threadpool(_run_analysis, raw_bytes)


def _run_analysis(raw_bytes: bytes) -> AnalyzeResponse:
    with _ANALYSIS_LOCK:
        return _analyze_bytes(raw_bytes)


def _analyze_bytes(raw_bytes: bytes) -> AnalyzeResponse:
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)  # fix sideways/upside-down phone photos
        pil_img = pil_img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    processing_img, scale = resize_for_processing(pil_img)
    bgr = cv2.cvtColor(np.array(processing_img), cv2.COLOR_RGB2BGR)

    try:
        if _available_count > 0:
            result = analyze_face_ml(bgr)
        else:
            result = analyze_face(bgr)
    except Exception as exc:
        # if the trained model(s) error on a specific image, fall back to
        # the heuristic rather than failing the request outright — but LOG
        # it: a silent fallback previously made the heuristic's output look
        # like the trained model's (see Render logs for this line).
        import traceback
        print(f"[analyze] trained-model path FAILED, falling back to heuristic: {exc!r}")
        traceback.print_exc()
        if _available_count > 0:
            try:
                result = analyze_face(bgr)
            except Exception as fallback_exc:
                raise HTTPException(status_code=422, detail=f"Analysis failed: {fallback_exc}")
        else:
            raise HTTPException(status_code=422, detail=f"Analysis failed: {exc}")

    if scale != 1.0:
        # map detection boxes from the downscaled processing image back to
        # the original image's coordinate space
        for detection in result.detections:
            detection.box = [round(v / scale) for v in detection.box]

    return result
