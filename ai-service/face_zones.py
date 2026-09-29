"""
face_zones.py — Split a face into the 5 Skinglow analysis zones
================================================================
Zones (6 regions, 5 zone types — cheeks are reported left/right):

    forehead      หน้าผาก
    left_cheek    แก้มซ้าย   (the subject's left = image right on a selfie)
    right_cheek   แก้มขวา
    nose          จมูก
    under_eye     ใต้ตา (both eyes, one zone)
    chin          คาง

Each zone is a polygon built from MediaPipe FaceMesh landmark indices, so it
follows the face's pose and shape instead of fixed rectangles. Used by both
the dataset builder (training) and main.py (inference), so the zones seen at
training time are exactly the zones seen in production.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ZONE_NAMES = ["forehead", "left_cheek", "right_cheek", "nose", "under_eye", "chin"]

# Skin conditions (same order as the YOLO data.yaml) and where each can
# plausibly appear. "patch" = a close-up photo with no whole face, allows all.
# Outside these zones a label is ignored in training and forced to "absent"
# by the API.
CLASSES = ["acne", "black_spot", "eyebag", "oiliness", "redness", "wrinkle"]
CLASS_TH = {
    "acne": "สิว", "black_spot": "จุดด่างดำ", "eyebag": "ถุงใต้ตา",
    "oiliness": "ความมัน", "redness": "รอยแดง", "wrinkle": "ริ้วรอย",
}
ALLOWED_ZONES = {
    "acne":       {"forehead", "left_cheek", "right_cheek", "nose", "chin"},
    "black_spot": set(ZONE_NAMES),
    "eyebag":     {"under_eye"},
    "oiliness":   {"forehead", "left_cheek", "right_cheek", "nose", "chin"},
    "redness":    {"forehead", "left_cheek", "right_cheek", "nose", "chin"},
    "wrinkle":    {"forehead", "left_cheek", "right_cheek", "under_eye", "chin"},
}


def class_allowed(cls: str, zone: str) -> bool:
    return zone == "patch" or zone in ALLOWED_ZONES[cls]
ZONE_TH = {
    "forehead": "หน้าผาก",
    "left_cheek": "แก้มซ้าย",
    "right_cheek": "แก้มขวา",
    "nose": "จมูก",
    "under_eye": "ใต้ตา",
    "chin": "คาง",
}

# MediaPipe FaceMesh (468-point) indices. "L" = image-left side of the mesh
# (the subject's RIGHT), "R" = image-right side (subject's LEFT). Every list
# is an ordered polygon outline; each has an exact mirror on the other side.
_POLYS: Dict[str, List[List[int]]] = {
    "forehead": [[54, 103, 67, 109, 10, 338, 297, 332, 284,
                  300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70]],
    "nose": [[168, 122, 196, 198, 209, 129, 98, 97, 2,
              326, 327, 358, 429, 420, 419, 351]],
    "under_eye": [
        [33, 7, 163, 144, 145, 153, 154, 155, 133, 243, 121, 120, 119, 118, 117, 111, 35, 226],
        [263, 249, 390, 373, 374, 380, 381, 382, 362, 463, 350, 349, 348, 347, 346, 340, 265, 446],
    ],
    "cheek_L": [[116, 117, 118, 119, 120, 121, 47, 142, 203, 206, 216, 212, 214, 138, 215, 177, 137, 227, 34, 143]],
    "cheek_R": [[345, 346, 347, 348, 349, 350, 277, 371, 423, 426, 436, 432, 434, 367, 435, 401, 366, 447, 264, 372]],
    "chin": [[43, 106, 182, 83, 18, 313, 406, 335, 273, 422, 430, 365, 379, 378, 400,
              377, 152, 148, 176, 149, 150, 136, 210, 202]],
}

_mp_face_mesh = None


def _get_mesh():
    """Lazily create one FaceMesh (not thread-safe: callers serialise)."""
    global _mp_face_mesh
    if _mp_face_mesh is None:
        _mp_face_mesh = make_face_mesh()
    return _mp_face_mesh


def make_face_mesh(min_detection_confidence: float = 0.3):
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=min_detection_confidence,
    )


@dataclass
class Zone:
    name: str
    mask: np.ndarray                 # uint8 HxW (full image), 255 inside zone
    box: Tuple[int, int, int, int]   # x1, y1, x2, y2 (full image coords)
    area: int                        # pixel count inside the mask

    @property
    def name_th(self) -> str:
        return ZONE_TH[self.name]


def _landmarks_once(bgr: np.ndarray, mesh) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    res = mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in lm[:468]], dtype=np.float32)


def detect_landmarks(bgr: np.ndarray, mesh=None) -> Optional[np.ndarray]:
    """Return (468, 2) float pixel coords or None if no face.

    Tries the image as-is, then with a black border around it: a face that
    fills the whole frame (common in phone selfies and in the dataset) is
    often missed by the detector until it has some margin around it.
    """
    mesh = mesh or _get_mesh()
    lm = _landmarks_once(bgr, mesh)
    if lm is not None:
        return lm
    h, w = bgr.shape[:2]
    pad = int(0.25 * max(h, w))
    padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    lm = _landmarks_once(padded, mesh)
    if lm is None:
        return None
    lm -= pad
    # a face detected mostly outside the real image is not a usable face
    inside = (lm[:, 0] >= 0) & (lm[:, 0] < w) & (lm[:, 1] >= 0) & (lm[:, 1] < h)
    if inside.mean() < 0.6:
        return None
    lm[:, 0] = np.clip(lm[:, 0], 0, w - 1)
    lm[:, 1] = np.clip(lm[:, 1], 0, h - 1)
    return lm


def _poly_mask(shape, pts_list) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for pts in pts_list:
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    return m


# MediaPipe's image-left points (33, 234 ...) are the SUBJECT's right cheek on
# a normal, un-mirrored photo; zone names follow that anatomical convention.
_SIDE_NAMES = {"cheek_L": "right_cheek", "cheek_R": "left_cheek"}


def segment_zones(bgr: np.ndarray, landmarks: Optional[np.ndarray] = None,
                  min_area_frac: float = 0.002,
                  occluded_ratio: float = 0.30) -> Optional[Dict[str, Zone]]:
    """Segment the face in `bgr` into zones. Returns None if no face.

    Zones smaller than `min_area_frac` of the image (e.g. a cheek hidden by a
    strong head turn) are dropped rather than returned as a sliver, and a
    cheek less than `occluded_ratio` the size of the other cheek is treated
    as turned away from the camera.
    """
    lm = detect_landmarks(bgr) if landmarks is None else landmarks
    if lm is None:
        return None
    h, w = bgr.shape[:2]
    zones: Dict[str, Zone] = {}
    for key, polys in _POLYS.items():
        name = _SIDE_NAMES.get(key, key)
        mask = _poly_mask((h, w), [lm[p] for p in polys])
        area = int(cv2.countNonZero(mask))
        if area < min_area_frac * h * w:
            continue
        ys, xs = np.nonzero(mask)
        zones[name] = Zone(name, mask, (int(xs.min()), int(ys.min()),
                                        int(xs.max()) + 1, int(ys.max()) + 1), area)

    # A strongly turned head squeezes the far cheek into a thin sliver that
    # is mostly nose/ear edge rather than cheek skin — drop it instead of
    # analysing a misleading crop.
    lc, rc = zones.get("left_cheek"), zones.get("right_cheek")
    if lc and rc:
        if lc.area < occluded_ratio * rc.area:
            zones.pop("left_cheek")
        elif rc.area < occluded_ratio * lc.area:
            zones.pop("right_cheek")
    return zones


def crop_zone(bgr: np.ndarray, zone: Zone, size: int = 224, pad: float = 0.08,
              blackout: bool = True) -> np.ndarray:
    """Square `size` x `size` crop of a zone for the classifier.

    Pixels outside the zone polygon are set to black (blackout=True) so the
    classifier only sees skin from that zone. Wide zones (under_eye spans
    both eyes, forehead and chin are long strips) are cut into a left and a
    right half stacked on top of each other, so they fill the square instead
    of becoming a thin, low-resolution stripe. The same function is used in
    training and in the API, so both see identical inputs.
    """
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = zone.box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
    img = bgr[y1:y2, x1:x2].copy()
    if blackout:
        img[zone.mask[y1:y2, x1:x2] == 0] = 0
    ch, cw = img.shape[:2]
    if cw > 1.8 * ch:
        mid = cw // 2
        left, right = img[:, :mid], img[:, mid:mid * 2]
        img = np.vstack([left, right])
        ch, cw = img.shape[:2]
    side = max(ch, cw)
    canvas = np.zeros((side, side, 3), np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = img
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interp)


def zone_outlines(mask: np.ndarray, epsilon: float = 1.5) -> List[List[List[int]]]:
    """Zone mask -> simplified outline(s) as [[[x, y], ...], ...]."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        c = cv2.approxPolyDP(c, epsilon, True)
        if len(c) >= 3:
            out.append([[int(p[0][0]), int(p[0][1])] for p in c])
    return out


def zone_of_box(zones: Dict[str, Zone], box) -> Optional[str]:
    """Which zone a detection box mostly falls in (None if outside all)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    best, best_px = None, 0
    for name, z in zones.items():
        px = int(cv2.countNonZero(z.mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]))
        if px > best_px:
            best, best_px = name, px
    return best


ZONE_COLORS = {  # BGR, for debug overlays
    "forehead": (255, 170, 60), "left_cheek": (80, 200, 80), "right_cheek": (60, 160, 255),
    "nose": (200, 80, 200), "under_eye": (60, 220, 230), "chin": (230, 120, 120),
}


def draw_zones(bgr: np.ndarray, zones: Dict[str, Zone], alpha: float = 0.35) -> np.ndarray:
    out = bgr.copy()
    overlay = bgr.copy()
    for name, z in zones.items():
        overlay[z.mask > 0] = ZONE_COLORS[name]
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    for name, z in zones.items():
        cnts, _ = cv2.findContours(z.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, ZONE_COLORS[name], 2)
    return out
