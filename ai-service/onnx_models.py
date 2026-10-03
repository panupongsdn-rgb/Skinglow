"""
onnx_models.py — รันโมเดล YOLO และ zone classifier ด้วย ONNX Runtime (ไม่ต้องใช้ PyTorch)
=========================================================================================
PyTorch + ultralytics take ~400 MB of RAM just to be imported, which does not
fit Render's free 512 MB instance next to MediaPipe and the two models. The
same models exported with export_onnx.py run here on onnxruntime + numpy only.

Pre/post-processing mirrors ultralytics 8.4 predict() for a .pt model
(rect letterbox to stride 32, per-class NMS at IoU 0.7, max 300 boxes) so the
detections match the PyTorch path.
"""
from __future__ import annotations

import ast
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from face_zones import CLASSES, ZONE_NAMES, class_allowed

ZONES = ZONE_NAMES + ["patch"]
MEAN = np.array((0.485, 0.456, 0.406), np.float32).reshape(1, 1, 3)
STD = np.array((0.229, 0.224, 0.225), np.float32).reshape(1, 1, 3)


def _session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False  # lower resident memory on small instances
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# YOLOv8 detector
# ---------------------------------------------------------------------------
class OnnxYolo:
    def __init__(self, path: str, imgsz: int = 640, iou: float = 0.7, max_det: int = 300):
        self.sess = _session(path)
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names: Dict[int, str] = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
        self.stride = int(meta.get("stride", 32))
        self.imgsz, self.iou, self.max_det = imgsz, iou, max_det
        self.input = self.sess.get_inputs()[0].name

    def _letterbox(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = round(w * r), round(h * r)
        dw, dh = (self.imgsz - nw) % self.stride / 2, (self.imgsz - nh) % self.stride / 2
        img = bgr if (w, h) == (nw, nh) else cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1),
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), img.shape[:2]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> List[int]:
        order = scores.argsort()[::-1]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = []
        while order.size:
            i = order[0]; keep.append(int(i))
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            ov = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][ov <= iou]
        return keep

    def detect(self, bgr: np.ndarray, conf: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (xyxy boxes in original pixels, confidences, class ids)."""
        x, (ih, iw) = self._letterbox(bgr)
        pred = self.sess.run(None, {self.input: x})[0][0].T  # (anchors, 4 + nc)
        cls_scores = pred[:, 4:]
        cls = cls_scores.argmax(1); score = cls_scores[np.arange(len(cls)), cls]
        m = score > conf
        if not m.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        b, score, cls = pred[m, :4], score[m], cls[m]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        keep = self._nms(xyxy + cls[:, None] * 7680.0, score, self.iou)[: self.max_det]
        xyxy, score, cls = xyxy[keep], score[keep], cls[keep]
        h0, w0 = bgr.shape[:2]
        gain = min(ih / h0, iw / w0)
        px = round((iw - round(w0 * gain)) / 2 - 0.1); py = round((ih - round(h0 * gain)) / 2 - 0.1)
        xyxy = (xyxy - np.array([px, py, px, py], np.float32)) / gain
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0); xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0)
        return xyxy, score, cls


# ---------------------------------------------------------------------------
# Per-zone classifier (same interface as zone_model.ZonePredictor)
# ---------------------------------------------------------------------------
class OnnxZonePredictor:
    def __init__(self, path: str):
        self.sess = _session(path)
        meta = self.sess.get_modelmeta().custom_metadata_map
        th = json.loads(meta["thresholds"])
        self.meta = {"arch": meta.get("arch"), "epoch": meta.get("epoch"), "thresholds": th,
                     "img_size": int(meta.get("img_size", 224)), "runtime": "onnxruntime"}
        self.img_size = self.meta["img_size"]
        self.thresholds = np.array([th[c] for c in CLASSES], dtype=np.float32)
        self.allowed = np.array([[1.0 if class_allowed(c, z) else 0.0 for c in CLASSES] for z in ZONES], np.float32)

    def predict(self, crops_bgr: List[np.ndarray], zone_names: List[str]) -> List[Dict[str, dict]]:
        """Return, per crop, {class: {"prob": p, "present": bool}}."""
        if not crops_bgr:
            return []
        x = np.stack([((c[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1) for c in crops_bgr])
        zi = np.array([ZONES.index(z) for z in zone_names], dtype=np.int64)
        logits = self.sess.run(None, {"image": np.ascontiguousarray(x), "zone": zi})[0]
        probs = (1.0 / (1.0 + np.exp(-logits))) * self.allowed[zi]
        return [{c: {"prob": round(float(p[j]), 3), "present": bool(p[j] >= self.thresholds[j])}
                 for j, c in enumerate(CLASSES)} for p in probs]
