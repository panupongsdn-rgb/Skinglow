from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


CLASS_THRESHOLDS = {
    "acne": 0.35,
    "black_spot": 0.55,
    "eyebag": 0.35,
    "oiliness": 0.45,
    "redness": 0.35,
    "wrinkle": 0.40,
}


class FaceSkinDetector:
    """Run skin-condition YOLO only inside a MediaPipe face-oval mask."""

    def __init__(
        self,
        weights_path: str | Path,
        image_size: int = 640,
        face_overlap_threshold: float = 0.60,
    ) -> None:
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}"
            )

        self.model = YOLO(str(self.weights_path))
        self.image_size = image_size
        self.face_overlap_threshold = face_overlap_threshold
        self._lock = Lock()

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.60,
        )

        self._face_oval_indices = self._connection_indices(
            self._mp_face_mesh.FACEMESH_FACE_OVAL
        )
        self._left_eye_indices = self._connection_indices(
            self._mp_face_mesh.FACEMESH_LEFT_EYE
        )
        self._right_eye_indices = self._connection_indices(
            self._mp_face_mesh.FACEMESH_RIGHT_EYE
        )
        self._lips_indices = self._connection_indices(
            self._mp_face_mesh.FACEMESH_LIPS
        )

    @staticmethod
    def _connection_indices(connections) -> list[int]:
        indices: set[int] = set()

        for start_index, end_index in connections:
            indices.add(start_index)
            indices.add(end_index)

        return sorted(indices)

    @staticmethod
    def _landmarks_to_points(
        landmarks,
        indices: list[int],
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        points = []

        for index in indices:
            landmark = landmarks[index]

            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)

            x = max(0, min(x, image_width - 1))
            y = max(0, min(y, image_height - 1))

            points.append([x, y])

        return np.asarray(points, dtype=np.int32)

    def _remove_feature_from_mask(
        self,
        mask: np.ndarray,
        landmarks,
        indices: list[int],
        image_width: int,
        image_height: int,
    ) -> None:
        points = self._landmarks_to_points(
            landmarks=landmarks,
            indices=indices,
            image_width=image_width,
            image_height=image_height,
        )

        if len(points) < 3:
            return

        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 0)

    def _create_face_mask(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        image_height, image_width = image.shape[:2]

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        face_mesh_result = self._face_mesh.process(rgb_image)

        if not face_mesh_result.multi_face_landmarks:
            return None

        landmarks = face_mesh_result.multi_face_landmarks[0].landmark

        face_points = self._landmarks_to_points(
            landmarks=landmarks,
            indices=self._face_oval_indices,
            image_width=image_width,
            image_height=image_height,
        )

        face_hull = cv2.convexHull(face_points)

        face_mask = np.zeros(
            (image_height, image_width),
            dtype=np.uint8,
        )

        cv2.fillConvexPoly(
            face_mask,
            face_hull,
            255,
        )

        # Remove actual eyeballs and lips from the skin mask. The area below
        # the eyes remains available for the eyebag class.
        self._remove_feature_from_mask(
            face_mask,
            landmarks,
            self._left_eye_indices,
            image_width,
            image_height,
        )
        self._remove_feature_from_mask(
            face_mask,
            landmarks,
            self._right_eye_indices,
            image_width,
            image_height,
        )
        self._remove_feature_from_mask(
            face_mask,
            landmarks,
            self._lips_indices,
            image_width,
            image_height,
        )

        x, y, width, height = cv2.boundingRect(face_hull)

        padding_x = int(width * 0.03)
        padding_y = int(height * 0.03)

        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(image_width, x + width + padding_x)
        y2 = min(image_height, y + height + padding_y)

        return face_mask, (x1, y1, x2, y2)

    @staticmethod
    def _calculate_face_overlap(
        face_mask: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> float:
        image_height, image_width = face_mask.shape
        x1, y1, x2, y2 = box

        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(x1 + 1, min(x2, image_width))
        y2 = max(y1 + 1, min(y2, image_height))

        box_mask = face_mask[y1:y2, x1:x2]
        box_area = (x2 - x1) * (y2 - y1)

        if box_area <= 0:
            return 0.0

        face_pixels = cv2.countNonZero(box_mask)

        return face_pixels / box_area

    def analyze(self, image: np.ndarray) -> dict[str, Any]:
        """Analyze one BGR OpenCV image and return global image coordinates."""

        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        with self._lock:
            face_result = self._create_face_mask(image)

            if face_result is None:
                return {
                    "face_detected": False,
                    "face_box": None,
                    "detections": [],
                }

            face_mask, face_box = face_result
            face_x1, face_y1, face_x2, face_y2 = face_box

            face_crop = image[
                face_y1:face_y2,
                face_x1:face_x2,
            ]

            prediction = self.model.predict(
                source=face_crop,
                imgsz=self.image_size,
                conf=0.20,
                iou=0.50,
                max_det=100,
                verbose=False,
            )[0]

        image_height, image_width = image.shape[:2]
        detections = []

        for predicted_box in prediction.boxes:
            class_id = int(predicted_box.cls[0].item())
            confidence = float(predicted_box.conf[0].item())
            class_name = self.model.names[class_id]

            required_confidence = CLASS_THRESHOLDS.get(
                class_name,
                0.35,
            )

            if confidence < required_confidence:
                continue

            local_x1, local_y1, local_x2, local_y2 = (
                predicted_box.xyxy[0].cpu().tolist()
            )

            global_x1 = int(local_x1 + face_x1)
            global_y1 = int(local_y1 + face_y1)
            global_x2 = int(local_x2 + face_x1)
            global_y2 = int(local_y2 + face_y1)

            global_x1 = max(0, min(global_x1, image_width - 1))
            global_y1 = max(0, min(global_y1, image_height - 1))
            global_x2 = max(
                global_x1 + 1,
                min(global_x2, image_width),
            )
            global_y2 = max(
                global_y1 + 1,
                min(global_y2, image_height),
            )

            center_x = (global_x1 + global_x2) // 2
            center_y = (global_y1 + global_y2) // 2

            if face_mask[center_y, center_x] == 0:
                continue

            face_overlap = self._calculate_face_overlap(
                face_mask=face_mask,
                box=(
                    global_x1,
                    global_y1,
                    global_x2,
                    global_y2,
                ),
            )

            if face_overlap < self.face_overlap_threshold:
                continue

            detections.append(
                {
                    "box": [
                        global_x1,
                        global_y1,
                        global_x2,
                        global_y2,
                    ],
                    "label": class_name,
                    "confidence": round(confidence, 4),
                    "face_overlap": round(face_overlap, 4),
                }
            )

        return {
            "face_detected": True,
            "face_box": [
                face_x1,
                face_y1,
                face_x2,
                face_y2,
            ],
            "detections": detections,
        }

    def close(self) -> None:
        self._face_mesh.close()
