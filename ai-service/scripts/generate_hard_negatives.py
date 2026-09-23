"""
generate_hard_negatives.py — Turn your EXISTING dataset images into extra
"hard negative" training images, without needing any new photos.

WHY THIS EXISTS: the trained model sometimes draws bounding boxes on hair,
ears, or background clutter (see the confusion matrix's high background
false-positive counts, and the live example that prompted this script).
The `filter_detections_to_face()` fix in ai-service/main.py patches this
at inference time, but the real fix is teaching the model during training
that "hair/ears/background ≠ a skin condition." Standard YOLO practice
for this: add background images with EMPTY label files — no new photos
needed, just crops of the non-face regions you already have.

WHAT IT DOES:
  For each image in your training split:
    1. Detect the face (Haar cascade, same detector used in production).
    2. Read that image's existing YOLO label boxes (if any).
    3. Randomly sample small patches that do NOT overlap the face box or
       any existing labeled box — i.e. genuinely "nothing here" regions:
       hair, ears, clothing, background.
    4. Skip near-uniform patches (blank wall, solid color) — not useful
       as negatives, just noise.
    5. Save each kept patch as a new image + an EMPTY .txt label file.

  Images with no detectable face are skipped (can't safely say what's
  "non-face" without knowing where the face is).

This is NON-DESTRUCTIVE: it never touches your existing images/labels,
only writes new files to a separate output folder. Review the output,
then copy/merge it into your train split yourself.

Usage:
    pip install opencv-python-headless numpy pyyaml
    python generate_hard_negatives.py --data data.yaml --split train \
        --output-dir hard_negatives --patches-per-image 3

    # then, after reviewing hard_negatives/images and hard_negatives/labels:
    #   copy hard_negatives/images/*  -> your train/images/
    #   copy hard_negatives/labels/*  -> your train/labels/
    #   re-run scripts/audit_dataset.py to confirm no duplicates/leakage
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def load_data_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_split_dirs(cfg: dict, yaml_dir: Path, split: str):
    base = Path(cfg.get("path", "."))
    if not base.is_absolute():
        base = yaml_dir / base
    img_dir = base / cfg[split]
    label_dir = Path(str(img_dir).replace("images", "labels"))
    return img_dir, label_dir


def read_yolo_boxes_px(label_path: Path, img_w: int, img_h: int):
    """Returns existing boxes as pixel (x0,y0,x1,y1) tuples, ignoring class id."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in parts)
        x0 = (cx - bw / 2) * img_w
        y0 = (cy - bh / 2) * img_h
        x1 = (cx + bw / 2) * img_w
        y1 = (cy + bh / 2) * img_h
        boxes.append((x0, y0, x1, y1))
    return boxes


def detect_largest_face(img_bgr: np.ndarray):
    """Returns (x, y, w, h) of the largest detected face, or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda f: f[2] * f[3]))


def sample_patches_avoiding(img_shape, face_box, existing_boxes_px, patches_per_image,
                             patch_size, rng, max_attempts=80):
    """Core sampling logic — pure geometry, no image I/O, so it's testable
    in isolation from face detection. Returns a list of (x0,y0,x1,y1)
    candidate patch coordinates that don't overlap the face box or any
    existing labeled box."""
    h, w = img_shape[:2]
    fx, fy, fw, fh = face_box
    avoid_boxes = [(fx, fy, fx + fw, fy + fh)] + list(existing_boxes_px)

    def overlaps_any(px0, py0, px1, py1):
        for (bx0, by0, bx1, by1) in avoid_boxes:
            if not (px1 <= bx0 or px0 >= bx1 or py1 <= by0 or py0 >= by1):
                return True
        return False

    if w - patch_size <= 0 or h - patch_size <= 0:
        return []

    patches = []
    attempts = 0
    while len(patches) < patches_per_image and attempts < max_attempts:
        attempts += 1
        x0 = rng.randint(0, w - patch_size)
        y0 = rng.randint(0, h - patch_size)
        x1, y1 = x0 + patch_size, y0 + patch_size
        if overlaps_any(x0, y0, x1, y1):
            continue
        patches.append((x0, y0, x1, y1))
    return patches


def is_informative_patch(patch: np.ndarray, min_std: float) -> bool:
    """Rejects near-uniform patches (blank wall, solid-color background) —
    not useful as negatives, and could even hurt training if truly blank."""
    return float(patch.std()) >= min_std


def process_image(img_path: Path, label_path: Path, patches_per_image: int,
                   patch_ratio: float, min_std: float, rng: random.Random):
    img = cv2.imread(str(img_path))
    if img is None:
        return [], "unreadable"

    face = detect_largest_face(img)
    if face is None:
        return [], "no_face_detected"

    fx, fy, fw, fh = face
    h, w = img.shape[:2]
    existing_boxes = read_yolo_boxes_px(label_path, w, h)

    patch_size = max(24, int(max(fw, fh) * patch_ratio))
    candidates = sample_patches_avoiding(img.shape, face, existing_boxes,
                                          patches_per_image, patch_size, rng)

    kept_patches = []
    for (x0, y0, x1, y1) in candidates:
        patch = img[y0:y1, x0:x1]
        if is_informative_patch(patch, min_std):
            kept_patches.append(patch)

    return kept_patches, "ok"


def main():
    parser = argparse.ArgumentParser(description="Generate hard-negative training images from an existing dataset")
    parser.add_argument("--data", type=str, default="data.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                         help="Only sample from this split — never touch val/test with generated data without re-checking for leakage")
    parser.add_argument("--output-dir", type=str, default="hard_negatives")
    parser.add_argument("--patches-per-image", type=int, default=3)
    parser.add_argument("--patch-ratio", type=float, default=0.3,
                         help="Patch size as a fraction of the detected face's largest dimension")
    parser.add_argument("--min-std", type=float, default=15.0,
                         help="Minimum pixel std-dev for a patch to be kept (filters out blank/uniform crops)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    cfg = load_data_yaml(data_path)
    img_dir, label_dir = resolve_split_dirs(cfg, data_path.parent, args.split)

    out_img_dir = Path(args.output_dir) / "images"
    out_label_dir = Path(args.output_dir) / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    stats = {"processed": 0, "no_face": 0, "unreadable": 0, "patches_written": 0}

    for img_path in image_files:
        label_path = label_dir / (img_path.stem + ".txt")
        patches, status = process_image(img_path, label_path, args.patches_per_image,
                                         args.patch_ratio, args.min_std, rng)
        stats["processed"] += 1
        if status == "no_face_detected":
            stats["no_face"] += 1
        elif status == "unreadable":
            stats["unreadable"] += 1

        for i, patch in enumerate(patches):
            out_name = f"{img_path.stem}_neg{i}.jpg"
            cv2.imwrite(str(out_img_dir / out_name), patch)
            (out_label_dir / f"{img_path.stem}_neg{i}.txt").write_text("")  # empty = background image
            stats["patches_written"] += 1

    print(f"\nProcessed {stats['processed']} source images from '{args.split}' split ({img_dir})")
    print(f"  Skipped (no face detected): {stats['no_face']}")
    print(f"  Skipped (unreadable file):  {stats['unreadable']}")
    print(f"  Hard-negative patches written: {stats['patches_written']}")
    print(f"\nOutput: {out_img_dir} and {out_label_dir}")
    print("These are NEW files only — nothing in your original dataset was modified.")
    print("Review a sample of the output images before merging into your train split.")
    print("After merging, re-run scripts/audit_dataset.py to catch any accidental duplicates.")


if __name__ == "__main__":
    main()
