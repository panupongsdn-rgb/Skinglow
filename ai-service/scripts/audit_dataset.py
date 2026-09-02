"""
audit_dataset.py — Dataset quality audit for Skinglow YOLO training data.

Run this BEFORE every training run. Produces a markdown + JSON report
covering everything requested in the improvement plan:
  - images/boxes per class
  - class imbalance ratios
  - duplicate images (exact, via SHA256)
  - near-duplicate images (perceptual hash, catches resized/re-compressed dupes)
    NOTE: phash works on visual structure/frequency content. It's reliable on
    real photos but can false-positive on very simple/flat/low-detail images
    (tested during development: solid-color placeholder images all hashed as
    "identical"). If near-dup findings look suspiciously numerous, sanity
    check a few pairs visually before trusting the report — real face photos
    have enough texture that this is much less likely to be an issue than it
    was in synthetic testing.
  - corrupted/unreadable images
  - malformed or out-of-bounds bounding boxes
  - missing label files / images with no matching label
  - empty label files (legitimate as "background" images — reported, not flagged as errors)

Usage:
    pip install pillow numpy imagehash tqdm
    python audit_dataset.py --data data.yaml --out audit_report

This does NOT modify your dataset. It only reports. Fixing issues (removing
dupes, re-labeling, etc.) is a deliberate manual step — see the report's
"Recommended actions" section for what to do with each finding.
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("[warn] imagehash not installed — near-duplicate detection will be skipped.")
    print("       pip install imagehash to enable it.")


def load_data_yaml(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def find_split_dirs(cfg, yaml_dir):
    """Resolve image/label directories for train/val/test from data.yaml."""
    base = Path(cfg.get("path", ".") )
    if not base.is_absolute():
        base = yaml_dir / base

    splits = {}
    for split_key in ("train", "val", "test"):
        if split_key not in cfg:
            continue
        img_dir = base / cfg[split_key]
        # labels dir is conventionally images/ -> labels/ sibling
        label_dir = Path(str(img_dir).replace("images", "labels"))
        splits[split_key] = {"images": img_dir, "labels": label_dir}
    return splits


def sha256_of_file(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def parse_yolo_label(label_path):
    """Returns list of (class_id, x, y, w, h) tuples. Raises on malformed lines."""
    boxes = []
    with open(label_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"line {line_no}: expected 5 values, got {len(parts)}")
            cls_id = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
            boxes.append((cls_id, x, y, w, h))
    return boxes


def audit_split(split_name, img_dir, label_dir, class_names):
    """Returns a dict of findings for one split (train/val/test)."""
    findings = {
        "split": split_name,
        "num_images": 0,
        "num_labeled_images": 0,
        "num_background_images": 0,  # images with an empty label file (legitimate negatives)
        "num_missing_labels": 0,     # images with NO label file at all (likely an error)
        "corrupted_images": [],
        "malformed_labels": [],
        "invalid_boxes": [],         # out-of-[0,1] range or zero-size
        "per_class_image_count": defaultdict(set),   # class -> set of image stems (dedup)
        "per_class_box_count": defaultdict(int),
        "file_hashes": {},           # sha256 -> [filenames] for exact-dupe detection
        "phashes": {},                # phash -> [filenames] for near-dupe detection (if available)
    }

    if not img_dir.exists():
        findings["error"] = f"image directory not found: {img_dir}"
        return findings

    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    findings["num_images"] = len(image_files)

    for img_path in image_files:
        # --- corrupted image check ---
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                im.load()
        except (UnidentifiedImageError, OSError, ValueError) as e:
            findings["corrupted_images"].append(str(img_path))
            continue  # skip further checks on unreadable file

        # --- exact duplicate (hash) ---
        file_hash = sha256_of_file(img_path)
        findings["file_hashes"].setdefault(file_hash, []).append(str(img_path))

        # --- near-duplicate (perceptual hash) ---
        if HAS_IMAGEHASH:
            try:
                with Image.open(img_path) as im:
                    ph = str(imagehash.phash(im))
                findings["phashes"].setdefault(ph, []).append(str(img_path))
            except Exception:
                pass

        # --- matching label file ---
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            findings["num_missing_labels"] += 1
            continue

        try:
            boxes = parse_yolo_label(label_path)
        except ValueError as e:
            findings["malformed_labels"].append(f"{label_path}: {e}")
            continue

        if len(boxes) == 0:
            findings["num_background_images"] += 1
            continue

        findings["num_labeled_images"] += 1
        seen_classes_this_image = set()
        for (cls_id, x, y, w, h) in boxes:
            # bbox sanity: center + half-width/height must stay within [0,1], w/h must be > 0
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                findings["invalid_boxes"].append(f"{label_path}: class={cls_id} box=({x},{y},{w},{h}) out of range")
                continue
            x0, x1 = x - w / 2, x + w / 2
            y0, y1 = y - h / 2, y + h / 2
            if x0 < -0.01 or x1 > 1.01 or y0 < -0.01 or y1 > 1.01:
                findings["invalid_boxes"].append(f"{label_path}: class={cls_id} box extends outside image bounds")
                continue

            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"UNKNOWN_{cls_id}"
            findings["per_class_box_count"][cls_name] += 1
            seen_classes_this_image.add(cls_name)

        for cls_name in seen_classes_this_image:
            findings["per_class_image_count"][cls_name].add(img_path.stem)

    return findings


def cross_split_leakage(splits_findings):
    """Flags exact and near-duplicate images that appear across different splits
    (e.g. the same photo in both train/ and valid/) — this is data leakage."""
    exact_leaks = []
    near_leaks = []

    # exact duplicates across splits (by sha256)
    hash_to_splits = defaultdict(list)
    for f in splits_findings:
        for h, paths in f["file_hashes"].items():
            for p in paths:
                hash_to_splits[h].append((f["split"], p))
    for h, occurrences in hash_to_splits.items():
        splits_involved = set(s for s, _ in occurrences)
        if len(splits_involved) > 1:
            exact_leaks.append({"hash": h, "occurrences": occurrences})

    # near-duplicates across splits (by phash, hamming distance <= 4)
    if HAS_IMAGEHASH:
        all_phashes = []
        for f in splits_findings:
            for ph, paths in f["phashes"].items():
                for p in paths:
                    all_phashes.append((f["split"], ph, p))

        # naive O(n^2) comparison — fine for a few thousand images, not for huge datasets
        n = len(all_phashes)
        for i in range(n):
            split_i, ph_i, path_i = all_phashes[i]
            hash_i = imagehash.hex_to_hash(ph_i)
            for j in range(i + 1, n):
                split_j, ph_j, path_j = all_phashes[j]
                if split_i == split_j:
                    continue
                hash_j = imagehash.hex_to_hash(ph_j)
                if hash_i - hash_j <= 4:  # small hamming distance = visually near-identical
                    near_leaks.append({"a": (split_i, path_i), "b": (split_j, path_j), "distance": int(hash_i - hash_j)})

    return exact_leaks, near_leaks


def main():
    parser = argparse.ArgumentParser(description="Audit a Skinglow YOLO dataset for quality issues")
    parser.add_argument("--data", type=str, default="data.yaml")
    parser.add_argument("--out", type=str, default="audit_report")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    cfg = load_data_yaml(data_path)
    class_names = cfg["names"]
    splits = find_split_dirs(cfg, data_path.parent)

    all_findings = []
    for split_name, dirs in splits.items():
        print(f"[audit] scanning {split_name} ({dirs['images']})...")
        findings = audit_split(split_name, dirs["images"], dirs["labels"], class_names)
        all_findings.append(findings)

    exact_leaks, near_leaks = cross_split_leakage(all_findings)

    # ---------------- build report ----------------
    os.makedirs(args.out, exist_ok=True)
    report_lines = ["# Skinglow Dataset Audit Report\n"]
    report_lines.append(f"Classes: {class_names}\n")

    for f in all_findings:
        report_lines.append(f"\n## Split: {f['split']}\n")
        if "error" in f:
            report_lines.append(f"⚠️ {f['error']}\n")
            continue
        report_lines.append(f"- Total images: {f['num_images']}")
        report_lines.append(f"- Labeled images (>=1 box): {f['num_labeled_images']}")
        report_lines.append(f"- Background images (empty label, legitimate negatives): {f['num_background_images']}")
        report_lines.append(f"- Images missing a label file entirely (likely error): {f['num_missing_labels']}")
        report_lines.append(f"- Corrupted/unreadable images: {len(f['corrupted_images'])}")
        report_lines.append(f"- Malformed label lines: {len(f['malformed_labels'])}")
        report_lines.append(f"- Invalid/out-of-bounds boxes: {len(f['invalid_boxes'])}")

        exact_dupes_in_split = sum(1 for paths in f["file_hashes"].values() if len(paths) > 1)
        report_lines.append(f"- Exact duplicate images WITHIN this split: {exact_dupes_in_split} groups")

        report_lines.append("\n**Per-class distribution (this split):**\n")
        report_lines.append("| Class | Images | Boxes | Avg boxes/image |")
        report_lines.append("|---|---:|---:|---:|")
        for cls in class_names:
            n_img = len(f["per_class_image_count"].get(cls, set()))
            n_box = f["per_class_box_count"].get(cls, 0)
            avg = round(n_box / n_img, 2) if n_img else 0
            report_lines.append(f"| {cls} | {n_img} | {n_box} | {avg} |")

        if f["corrupted_images"]:
            report_lines.append(f"\n**Corrupted files:**\n```\n" + "\n".join(f["corrupted_images"][:50]) + "\n```")
        if f["malformed_labels"]:
            report_lines.append(f"\n**Malformed labels:**\n```\n" + "\n".join(f["malformed_labels"][:50]) + "\n```")
        if f["invalid_boxes"]:
            report_lines.append(f"\n**Invalid boxes:**\n```\n" + "\n".join(f["invalid_boxes"][:50]) + "\n```")

    report_lines.append("\n## Cross-split leakage (train/val/test overlap)\n")
    report_lines.append(f"- Exact-duplicate leaks: {len(exact_leaks)}")
    report_lines.append(f"- Near-duplicate leaks (phash distance <= 4): {len(near_leaks)}")
    if exact_leaks:
        report_lines.append("\n**Exact duplicates across splits (THIS IS DATA LEAKAGE — fix before training):**\n")
        for leak in exact_leaks[:50]:
            report_lines.append(f"- {leak['occurrences']}")
    if near_leaks:
        report_lines.append("\n**Near-duplicates across splits (probably leakage — review manually):**\n")
        for leak in near_leaks[:50]:
            report_lines.append(f"- {leak['a']} <-> {leak['b']} (distance {leak['distance']})")

    report_lines.append("\n## Recommended actions\n")
    report_lines.append("- Any split with `error` above: fix the path in data.yaml first, nothing else in this report is valid until that's resolved.")
    report_lines.append("- Corrupted images: remove them (or re-export from source) — they will crash or silently skip during training.")
    report_lines.append("- Malformed labels / invalid boxes: fix by hand or re-export from your annotation tool; don't just delete unless the image itself is unusable.")
    report_lines.append("- Exact cross-split duplicates: remove the copy from val/test (never from train) — keeping it in val too means you're testing on training data.")
    report_lines.append("- Near-duplicate cross-split leaks: inspect each pair; if they're genuinely the same photo (or same subject, same pose, minor edits), move all copies into ONE split only.")
    report_lines.append("- Classes with very few images/boxes (see per-class table): this is your class-imbalance evidence — cross-reference with the per-class validation metrics you already have.")

    report_path = Path(args.out) / "audit_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    # also dump raw JSON for programmatic use
    json_safe = []
    for f in all_findings:
        f2 = dict(f)
        f2["per_class_image_count"] = {k: len(v) for k, v in f["per_class_image_count"].items()}
        f2["per_class_box_count"] = dict(f["per_class_box_count"])
        f2["file_hashes"] = {k: v for k, v in f["file_hashes"].items() if len(v) > 1}  # only keep dupes
        f2["phashes"] = {}  # too large to dump raw; leakage already summarized above
        json_safe.append(f2)

    with open(Path(args.out) / "audit_report.json", "w") as f:
        json.dump({"findings": json_safe, "exact_leaks": exact_leaks, "near_leaks": near_leaks}, f, indent=2)

    print(f"\n[audit] Done. Report written to {report_path}")
    print(f"[audit] Total cross-split leaks found: {len(exact_leaks)} exact, {len(near_leaks)} near-duplicate")


if __name__ == "__main__":
    main()
