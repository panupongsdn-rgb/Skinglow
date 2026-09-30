"""
build_zone_dataset.py — Turn the YOLO box dataset into a per-ZONE dataset
=========================================================================
Input : dataset_clean_v2/  (train|valid|test)/images + labels, manifest.jsonl
Output: zone_dataset/
          train/*.jpg, valid/*.jpg, test/*.jpg   one 224x224 crop per face zone
                                                 (+ one "patch" crop per close-up photo)
          labels.csv                             one row per crop, 6 label columns
          stats.json                             counts, face-detection rate, rules used
          faces.jsonl                            per image: zone outlines + label boxes (used by review_tool.py)
          preview/*.jpg                          zone + box overlays for a visual check

Each face is split into forehead / left_cheek / right_cheek / nose / under_eye
/ chin with ai-service/face_zones.py (the same code the API uses), and every
crop gets one label per class:

     1  the class is present in this zone
     0  the class is absent from this zone
    -1  unknown -> ignored by the loss and by every metric

Why "unknown" exists (this matters for an honest Accuracy/F1):
  * The dataset was merged from several source datasets that each labelled
    only SOME classes (e.g. the LINE_ALBUM oily-skin set only draws
    `oiliness`, `levle0` only `black_spot`, `img-xxxx` only `wrinkle`).
    A face from the oily set may well have acne nobody boxed, so "no acne box"
    there is NOT evidence of "no acne". Classes a source never labels -> -1.
  * `oiliness` is annotated as a few sample patches per face, not every oily
    area, so zones of an oily-positive face without a patch -> -1.
  * A big box that only grazes a zone -> -1 rather than guessing.
  * An image with no boxes at all -> all -1 by default. Spot checks found
    unannotated images with clearly visible spots/acne, so an empty label
    file is not reliable evidence of clear skin (use --trust-empty-images to
    count them as negatives anyway). Negatives still come from the zones of
    annotated faces that no box touches.
  * Classes that cannot occur in a zone (eyebag outside under_eye) -> -1;
    the API forces those to 0 at inference time.

Run (on the machine that has the dataset):
    python build_zone_dataset.py --data-root "E:/Project END/Sangmai/skinglow-trian/dataset_clean_v2" --out zone_dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool, freeze_support
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import face_zones as fz  # noqa: E402

CLASSES = fz.CLASSES

# Classes annotated as sample patches rather than exhaustively.
PARTIAL_POSITIVE_CLASSES = {"oiliness"}

POS_FRAC = 0.30        # box covers >=30% of the zone, or >=30% of the box is in the zone -> present
UNCERTAIN_FRAC = 0.05  # overlaps, but less than POS_FRAC -> unknown
MIN_FAMILY_SIZE = 20   # smaller source families are pooled into "misc"


# ----------------------------------------------------------------------------
# Source families and the classes each one actually labels
# ----------------------------------------------------------------------------
def raw_family(image_name: str) -> str:
    stem = image_name.split(".rf.")[0]
    if stem.startswith("neg_"):
        return "neg"
    m = re.match(r"([A-Za-z]+(?:[_ -][A-Za-z]+)*)", stem)
    return m.group(1).lower() if m and len(m.group(1)) > 2 else "misc"


def family_label_sets(rows):
    size = Counter(raw_family(r["image"]) for r in rows)
    fam = {r["image"]: (raw_family(r["image"]) if size[raw_family(r["image"])] >= MIN_FAMILY_SIZE else "misc")
           for r in rows}
    per = defaultdict(Counter)
    n = Counter()
    for r in rows:
        f = fam[r["image"]]
        n[f] += 1
        for c in set(r["classes"]):
            per[f][c] += 1
    sets = {}
    for f in n:
        # a class counts as "labelled by this source" if it shows up in at
        # least 3 images and 1% of the source (filters stray relabels)
        sets[f] = sorted(c for c, k in per[f].items() if k >= max(3, 0.01 * n[f]))
    # the neg_ patches are non-face crops; they never produce zones anyway
    return fam, sets, dict(n)


# ----------------------------------------------------------------------------
# Per-image worker
# ----------------------------------------------------------------------------
_MESH = None


def _init_worker():
    global _MESH
    _MESH = fz.make_face_mesh()


def read_boxes(label_path: Path, w: int, h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        if len(parts) > 5:  # polygon label -> use its bounding box
            xs = [float(v) for v in parts[1::2]]
            ys = [float(v) for v in parts[2::2]]
            x1, x2, y1, y2 = min(xs) * w, max(xs) * w, min(ys) * h, max(ys) * h
        else:
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1, x2 = (cx - bw / 2) * w, (cx + bw / 2) * w
            y1, y2 = (cy - bh / 2) * h, (cy + bh / 2) * h
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
        if x2 > x1 and y2 > y1 and 0 <= c < len(CLASSES):
            boxes.append((CLASSES[c], x1, y1, x2, y2))
    return boxes


def zone_labels(zone: fz.Zone, boxes, labelled_classes, image_classes):
    out = {}
    for c in CLASSES:
        if not fz.class_allowed(c, zone.name) or c not in labelled_classes:
            out[c] = -1
            continue
        state = 0
        for bc, x1, y1, x2, y2 in boxes:
            if bc != c:
                continue
            inter = int(cv2.countNonZero(zone.mask[y1:y2, x1:x2]))
            if inter == 0:
                continue
            f_box = inter / max(1, (x2 - x1) * (y2 - y1))
            f_zone = inter / max(1, zone.area)
            if f_box >= POS_FRAC or f_zone >= POS_FRAC:
                state = 1
                break
            if f_box >= UNCERTAIN_FRAC or f_zone >= UNCERTAIN_FRAC:
                state = -1
        if state == 0 and c in PARTIAL_POSITIVE_CLASSES and c in image_classes:
            state = -1
        out[c] = state
    return out


def process(job):
    split, image_path, label_path, out_dir, size, meta, labelled, preview_path, closeups, trust_empty = job
    img = cv2.imread(image_path)
    if img is None:
        return {"image": meta["image"], "status": "unreadable", "rows": [], "face": None}
    h, w = img.shape[:2]
    boxes = read_boxes(Path(label_path), w, h)
    if not boxes and not trust_empty:
        labelled = set()  # nothing annotated -> every label unknown
    lm = fz.detect_landmarks(img, _MESH)
    zones = fz.segment_zones(img, landmarks=lm) if lm is not None else None
    if not zones:
        if not closeups or meta["family"] == "neg":
            return {"image": meta["image"], "status": "no_face", "rows": [], "face": None}
        # Close-up skin photo (no whole face visible): keep it as one
        # "patch" sample. It teaches what acne/spots/oil look like up close,
        # but is reported separately from the zone metrics.
        full = np.full((h, w), 255, np.uint8)
        zones = {"patch": fz.Zone("patch", full, (0, 0, w, h), h * w)}
        status = "closeup"
    else:
        status = "ok"
    image_classes = {b[0] for b in boxes}
    stem = Path(image_path).stem
    rows = []
    for name, z in zones.items():
        crop = fz.crop_zone(img, z, size=size)
        rel = f"{split}/{stem}__{name}.jpg"
        cv2.imwrite(str(Path(out_dir) / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        labels = zone_labels(z, boxes, labelled, image_classes)
        rows.append({"split": split, "crop": rel, "image": meta["image"],
                     "source_group": meta["source_group"], "family": meta["family"],
                     "zone": name, **labels})
    if preview_path and status == "ok":
        vis = fz.draw_zones(img, zones)
        for c, x1, y1, x2, y2 in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(vis, c, (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(preview_path, vis)
    face = {"image": meta["image"], "split": split, "family": meta["family"], "status": status,
            "width": w, "height": h,
            "zones": {n: fz.zone_outlines(z.mask) for n, z in zones.items()},
            "boxes": [[c, x1, y1, x2, y2] for c, x1, y1, x2, y2 in boxes]}
    return {"image": meta["image"], "status": status, "rows": rows, "face": face}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="dataset_clean_v2 folder (has manifest.jsonl)")
    ap.add_argument("--out", default="zone_dataset")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--previews", type=int, default=40, help="overlay images to save for a visual check")
    ap.add_argument("--no-closeups", action="store_true",
                    help="drop close-up photos with no detectable face instead of keeping them as 'patch' samples")
    ap.add_argument("--trust-empty-images", action="store_true",
                    help="treat images with an empty label file as clear skin (negative for their source's classes)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only process N images")
    ap.add_argument("--splits", default="train,valid,test",
                    help="comma-separated splits to build, e.g. 'test' to prepare only the test set for review")
    args = ap.parse_args()
    want_splits = {s.strip() for s in args.splits.split(",") if s.strip()}

    root, out = Path(args.data_root), Path(args.out)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    fam, label_sets, fam_sizes = family_label_sets(rows)
    for s in ("train", "valid", "test", "preview"):
        (out / s).mkdir(parents=True, exist_ok=True)

    jobs = []
    for i, r in enumerate(rows):
        if r["split"] not in want_splits:
            continue
        img = root / r["split"] / "images" / r["image"]
        if not img.exists():
            continue
        lbl = root / r["split"] / "labels" / (Path(r["image"]).stem + ".txt")
        meta = {"image": r["image"], "source_group": r["source_group"], "family": fam[r["image"]]}
        jobs.append([r["split"], str(img), str(lbl), str(out), args.size, meta,
                     set(label_sets[fam[r["image"]]]), "", not args.no_closeups, args.trust_empty_images])
        if args.limit and len(jobs) >= args.limit:
            break
    # overlay previews for a spread-out sample of images
    if args.previews > 0:
        step = max(1, len(jobs) // args.previews)
        for j in jobs[::step][:args.previews]:
            j[7] = str(out / "preview" / f"{j[0]}_{Path(j[1]).stem}.jpg")
    jobs = [tuple(j) for j in jobs]

    print(f"{len(jobs)} images, {args.workers} workers -> {out}")
    status = Counter()
    status_by_split = defaultdict(Counter)
    all_rows, faces = [], []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for k, res in enumerate(pool.imap_unordered(process, jobs, chunksize=8), 1):
            status[res["status"]] += 1
            all_rows.extend(res["rows"])
            if res["face"]:
                faces.append(res["face"])
            if res["rows"]:
                status_by_split[res["rows"][0]["split"]][res["status"]] += 1
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}  {dict(status)}")

    all_rows.sort(key=lambda r: r["crop"])
    fields = ["split", "crop", "image", "source_group", "family", "zone"] + CLASSES
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    faces.sort(key=lambda f: (f["split"], f["image"]))
    with open(out / "faces.jsonl", "w", encoding="utf-8") as f:
        for face in faces:
            f.write(json.dumps(face, ensure_ascii=False) + "\n")

    counts = {s: {c: {"pos": 0, "neg": 0, "unknown": 0} for c in CLASSES} for s in ("train", "valid", "test")}
    for r in all_rows:
        for c in CLASSES:
            counts[r["split"]][c][{1: "pos", 0: "neg", -1: "unknown"}[r[c]]] += 1
    stats = {
        "images": dict(status),
        "images_per_split": {s: dict(v) for s, v in status_by_split.items()},
        "crops_per_split": dict(Counter(r["split"] for r in all_rows)),
        "crops_per_zone": dict(Counter(r["zone"] for r in all_rows)),
        "label_counts": counts,
        "source_families": {f: {"images": fam_sizes[f], "labelled_classes": label_sets[f]} for f in sorted(fam_sizes)},
        "rules": {"trust_empty_images": args.trust_empty_images, "closeups": not args.no_closeups,
                  "POS_FRAC": POS_FRAC, "UNCERTAIN_FRAC": UNCERTAIN_FRAC,
                  "ALLOWED_ZONES": {c: sorted(z) for c, z in fz.ALLOWED_ZONES.items()},
                  "PARTIAL_POSITIVE_CLASSES": sorted(PARTIAL_POSITIVE_CLASSES)},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nImages:", dict(status))
    print("Crops :", stats["crops_per_split"])
    print(f"\n{'class':<11}" + "".join(f"{s:>24}" for s in ("train", "valid", "test")))
    for c in CLASSES:
        print(f"{c:<11}" + "".join(
            f"{counts[s][c]['pos']:>8}+ {counts[s][c]['neg']:>6}- {counts[s][c]['unknown']:>6}?" for s in ("train", "valid", "test")))
    print(f"\nWrote {out / 'labels.csv'} and {out / 'stats.json'}; check {out / 'preview'} before training.")


if __name__ == "__main__":
    freeze_support()
    main()
