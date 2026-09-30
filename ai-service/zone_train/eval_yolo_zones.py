"""
eval_yolo_zones.py — วัด YOLO เดิมแบบรายโซน บน test ชุดเดียวกับ CNN
====================================================================
Scores the old detector (skin_analysis_farmasi.pt) the way the app uses it in
"yolo_per_zone" mode, on exactly the human-checked test labels the zone CNN
is scored on, so the two numbers can be compared directly.

For every reviewed face image: MediaPipe zones (same code as the dataset
builder) -> YOLO boxes -> each box goes to the zone it mostly covers ->
zone score for a class = highest confidence of that class's boxes in the zone.
"Present" = score >= --conf (the app's DETECTION_CONFIDENCE, default 0.35).

    python eval_yolo_zones.py --weights ai-service/models/skin_analysis_farmasi.pt ^
        --data-root dataset_clean_v2 --reviews review_compare/zone_reviews_final.jsonl ^
        --cnn-metrics runs_zone/<run>/metrics_test.json --out yolo_vs_cnn

Outputs: metrics_yolo_test.json, compare.md, yolo_vs_cnn.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import face_zones as fz  # noqa: E402

CLASSES = fz.CLASSES
FACE = fz.ZONE_NAMES


def load_reviews(p):
    out = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); out[r["image"]] = r
    return out


def stats(y, s, thr):
    y = np.asarray(y); s = np.asarray(s); p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # AUC (rank based, ties averaged)
    pos, neg = s[y == 1], s[y == 0]
    auc = float("nan")
    if len(pos) and len(neg):
        allv = np.concatenate([pos, neg]); order = allv.argsort(kind="mergesort")
        ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
        for v in np.unique(allv):
            m = allv == v
            if m.sum() > 1:
                ranks[m] = ranks[m].mean()
        auc = float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    n = tp + fp + tn + fn
    return {"n": n, "pos": tp + fn, "TP": tp, "FP": fp, "TN": tn, "FN": fn, "precision": prec, "recall": rec,
            "f1": f1, "accuracy": (tp + tn) / n if n else float("nan"),
            "balanced_accuracy": (rec + spec) / 2, "auc": auc, "threshold": thr}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.35, help="app threshold (DETECTION_CONFIDENCE)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cnn-metrics", default=None, help="metrics_test.json of the zone CNN run")
    ap.add_argument("--out", default="yolo_vs_cnn")
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = {int(k): v for k, v in model.names.items()}
    revs = {k: v for k, v in load_reviews(args.reviews).items()
            if v.get("split", args.split) == args.split and v["status"] == "ok"}
    mesh = fz.make_face_mesh(0.3)
    Y = {c: [] for c in CLASSES}; S = {c: [] for c in CLASSES}
    used = skipped = 0
    for i, (img_name, r) in enumerate(sorted(revs.items())):
        bgr = cv2.imread(str(Path(args.data_root) / args.split / "images" / img_name))
        face_zones_in_review = [z for z in r["labels"] if z in FACE]
        if bgr is None or not face_zones_in_review:
            skipped += 1; continue
        lm = fz.detect_landmarks(bgr, mesh)
        zones = fz.segment_zones(bgr, landmarks=lm) if lm is not None else None
        if not zones:
            skipped += 1; continue
        res = model.predict(bgr, conf=0.01, imgsz=args.imgsz, verbose=False)[0]
        score = {z: {c: 0.0 for c in CLASSES} for z in zones}
        for b, cf, cl in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            z = fz.zone_of_box(zones, b); c = names[int(cl)]
            if z and c in CLASSES and fz.class_allowed(c, z):
                score[z][c] = max(score[z][c], float(cf))
        for z in face_zones_in_review:
            if z not in score:
                continue
            for c, v in r["labels"][z].items():
                Y[c].append(int(v)); S[c].append(score[z][c])
        used += 1
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(revs)}")
    per = {c: stats(Y[c], S[c], args.conf) for c in CLASSES}
    scored = [c for c in CLASSES if per[c]["pos"] and per[c]["n"] - per[c]["pos"]]
    res = {"overall": {"macro_f1": float(np.mean([per[c]["f1"] for c in scored])),
                       "macro_balanced_accuracy": float(np.mean([per[c]["balanced_accuracy"] for c in scored])),
                       "images": used, "skipped": skipped, "conf": args.conf},
           "per_class": per}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_yolo_test.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    cnn = json.loads(Path(args.cnn_metrics).read_text(encoding="utf-8")) if args.cnn_metrics else None
    L = ["# YOLO เดิม (รายโซน) vs Zone CNN — test ที่คนตรวจแล้ว", "",
         f"YOLO: {used} ภาพ ({skipped} ข้าม), threshold {args.conf} เท่ากับในแอป", "",
         "| ปัญหา | ตัวอย่างบวก | F1 YOLO | F1 CNN | Recall YOLO | Recall CNN | Precision YOLO | Precision CNN | AUC YOLO | AUC CNN |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    f = lambda v: "–" if v is None or v != v else f"{v * 100:.1f}%"
    for c in CLASSES:
        y = per[c]; k = cnn["per_class"][c] if cnn else {}
        L.append(f"| {c} | {y['pos']} | {f(y['f1'])} | {f(k.get('f1'))} | {f(y['recall'])} | {f(k.get('recall'))} "
                 f"| {f(y['precision'])} | {f(k.get('precision'))} | {f(y['auc'])} | {f(k.get('auc'))} |")
    L.append(f"| **Macro** | | **{f(res['overall']['macro_f1'])}** | **{f(cnn['overall']['macro_f1']) if cnn else '–'}** | | | | | | |")
    (out / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if cnn:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = np.arange(len(CLASSES)); w = 0.38
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        for a, key, ttl in [(ax[0], "f1", "F1"), (ax[1], "auc", "AUC")]:
            yv = [per[c][key] * 100 for c in CLASSES]; cv = [cnn["per_class"][c][key] * 100 for c in CLASSES]
            my, mc = np.mean(yv), np.mean(cv)
            b1 = a.bar(x - w / 2, yv, w, color="#B9C3D0", label=f"YOLO old, per zone (mean {ttl} {my:.1f}%)")
            b2 = a.bar(x + w / 2, cv, w, color="#2E9E6B", label=f"Zone CNN EfficientNet-B0 (mean {ttl} {mc:.1f}%)")
            a.bar_label(b1, fmt="%.0f", fontsize=8, padding=2); a.bar_label(b2, fmt="%.0f", fontsize=8, padding=2)
            a.set_xticks(x, CLASSES, rotation=15); a.set_ylim(0, 105); a.set_title(f"{ttl} per class — same human-checked test")
            a.spines[["top", "right"]].set_visible(False); a.legend(frameon=False, loc="upper left", fontsize=8)
            if key == "auc":
                a.axhline(50, ls=":", c="#999", lw=1)
        fig.tight_layout(); fig.savefig(out / "yolo_vs_cnn.png", dpi=150)


if __name__ == "__main__":
    main()
