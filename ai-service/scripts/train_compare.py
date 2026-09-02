"""
train_compare.py — Run and compare multiple YOLOv8 training configurations
for the Skinglow project, per the improvement plan's section 7.

Configs compared:
  A. yolov8n baseline (matches your existing 100-epoch run, for reference)
  B. yolov8s (larger model, same dataset)
  C. yolov8s + improved dataset (point --data-c at your cleaned/expanded data.yaml)
  D. yolov8s + improved dataset + tuned augmentation (heavier aug for the
     weak/rare classes: oiliness, eyebag, wrinkle)

This script does NOT decide "which config wins" for you by mAP50 alone —
it dumps every metric requested (precision, recall, mAP50, mAP50-95,
per-class AP) into one comparison table so you make that call with full
information, per the instruction not to cherry-pick a single metric.

Requires a GPU. Each config can take anywhere from ~20min to a few hours
depending on dataset size — this is NOT something to run inside a chat
sandbox; run it on your RTX 3070 machine.

Usage:
    python train_compare.py --data-a data.yaml --data-c improved/data.yaml
    # add --skip A,B to only re-run C and D once you've already run the others once
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


CONFIGS = {
    "A": {
        "desc": "yolov8n baseline",
        "model": "yolov8n.pt",
        "aug": "default",
    },
    "B": {
        "desc": "yolov8s, same dataset",
        "model": "yolov8s.pt",
        "aug": "default",
    },
    "C": {
        "desc": "yolov8s + improved dataset",
        "model": "yolov8s.pt",
        "aug": "default",
        "use_data_c": True,
    },
    "D": {
        "desc": "yolov8s + improved dataset + tuned augmentation",
        "model": "yolov8s.pt",
        "aug": "tuned",
        "use_data_c": True,
    },
}

# Default augmentation (ultralytics defaults, roughly)
AUG_DEFAULT = dict(fliplr=0.5, flipud=0.0, degrees=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mosaic=1.0)

# Tuned for face photos with rare/subtle classes (oiliness, eyebag, wrinkle):
# - conservative geometric aug (faces have left/right symmetry but not much else)
# - slightly reduced HSV jitter so subtle color cues (redness, oiliness sheen) aren't washed out
# - lower mosaic probability late in training so the model sees more "whole face" context,
#   since these conditions are defined relative to facial position (T-zone, under-eye, etc.)
AUG_TUNED = dict(fliplr=0.5, flipud=0.0, degrees=5.0, hsv_h=0.01, hsv_s=0.5, hsv_v=0.3, mosaic=0.5, close_mosaic=15)


def run_config(key, cfg, data_a, data_c, epochs, batch, imgsz, patience, project_dir):
    print(f"\n{'='*60}\nRunning config {key}: {cfg['desc']}\n{'='*60}")

    data_path = data_c if cfg.get("use_data_c") else data_a
    aug = AUG_TUNED if cfg["aug"] == "tuned" else AUG_DEFAULT

    model = YOLO(cfg["model"])
    run_name = f"config_{key}"

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=project_dir,
        name=run_name,
        **aug,
    )

    metrics = model.val(data=data_path, project=project_dir, name=f"{run_name}_val")

    return {
        "config": key,
        "description": cfg["desc"],
        "model": cfg["model"],
        "data": str(data_path),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50": {
            name: float(ap) for name, ap in zip(model.names.values(), metrics.box.ap50)
        },
        "run_dir": str(Path(project_dir) / run_name),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare YOLOv8 training configurations for Skinglow")
    parser.add_argument("--data-a", type=str, default="data.yaml", help="data.yaml for configs A and B (current dataset)")
    parser.add_argument("--data-c", type=str, default=None, help="data.yaml for configs C and D (improved/expanded dataset). Required if running C or D.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--project", type=str, default="runs/compare")
    parser.add_argument("--only", type=str, default="A,B,C,D", help="Comma-separated configs to run, e.g. --only C,D")
    args = parser.parse_args()

    run_keys = [k.strip().upper() for k in args.only.split(",")]
    for k in run_keys:
        if CONFIGS[k].get("use_data_c") and not args.data_c:
            raise SystemExit(f"Config {k} requires --data-c (improved dataset path)")

    results = []
    for key in run_keys:
        cfg = CONFIGS[key]
        result = run_config(key, cfg, args.data_a, args.data_c, args.epochs, args.batch, args.imgsz, args.patience, args.project)
        results.append(result)

        # write incrementally so a crash on config D doesn't lose A/B/C results
        with open(Path(args.project) / "comparison_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # ---------------- comparison table ----------------
    print(f"\n\n{'='*70}\nCOMPARISON SUMMARY\n{'='*70}")
    header = f"{'Config':<8}{'Precision':>12}{'Recall':>10}{'mAP50':>10}{'mAP50-95':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['config']:<8}{r['precision']:>12.3f}{r['recall']:>10.3f}{r['map50']:>10.3f}{r['map50_95']:>12.3f}")

    print("\nPer-class mAP50:")
    class_names = list(results[0]["per_class_map50"].keys())
    header2 = f"{'Class':<14}" + "".join(f"{r['config']:>10}" for r in results)
    print(header2)
    for cls in class_names:
        row = f"{cls:<14}" + "".join(f"{r['per_class_map50'].get(cls, 0):>10.3f}" for r in results)
        print(row)

    print(f"\nFull results + per-run artifacts (confusion matrix, PR curve, F1 curve, "
          f"results.png) are in {args.project}/config_*/")
    print("Pick a config using the full table above, not mAP50 alone — check whether "
          "gains are concentrated in already-strong classes (Acne, Black Spot, Redness) "
          "or actually help the weak ones (Oilness, Wrinkle) before deciding.")


if __name__ == "__main__":
    main()
