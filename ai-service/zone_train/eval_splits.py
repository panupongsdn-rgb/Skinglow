"""
eval_splits.py — Train / Validation / Test accuracy ของโมเดลตัวเดียวกัน
=====================================================================
Scores one checkpoint (default: the deployed retuned best.pt) on all three
splits with the same thresholds and the same metric code as training.

Train has only automatic labels, so every split is scored twice when possible:
  auto  = automatic zone labels from the dataset builder (all splits)
  human = human-reviewed labels (valid / test only)

    python eval_splits.py --ckpt runs_zone\\<run>\\retuned\\best.pt --data zone_dataset ^
        --valid-reviews zone_reviews.jsonl --test-reviews review_compare\\zone_reviews_final.jsonl

Writes <ckpt folder>\\splits\\: accuracy_splits.json, accuracy_splits.md, accuracy_splits.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="zone_dataset")
    ap.add_argument("--valid-reviews", default="zone_reviews.jsonl")
    ap.add_argument("--test-reviews", default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    import train_zone_classifier as tz
    from zone_model import ZoneClassifier

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZoneClassifier(ck["arch"], pretrained=False); model.load_state_dict(ck["state_dict"]); model.to(device)
    img, th = ck.get("img_size") or 224, ck["thresholds"]
    data = Path(args.data)
    out = Path(args.ckpt).parent / "splits"; out.mkdir(parents=True, exist_ok=True)

    auto_rows = tz.read_rows(data, None)
    review_files = {"valid": args.valid_reviews, "test": args.test_reviews or args.valid_reviews}
    results = {}
    for s in ("train", "valid", "test"):
        rows = [r for r in auto_rows if r["split"] == s]
        print(f"{s}: {len(rows)} crops ...", flush=True)
        dl = DataLoader(tz.ZoneCrops(rows, data, tz.transforms(img, False)), batch_size=args.batch,
                        num_workers=args.workers)
        Y, P, Z = tz.predict(model, dl, device, device.type == "cuda")
        results[f"{s}_auto"] = tz.evaluate(Y, P, Z, th, tz.FACE_ZONES)
        if s != "train" and review_files[s] and Path(review_files[s]).exists():
            revs = tz.load_reviews(Path(review_files[s]))
            hrows = {r["crop"]: r for r in tz.read_rows(data, revs) if r["split"] == s and r["reviewed"]}
            idx = [i for i, r in enumerate(rows) if r["crop"] in hrows]
            if idx:
                Yh = np.stack([hrows[rows[i]["crop"]]["y"] for i in idx])
                results[f"{s}_human"] = tz.evaluate(Yh, P[idx], Z[idx], th, tz.FACE_ZONES)

    (out / "accuracy_splits.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    f = lambda v: f"{v * 100:.1f}%"
    L = ["# Train / Validation / Test accuracy", "", f"checkpoint: `{args.ckpt}` (epoch {ck.get('epoch')})", "",
         "| ชุดข้อมูล | ป้ายกำกับ | Crops | Accuracy | Balanced acc | Macro-F1 |", "|---|---|---:|---:|---:|---:|"]
    for k, r in results.items():
        s, lab = k.split("_"); o = r["overall"]
        L.append(f"| {s} | {'อัตโนมัติ' if lab == 'auto' else 'คนตรวจ'} | {o['crops']} | {f(o['micro_accuracy'])} "
                 f"| {f(o['macro_balanced_accuracy'])} | {f(o['macro_f1'])} |")
    (out / "accuracy_splits.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = list(results); x = np.arange(len(keys)); w = 0.27
    labels = [k.replace("_auto", "\n(auto labels)").replace("_human", "\n(human labels)") for k in keys]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (m, name, col) in enumerate([("micro_accuracy", "Accuracy", "#2E6FB0"),
                                        ("macro_balanced_accuracy", "Balanced accuracy", "#2E9E6B"),
                                        ("macro_f1", "Macro-F1", "#E08A2E")]):
        v = [results[k]["overall"][m] * 100 for k in keys]
        b = ax.bar(x + (i - 1) * w, v, w, label=name, color=col); ax.bar_label(b, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, labels); ax.set_ylim(0, 105); ax.set_ylabel("%")
    ax.set_title("Zone CNN (EfficientNet-B0) — train / valid / test")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "accuracy_splits.png", dpi=150)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
