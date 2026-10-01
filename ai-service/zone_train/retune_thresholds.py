"""
retune_thresholds.py — ปรับ threshold ใหม่บน valid ที่คนตรวจแล้ว (ไม่ต้องเทรนใหม่)
==============================================================================
The per-class thresholds in best.pt were tuned on valid crops with automatic
labels. Once valid has been reviewed with review_tool.py, re-tune them on the
human labels and re-score test — no retraining, about a minute on a GPU.

    python retune_thresholds.py --run runs_zone\\<run> --data zone_dataset ^
        --reviews zone_reviews.jsonl --test-reviews review_compare\\zone_reviews_final.jsonl

--reviews       file with the reviewed VALID images (zone_reviews.jsonl)
--test-reviews  file with the final TEST labels (review_compare\\zone_reviews_final.jsonl)

Writes <run>\\retuned\\: best.pt (same weights, new thresholds — deploy this one),
metrics_valid.json, metrics_test.json, report_valid.md, report_test.md and all figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="zone_dataset")
    ap.add_argument("--reviews", default="zone_reviews.jsonl", help="reviewed valid images")
    ap.add_argument("--test-reviews", default=None, help="final test labels (defaults to --reviews)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--target", type=float, default=0.80)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    import train_zone_classifier as tz
    from zone_model import ZONES, ZoneClassifier
    from plot_results import plot_all

    run = Path(args.run); out = run / "retuned"; out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZoneClassifier(ck["arch"], pretrained=False); model.load_state_dict(ck["state_dict"]); model.to(device)
    img = ck.get("img_size") or 224
    face_idx = np.array([ZONES.index(z) for z in tz.FACE_ZONES])
    review_files = {"valid": args.reviews, "test": args.test_reviews or args.reviews}

    preds = {}
    for s in ("valid", "test"):
        revs = tz.load_reviews(Path(review_files[s]))
        rows = [r for r in tz.read_rows(data, revs) if r["split"] == s and r["reviewed"]]
        if not rows:
            sys.exit(f"no reviewed {s} crops found in {review_files[s]} — review {s} first")
        dl = DataLoader(tz.ZoneCrops(rows, data, tz.transforms(img, False)), batch_size=args.batch, num_workers=args.workers)
        Y, P, Z = tz.predict(model, dl, device, device.type == "cuda")
        preds[s] = (Y, P, Z)
        np.savez_compressed(out / f"preds_{s}.npz", Y=Y, P=P, Z=Z, face_idx=face_idx)
        print(f"{s}: {len(rows)} reviewed crops")

    Yv, Pv, Zv = preds["valid"]
    fm = np.isin(Zv, face_idx)
    old_th = ck["thresholds"]
    new_th = tz.tune_thresholds(Yv[fm], Pv[fm])
    rep_args = SimpleNamespace(arch=ck["arch"], img=img, data=str(data), reviews=review_files["test"],
                               eval_reviewed_only=True, reviewed_only_splits={"valid", "test"})
    results = {}
    for s, (Y, P, Z) in preds.items():
        res = tz.evaluate(Y, P, Z, new_th, tz.FACE_ZONES)
        res["best_epoch"] = ck.get("epoch"); res["thresholds_tuned_on"] = "human-reviewed valid"
        (out / f"metrics_{s}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        tz.write_report(out / f"report_{s}.md", res, None, s, rep_args, args.target)
        results[s] = res
    old_test = tz.evaluate(*preds["test"], old_th, tz.FACE_ZONES)

    ck2 = dict(ck); ck2["thresholds"] = new_th; ck2["thresholds_tuned_on"] = "human-reviewed valid"
    torch.save(ck2, out / "best.pt")
    figs = plot_all(out, args.target)

    print("\nclass        old thr -> new thr   test F1 old -> new")
    for c in tz.CLASSES:
        print(f"{c:11s}  {old_th[c]:.2f} -> {new_th[c]:.2f}        "
              f"{old_test['per_class'][c]['f1'] * 100:5.1f}% -> {results['test']['per_class'][c]['f1'] * 100:5.1f}%")
    print(f"Macro-F1 test: {old_test['overall']['macro_f1'] * 100:.1f}% -> {results['test']['overall']['macro_f1'] * 100:.1f}%"
          f"   (valid {results['valid']['overall']['macro_f1'] * 100:.1f}%)")
    print("figures:", ", ".join(figs))
    print(f"\nDeploy: copy {out / 'best.pt'} to ai-service/models/zone_classifier.pt")


if __name__ == "__main__":
    main()
