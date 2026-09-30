"""
train_zone_classifier.py — Train + evaluate the per-zone skin classifier
=========================================================================
Input : the folder written by build_zone_dataset.py (labels.csv + crops)
Output: runs_zone/<name>/
          best.pt              weights + per-class thresholds (copy to ai-service/models/zone_classifier.pt)
          history.csv          loss / valid F1 per epoch
          metrics_valid.json   metrics on the validation split (thresholds tuned here)
          metrics_test.json    metrics on the untouched test split  <- report THESE in the thesis
          report_test.md       the same, as tables (Thai + English)
          curves.png           training curves (if matplotlib is installed)
          confusion_/per_class_/per_zone_/roc_/pr_<split>.png, valid_vs_test.png   result figures

Metrics (all computed only where the label is known, see build_zone_dataset.py):
  per class : Accuracy, Precision, Recall, F1, Balanced accuracy, AUC, TP/FP/TN/FN
  overall   : Macro-F1 (mean of the 6 class F1s)  <- the honest headline number
              Macro balanced accuracy, micro Accuracy
  per zone  : Accuracy / Macro-F1 for forehead, cheeks, nose, under-eye, chin
Plain Accuracy is always high when most zones are clear (a model that says
"no acne" everywhere scores well), so read it next to F1 and balanced accuracy.

The decision threshold of each class is tuned on VALID to maximise F1 and
then frozen for TEST, so the test numbers are not tuned on the test set.

Human-checked labels from review_tool.py (--reviews zone_reviews.jsonl)
replace the automatic labels of every reviewed image: all conditions in all
its zones become known. Images marked "unusable" are dropped. With
--eval-reviewed-only, valid/test are scored ONLY on reviewed images — use
that for the numbers in the thesis once the test set has been reviewed.

Typical run on an RTX 3070 (8 GB):
    python train_zone_classifier.py --data zone_dataset --arch efficientnet_b0 --epochs 40 --batch 64
    python train_zone_classifier.py --data zone_dataset --reviews zone_reviews.jsonl --eval-reviewed-only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from face_zones import CLASS_TH, ZONE_TH  # noqa: E402
from zone_model import CLASSES, MEAN, STD, ZONES, ZoneClassifier, allowed_matrix  # noqa: E402

FACE_ZONES = [z for z in ZONES if z != "patch"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_reviews(path: Path):
    """Latest review per image from review_tool.py's append-only jsonl."""
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                latest[r["image"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return latest


def read_rows(data_dir: Path, reviews: dict | None = None):
    with open(data_dir / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    allowed = allowed_matrix().numpy()
    reviews = reviews or {}
    out = []
    for r in rows:
        zi = ZONES.index(r["zone"])
        y = np.array([int(r[c]) for c in CLASSES], dtype=np.float32)
        rev = reviews.get(r["image"])
        r["reviewed"] = False
        if rev is not None:
            if rev.get("status") == "bad":
                continue  # a person marked the image unusable
            zl = rev.get("labels", {}).get(r["zone"])
            if zl is not None:
                for j, c in enumerate(CLASSES):
                    if c in zl:
                        y[j] = float(zl[c])
                r["reviewed"] = True
        y[allowed[zi] == 0] = -1  # never learn / score impossible pairs
        r["y"], r["zi"] = y, zi
        out.append(r)
    return out


class ZoneCrops(Dataset):
    def __init__(self, rows, data_dir: Path, tf):
        self.rows, self.dir, self.tf = rows, data_dir, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.dir / r["crop"]).convert("RGB")
        return self.tf(img), r["zi"], torch.from_numpy(r["y"])


def transforms(img: int, train: bool):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if not train:
        return T.Compose([T.Resize((img, img))] + norm)
    return T.Compose([
        T.RandomResizedCrop(img, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(10)], p=0.5),
        # modest colour jitter: redness / spots ARE colour, don't wash them out
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01),
    ] + norm)


def sample_weights(rows):
    """Oversample crops that contain rare classes (redness, eyebag, acne)."""
    Y = np.stack([r["y"] for r in rows])
    pos = np.maximum((Y == 1).sum(0), 1)
    boost = np.minimum(10.0, np.sqrt(pos.max() / pos))
    w = np.ones(len(rows))
    for i, y in enumerate(Y):
        if (y == 1).any():
            w[i] = max(1.0, boost[y == 1].max())
    return w, pos


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def auc_score(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allp = np.concatenate([pos, neg])
    for v in np.unique(allp):
        m = allp == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def binary_stats(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": n, "pos": tp + fn, "neg": tn + fp, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": (tp + tn) / n if n else float("nan"), "precision": prec, "recall": rec,
            "specificity": spec, "f1": f1, "balanced_accuracy": (rec + spec) / 2}


def tune_thresholds(Y, P):
    th = {}
    for j, c in enumerate(CLASSES):
        m = Y[:, j] >= 0
        y, p = Y[m, j], P[m, j]
        best, best_t = -1.0, 0.5
        if (y == 1).any():
            for t in np.arange(0.05, 0.951, 0.01):
                f = binary_stats(y, (p >= t).astype(int))["f1"]
                if f > best:
                    best, best_t = f, float(t)
        th[c] = round(best_t, 2)
    return th


def evaluate(Y, P, Z, th, zones_filter=None):
    """Y: labels (-1/0/1), P: probs, Z: zone idx. Metrics on known labels only."""
    keep = np.ones(len(Y), bool) if zones_filter is None else np.isin(Z, [ZONES.index(z) for z in zones_filter])
    Y, P, Z = Y[keep], P[keep], Z[keep]
    pred = np.stack([(P[:, j] >= th[c]).astype(int) for j, c in enumerate(CLASSES)], 1)
    per_class = {}
    for j, c in enumerate(CLASSES):
        m = Y[:, j] >= 0
        s = binary_stats(Y[m, j], pred[m, j])
        s["auc"] = auc_score(Y[m, j], P[m, j])
        s["threshold"] = th[c]
        per_class[c] = s
    scored = [c for c in CLASSES if per_class[c]["pos"] > 0 and per_class[c]["neg"] > 0]
    known = Y >= 0
    overall = {
        "macro_f1": float(np.mean([per_class[c]["f1"] for c in scored])) if scored else float("nan"),
        "macro_balanced_accuracy": float(np.mean([per_class[c]["balanced_accuracy"] for c in scored])) if scored else float("nan"),
        "macro_accuracy": float(np.mean([per_class[c]["accuracy"] for c in scored])) if scored else float("nan"),
        "micro_accuracy": float((pred[known] == Y[known]).mean()) if known.any() else float("nan"),
        "classes_scored": scored, "crops": int(len(Y)), "known_labels": int(known.sum()),
    }
    per_zone = {}
    for z in (zones_filter or ZONES):
        zm = Z == ZONES.index(z)
        if not zm.any():
            continue
        f1s, accs = [], []
        for j, c in enumerate(CLASSES):
            m = zm & (Y[:, j] >= 0)
            if (Y[m, j] == 1).any() and (Y[m, j] == 0).any():
                s = binary_stats(Y[m, j], pred[m, j]); f1s.append(s["f1"]); accs.append(s["accuracy"])
        k = zm[:, None] & known
        per_zone[z] = {"crops": int(zm.sum()),
                       "accuracy": float((pred[k] == Y[k]).mean()) if k.any() else float("nan"),
                       "macro_f1": float(np.mean(f1s)) if f1s else float("nan")}
    return {"overall": overall, "per_class": per_class, "per_zone": per_zone}


# ----------------------------------------------------------------------------
# Train / predict
# ----------------------------------------------------------------------------
def masked_bce(logits, y, pos_weight):
    mask = (y >= 0).float()
    target = y.clamp(min=0)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    Ps, Ys, Zs = [], [], []
    for x, zi, y in loader:
        x, zi = x.to(device, non_blocking=True), zi.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x, zi)
        Ps.append(torch.sigmoid(logits.float()).cpu().numpy()); Ys.append(y.numpy()); Zs.append(zi.cpu().numpy())
    return np.concatenate(Ys), np.concatenate(Ps), np.concatenate(Zs)


def fmt(v):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.1f}%"


def write_report(path: Path, res: dict, res_patch: dict | None, split: str, args, target: float):
    o = res["overall"]
    L = [f"# Skinglow zone classifier — {split} results", "",
         f"- Model: `{args.arch}` · image {args.img}px · data `{args.data}`",
         f"- Crops evaluated (face zones): {o['crops']} · known labels: {o['known_labels']}",
         f"- Thresholds: tuned on **valid** for best F1, frozen for {split}",
         f"- Labels: {'**human-reviewed images only** (review_tool.py)' if split in getattr(args, 'reviewed_only_splits', ()) else 'automatic zone labels' + (' + reviews' if args.reviews else '')}", "",
         "## ผลรวม / Overall", "",
         "| Metric | Value |", "|---|---|",
         f"| **Macro-F1** (ค่าเฉลี่ย F1 ของ {len(o['classes_scored'])} คลาส) | **{fmt(o['macro_f1'])}** |",
         f"| Macro balanced accuracy | {fmt(o['macro_balanced_accuracy'])} |",
         f"| Macro accuracy | {fmt(o['macro_accuracy'])} |",
         f"| Micro accuracy (all known labels) | {fmt(o['micro_accuracy'])} |", "",
         f"Target {target * 100:.0f}% Macro-F1: **{'reached' if o['macro_f1'] >= target else 'not reached'}**", "",
         "## รายคลาส / Per class", "",
         "| Class | ไทย | Pos | Neg | Accuracy | Precision | Recall | F1 | Bal. acc | AUC | Thr |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, s in res["per_class"].items():
        L.append(f"| {c} | {CLASS_TH[c]} | {s['pos']} | {s['neg']} | {fmt(s['accuracy'])} | {fmt(s['precision'])} | "
                 f"{fmt(s['recall'])} | **{fmt(s['f1'])}** | {fmt(s['balanced_accuracy'])} | {fmt(s['auc'])} | {s['threshold']:.2f} |")
    L += ["", "## รายโซน / Per zone", "", "| Zone | ไทย | Crops | Accuracy | Macro-F1 |", "|---|---|---|---|---|"]
    for z, s in res["per_zone"].items():
        L.append(f"| {z} | {ZONE_TH.get(z, z)} | {s['crops']} | {fmt(s['accuracy'])} | {fmt(s['macro_f1'])} |")
    L += ["", "## Confusion (TP / FP / TN / FN)", "", "| Class | TP | FP | TN | FN |", "|---|---|---|---|---|"]
    for c, s in res["per_class"].items():
        L.append(f"| {c} | {s['TP']} | {s['FP']} | {s['TN']} | {s['FN']} |")
    if res_patch and res_patch["overall"]["crops"]:
        po = res_patch["overall"]
        L += ["", "## Close-up patches (reported separately)", "",
              f"Crops {po['crops']} · Macro-F1 {fmt(po['macro_f1'])} · Macro balanced accuracy {fmt(po['macro_balanced_accuracy'])}"]
    L += ["", "Labels marked unknown (-1) — classes a source dataset never annotated, boxes that only graze a zone, "
              "unannotated images — are excluded from every number above.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="zone_dataset")
    ap.add_argument("--arch", default="efficientnet_b0",
                    choices=["efficientnet_b0", "efficientnet_b2", "mobilenet_v3_large", "resnet50"])
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8, help="stop after N epochs without a better valid Macro-F1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (debug only; much worse)")
    ap.add_argument("--no-sampler", action="store_true", help="disable rare-class oversampling")
    ap.add_argument("--no-patches", action="store_true", help="train on face zones only")
    ap.add_argument("--reviews", default=None, help="zone_reviews.jsonl from review_tool.py")
    ap.add_argument("--eval-reviewed-only", action="store_true",
                    help="tune thresholds and report metrics only on human-reviewed valid/test images")
    ap.add_argument("--target", type=float, default=0.80, help="Macro-F1 goal printed in the report")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="runs_zone")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after N batches")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    amp = device.type == "cuda"
    data_dir = Path(args.data)
    run = Path(args.out) / (args.name or f"{args.arch}_{time.strftime('%Y%m%d_%H%M')}")
    run.mkdir(parents=True, exist_ok=True)

    reviews = load_reviews(Path(args.reviews)) if args.reviews else {}
    rows = read_rows(data_dir, reviews)
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}
    reviewed_counts = {s: sum(r["reviewed"] for r in v) for s, v in split.items()}
    if args.reviews:
        print(f"reviews: {len(reviews)} images · reviewed crops " +
              ", ".join(f"{s}={n}" for s, n in reviewed_counts.items()))
    args.reviewed_only_splits = set()
    if args.eval_reviewed_only:
        for s in ("valid", "test"):
            only = [r for r in split[s] if r["reviewed"]]
            if only:
                split[s] = only
                args.reviewed_only_splits.add(s)
            else:
                print(f"WARNING: no reviewed {s} crops yet — evaluating {s} on all crops")
    if args.no_patches:
        split["train"] = [r for r in split["train"] if r["zone"] != "patch"]
    print(f"device={device}  crops: " + ", ".join(f"{s}={len(v)}" for s, v in split.items()))

    w, pos = sample_weights(split["train"])
    Ytr = np.stack([r["y"] for r in split["train"]])
    neg = (Ytr == 0).sum(0)
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 20.0), dtype=torch.float32, device=device)
    print("train positives per class:", dict(zip(CLASSES, pos.tolist())))
    print("pos_weight:", dict(zip(CLASSES, [round(v, 2) for v in pos_weight.tolist()])))

    ds_tr = ZoneCrops(split["train"], data_dir, transforms(args.img, True))
    sampler = None if args.no_sampler else WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
                       num_workers=args.workers, pin_memory=amp, drop_last=len(ds_tr) > args.batch,
                       persistent_workers=args.workers > 0)
    dl = {s: DataLoader(ZoneCrops(split[s], data_dir, transforms(args.img, False)), batch_size=args.batch * 2,
                        num_workers=args.workers, pin_memory=amp) for s in ("valid", "test")}

    model = ZoneClassifier(args.arch, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": list(model.head.parameters()) + list(model.zone_emb.parameters()), "lr": args.lr * 5},
    ], weight_decay=args.weight_decay)
    steps_per_epoch = args.max_steps or len(dl_tr)
    total = max(1, args.epochs * steps_per_epoch)
    warm = min(steps_per_epoch, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_f1, best_epoch, history = -1.0, -1, []
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); run_loss = n = 0
        for step, (x, zi, y) in enumerate(dl_tr):
            if args.max_steps and step >= args.max_steps:
                break
            x, zi, y = x.to(device, non_blocking=True), zi.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x, zi)
            loss = masked_bce(logits.float(), y, pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item() * len(x); n += len(x)

        Yv, Pv, Zv = predict(model, dl["valid"], device, amp)
        th = tune_thresholds(Yv[np.isin(Zv, [ZONES.index(z) for z in FACE_ZONES])],
                             Pv[np.isin(Zv, [ZONES.index(z) for z in FACE_ZONES])])
        rv = evaluate(Yv, Pv, Zv, th, FACE_ZONES)
        f1 = rv["overall"]["macro_f1"]; f1 = -1.0 if math.isnan(f1) else f1
        history.append({"epoch": epoch, "train_loss": run_loss / max(1, n), "valid_macro_f1": f1,
                        "valid_macro_bal_acc": rv["overall"]["macro_balanced_accuracy"],
                        "valid_micro_acc": rv["overall"]["micro_accuracy"], "sec": round(time.time() - t0, 1)})
        flag = ""
        if f1 > best_f1:
            best_f1, best_epoch, flag = f1, epoch, "  * saved"
            torch.save({"arch": args.arch, "state_dict": model.state_dict(), "thresholds": th,
                        "classes": CLASSES, "zones": ZONES, "img_size": args.img, "epoch": epoch,
                        "valid_macro_f1": f1, "mean": MEAN, "std": STD}, run / "best.pt")
        print(f"epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  valid Macro-F1 {fmt(f1)}  "
              f"bal-acc {fmt(rv['overall']['macro_balanced_accuracy'])}  ({history[-1]['sec']}s){flag}")
        if epoch - best_epoch >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    with open(run / "history.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(history[0].keys())); wr.writeheader(); wr.writerows(history)

    # ---- final evaluation with the best checkpoint ------------------------
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); th = ckpt["thresholds"]
    for s in ("valid", "test"):
        Y, P, Z = predict(model, dl[s], device, amp)
        res = evaluate(Y, P, Z, th, FACE_ZONES)
        res_patch = evaluate(Y, P, Z, th, ["patch"]) if (Z == ZONES.index("patch")).any() else None
        res["patch"] = res_patch["overall"] if res_patch else None
        res["best_epoch"] = ckpt["epoch"]
        np.savez_compressed(run / f"preds_{s}.npz", Y=Y, P=P, Z=Z,
                            face_idx=np.array([ZONES.index(z) for z in FACE_ZONES]))
        (run / f"metrics_{s}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(run / f"report_{s}.md", res, res_patch, s, args, args.target)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ep = [h["epoch"] for h in history]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(ep, [h["train_loss"] for h in history]); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
        ax[1].plot(ep, [h["valid_macro_f1"] for h in history], label="Macro-F1")
        ax[1].plot(ep, [h["valid_macro_bal_acc"] for h in history], label="Macro balanced acc")
        ax[1].axhline(args.target, ls="--", c="gray", lw=1); ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].set_title("valid")
        fig.tight_layout(); fig.savefig(run / "curves.png", dpi=120)
    except Exception as exc:
        print("curves.png skipped:", exc)
    try:
        from plot_results import plot_all
        print("figures:", ", ".join(plot_all(run, args.target)))
    except Exception as exc:
        print("result figures skipped:", exc)

    print("\n" + (run / "report_test.md").read_text(encoding="utf-8"))
    print(f"Run folder: {run}\nDeploy: copy {run / 'best.pt'} to ai-service/models/zone_classifier.pt")


if __name__ == "__main__":
    freeze_support()
    main()
