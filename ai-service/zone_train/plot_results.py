"""
plot_results.py — กราฟผลการทดลองสำหรับใส่เล่ม
==============================================
Draws the evaluation figures of a zone-classifier run next to curves.png:

  confusion_<split>.png   2x2 confusion matrix per class (counts + % of the true row)
  confusion_overall_<split>.png   all classes pooled into one 2x2 + TP/FN/FP/TN table per class
  confusion_matrix[_normalized]_<split>.png   YOLO-style predicted x true (+ background) *
  per_class_<split>.png   Precision / Recall / F1 per class, with the target line
  per_zone_<split>.png    Accuracy and Macro-F1 per face zone
  roc_<split>.png         ROC curve per class (AUC in the legend)          *
  pr_<split>.png          Precision-Recall curve per class, threshold dot  *
  valid_vs_test.png       F1 per class on valid vs test
  (* needs the model's probabilities: preds_<split>.npz)

train_zone_classifier.py calls this automatically at the end of training.
For a run that finished before this file existed, point it at the run and the
dataset; it reloads best.pt and predicts valid/test once (GPU if available):

    python plot_results.py --run runs_zone\\efficientnet_b0_20261001_0320 --data zone_dataset ^
        --reviews review_compare\\zone_reviews_final.jsonl --eval-reviewed-only

Use the same --reviews / --eval-reviewed-only as the training run so the
figures describe exactly the images in report_test.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

CLASSES = ["acne", "black_spot", "eyebag", "oiliness", "redness", "wrinkle"]
LABEL = {"acne": "Acne", "black_spot": "Dark spot", "eyebag": "Eye bag",
         "oiliness": "Oiliness", "redness": "Redness", "wrinkle": "Wrinkle"}
ZONE_LABEL = {"forehead": "Forehead", "left_cheek": "Left cheek", "right_cheek": "Right cheek",
              "nose": "Nose", "under_eye": "Under eye", "chin": "Chin"}
C_P, C_R, C_F, C_A = "#2F6DB5", "#E0913A", "#2E9E6B", "#8A8F98"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.dpi": 100})
    return plt


def _title(split, m):
    o = m["overall"]
    return (f"{split} · Macro-F1 {o['macro_f1'] * 100:.1f}% · balanced acc "
            f"{o['macro_balanced_accuracy'] * 100:.1f}% · {o['crops']} zone crops")


def confusion(m, split, out: Path):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, c in zip(axes.flat, CLASSES):
        s = m["per_class"][c]
        mat = np.array([[s["TN"], s["FP"]], [s["FN"], s["TP"]]], float)
        rows = mat.sum(1, keepdims=True)
        pct = np.divide(mat, rows, out=np.zeros_like(mat), where=rows > 0)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(mat[i, j])}\n{pct[i, j] * 100:.1f}%", ha="center", va="center",
                        color="white" if pct[i, j] > 0.55 else "#1d2733", fontsize=11)
        ax.set_xticks([0, 1], ["Pred: no", "Pred: yes"]); ax.set_yticks([0, 1], ["True: no", "True: yes"])
        ax.set_title(f"{LABEL[c]}  (F1 {s['f1'] * 100:.1f}%, thr {s['threshold']:.2f})", fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Confusion matrix per class — " + _title(split, m), fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"confusion_{split}.png", dpi=150); plt.close(fig)


def confusion_overall(m, split, out: Path):
    """One figure for the whole model: all classes pooled into one 2x2 (micro)
    next to a table of every class's TP / FN / FP / TN."""
    plt = _plt()
    pc = m["per_class"]
    tot = {k: sum(pc[c][k] for c in CLASSES) for k in ("TP", "FP", "TN", "FN")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    mat = np.array([[tot["TN"], tot["FP"]], [tot["FN"], tot["TP"]]], float)
    rows = mat.sum(1, keepdims=True); pct = mat / np.maximum(rows, 1)
    a1.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a1.text(j, i, f"{int(mat[i, j])}\n{pct[i, j] * 100:.1f}%", ha="center", va="center",
                    color="white" if pct[i, j] > 0.55 else "#1d2733", fontsize=13)
    a1.set_xticks([0, 1], ["Predicted: no", "Predicted: yes"]); a1.set_yticks([0, 1], ["True: no", "True: yes"])
    prec = tot["TP"] / max(tot["TP"] + tot["FP"], 1); rec = tot["TP"] / max(tot["TP"] + tot["FN"], 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9); acc = (tot["TP"] + tot["TN"]) / max(sum(tot.values()), 1)
    a1.set_title(f"All classes pooled (micro)\nacc {acc * 100:.1f}% · precision {prec * 100:.1f}% · "
                 f"recall {rec * 100:.1f}% · F1 {f1 * 100:.1f}%", fontsize=10)
    for sp in a1.spines.values():
        sp.set_visible(False)
    # per-class table, each cell coloured by its share of that class's true row
    cols = ["TP", "FN", "FP", "TN"]
    data = np.array([[pc[c][k] for k in cols] for c in CLASSES], float)
    pos = data[:, 0] + data[:, 1]; neg = data[:, 2] + data[:, 3]
    share = np.c_[data[:, 0] / np.maximum(pos, 1), data[:, 1] / np.maximum(pos, 1),
                  data[:, 2] / np.maximum(neg, 1), data[:, 3] / np.maximum(neg, 1)]
    good = np.c_[share[:, 0], 1 - share[:, 1], 1 - share[:, 2], share[:, 3]]
    a2.imshow(good, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(CLASSES)):
        for j in range(4):
            a2.text(j, i, f"{int(data[i, j])}\n{share[i, j] * 100:.1f}%", ha="center", va="center", fontsize=9)
    a2.set_xticks(range(4), ["TP\n(true yes → yes)", "FN\n(true yes → no)", "FP\n(true no → yes)", "TN\n(true no → no)"])
    a2.set_yticks(range(len(CLASSES)), [f"{LABEL[c]}  F1 {pc[c]['f1'] * 100:.0f}%" for c in CLASSES])
    a2.set_title("Per class (% of that class's true yes / true no)", fontsize=10)
    for sp in a2.spines.values():
        sp.set_visible(False)
    fig.suptitle("Overall confusion — " + _title(split, m), fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"confusion_overall_{split}.png", dpi=150); plt.close(fig)


def confusion_multilabel(Y, P, Z, face_idx, th, split, out: Path):
    """YOLO-style confusion matrix (rows = predicted, columns = true, plus "background").
    A zone can have several problems, so each zone is matched like YOLO matches boxes:
      true & predicted            -> diagonal
      missed true + extra predicted in the same zone -> paired off-diagonal (a mix-up)
      missed true left over       -> predicted "background" (model saw nothing)
      extra predicted left over   -> true "background" (false alarm on clear skin)"""
    plt = _plt()
    keep = np.isin(Z, face_idx); Y, P = Y[keep], P[keep]
    pred = np.stack([(P[:, j] >= th[c]).astype(int) for j, c in enumerate(CLASSES)], 1)
    known = Y >= 0
    k = len(CLASSES); B = k
    M = np.zeros((k + 1, k + 1), int)          # M[pred, true]
    for y, pr, kn in zip(Y, pred, known):
        t = {j for j in range(k) if kn[j] and y[j] == 1}
        q = {j for j in range(k) if kn[j] and pr[j] == 1}
        for j in t & q:
            M[j, j] += 1
        miss, extra = sorted(t - q), sorted(q - t)
        while miss and extra:
            M[extra.pop(0), miss.pop(0)] += 1
        for j in miss:
            M[B, j] += 1
        for j in extra:
            M[j, B] += 1
    names = CLASSES + ["background"]
    for norm in (False, True):
        V = M / np.maximum(M.sum(0, keepdims=True), 1) if norm else M
        fig, ax = plt.subplots(figsize=(10, 7.5))
        im = ax.imshow(np.where(M > 0, V, np.nan), cmap="Blues", vmin=0, vmax=1 if norm else max(M.max(), 1))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(k + 1):
            for j in range(k + 1):
                if M[i, j]:
                    txt = f"{V[i, j]:.2f}" if norm else f"{M[i, j]}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                            color="white" if (V[i, j] > (0.55 if norm else 0.55 * M.max())) else "#1d2733")
        ax.set_xticks(range(k + 1), names, rotation=90); ax.set_yticks(range(k + 1), names)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix" + (" Normalized" if norm else "") + f" — {split} (zones)")
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"confusion_matrix{'_normalized' if norm else ''}_{split}.png", dpi=150); plt.close(fig)


def per_class(m, split, out: Path, target: float):
    plt = _plt()
    x = np.arange(len(CLASSES)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, (key, col, name) in enumerate([("precision", C_P, "Precision"), ("recall", C_R, "Recall"), ("f1", C_F, "F1")]):
        v = [m["per_class"][c][key] * 100 for c in CLASSES]
        b = ax.bar(x + (k - 1) * w, v, w, color=col, label=name)
        if key == "f1":
            ax.bar_label(b, fmt="%.0f", fontsize=8, padding=2)
    ax.axhline(target * 100, ls="--", c="#555", lw=1, label=f"Target {target * 100:.0f}%")
    ax.set_xticks(x, [f"{LABEL[c]}\n(n+={m['per_class'][c]['pos']})" for c in CLASSES])
    ax.set_ylim(0, 105); ax.set_ylabel("%"); ax.legend(ncol=4, loc="upper left", frameon=False)
    ax.set_title("Per-class metrics — " + _title(split, m), fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"per_class_{split}.png", dpi=150); plt.close(fig)


def per_zone(m, split, out: Path, target: float):
    plt = _plt()
    zones = [z for z in ZONE_LABEL if z in m.get("per_zone", {})]
    if not zones:
        return
    x = np.arange(len(zones)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    acc = [m["per_zone"][z]["accuracy"] * 100 for z in zones]
    f1 = [m["per_zone"][z]["macro_f1"] * 100 for z in zones]
    b1 = ax.bar(x - w / 2, acc, w, color=C_A, label="Accuracy")
    b2 = ax.bar(x + w / 2, f1, w, color=C_F, label="Macro-F1")
    ax.bar_label(b1, fmt="%.0f", fontsize=8, padding=2); ax.bar_label(b2, fmt="%.0f", fontsize=8, padding=2)
    ax.axhline(target * 100, ls="--", c="#555", lw=1)
    ax.set_xticks(x, [f"{ZONE_LABEL[z]}\n({m['per_zone'][z]['crops']})" for z in zones])
    ax.set_ylim(0, 105); ax.set_ylabel("%"); ax.legend(frameon=False, loc="upper right")
    ax.set_title("Per-zone results — " + _title(split, m), fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"per_zone_{split}.png", dpi=150); plt.close(fig)


def _curves(Y, P):
    """ROC and PR points for one class (y in {0,1})."""
    order = np.argsort(-P, kind="mergesort")
    y = Y[order]; p = P[order]
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    P_, N_ = max(tp[-1], 1), max(fp[-1], 1)
    tpr, fpr = np.r_[0, tp / P_], np.r_[0, fp / N_]
    prec = tp / np.maximum(tp + fp, 1); rec = tp / P_
    return fpr, tpr, np.r_[1, prec], np.r_[0, rec], p


def roc_pr(Y, P, Z, face_idx, m, split, out: Path):
    plt = _plt()
    keep = np.isin(Z, face_idx)
    Y, P = Y[keep], P[keep]
    cols = plt.get_cmap("tab10").colors
    fig1, a1 = plt.subplots(figsize=(5.6, 5)); fig2, a2 = plt.subplots(figsize=(5.6, 5))
    for k, c in enumerate(CLASSES):
        msk = Y[:, k] >= 0
        y, p = Y[msk, k], P[msk, k]
        if not ((y == 1).any() and (y == 0).any()):
            continue
        fpr, tpr, prec, rec, _ = _curves(y, p)
        s = m["per_class"][c]
        a1.plot(fpr, tpr, color=cols[k], lw=1.6, label=f"{LABEL[c]} (AUC {s['auc'] * 100:.1f})")
        ap = float(np.sum(np.diff(rec) * prec[1:]))
        a2.plot(rec, prec, color=cols[k], lw=1.6, label=f"{LABEL[c]} (AP {ap * 100:.1f})")
        a2.plot(s["recall"], s["precision"], "o", color=cols[k], ms=6, mec="white")
        a1.plot(1 - s["specificity"], s["recall"], "o", color=cols[k], ms=6, mec="white")
    a1.plot([0, 1], [0, 1], ls=":", c="#999", lw=1)
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate (recall)")
    a1.set_title(f"ROC — {split} (dot = chosen threshold)", fontsize=10); a1.legend(fontsize=8, frameon=False, loc="lower right")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_ylim(0, 1.02)
    a2.set_title(f"Precision-Recall — {split} (dot = chosen threshold)", fontsize=10); a2.legend(fontsize=8, frameon=False, loc="upper right")
    fig1.tight_layout(); fig1.savefig(out / f"roc_{split}.png", dpi=150); plt.close(fig1)
    fig2.tight_layout(); fig2.savefig(out / f"pr_{split}.png", dpi=150); plt.close(fig2)


def valid_vs_test(mv, mt, out: Path, target: float):
    plt = _plt()
    x = np.arange(len(CLASSES)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    v = [mv["per_class"][c]["f1"] * 100 for c in CLASSES]; t = [mt["per_class"][c]["f1"] * 100 for c in CLASSES]
    b1 = ax.bar(x - w / 2, v, w, color="#B9C3D0", label=f"valid (Macro-F1 {mv['overall']['macro_f1'] * 100:.1f}%)")
    b2 = ax.bar(x + w / 2, t, w, color=C_F, label=f"test (Macro-F1 {mt['overall']['macro_f1'] * 100:.1f}%)")
    ax.bar_label(b1, fmt="%.0f", fontsize=8, padding=2); ax.bar_label(b2, fmt="%.0f", fontsize=8, padding=2)
    ax.axhline(target * 100, ls="--", c="#555", lw=1)
    ax.set_xticks(x, [LABEL[c] for c in CLASSES]); ax.set_ylim(0, 105); ax.set_ylabel("F1 %")
    ax.legend(frameon=False, loc="upper left"); ax.set_title("F1 per class: valid vs test", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "valid_vs_test.png", dpi=150); plt.close(fig)


def plot_all(run: Path, target: float = 0.80, face_idx=None):
    """Draw every figure that the files in `run` allow. Returns the list of files written."""
    run = Path(run)
    before = set(run.glob("*.png"))
    metrics = {}
    for s in ("valid", "test"):
        f = run / f"metrics_{s}.json"
        if f.exists():
            metrics[s] = json.loads(f.read_text(encoding="utf-8"))
    for s, m in metrics.items():
        confusion(m, s, run); confusion_overall(m, s, run)
        per_class(m, s, run, target); per_zone(m, s, run, target)
        npz = run / f"preds_{s}.npz"
        if npz.exists():
            d = np.load(npz)
            fi = face_idx if face_idx is not None else d["face_idx"]
            roc_pr(d["Y"], d["P"], d["Z"], fi, m, s, run)
            th = {c: m["per_class"][c]["threshold"] for c in CLASSES}
            confusion_multilabel(d["Y"], d["P"], d["Z"], fi, th, s, run)
    if "valid" in metrics and "test" in metrics:
        valid_vs_test(metrics["valid"], metrics["test"], run, target)
    return sorted(p.name for p in set(run.glob("*.png")) - before) or sorted(p.name for p in run.glob("*.png"))


def predict_run(run: Path, data: Path, reviews: str | None, reviewed_only: bool, batch: int, workers: int):
    """Reload best.pt and save preds_valid/test.npz with the same crops as training."""
    import torch
    from torch.utils.data import DataLoader
    import train_zone_classifier as tz
    from zone_model import ZONES, ZoneClassifier
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZoneClassifier(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.to(device)
    revs = tz.load_reviews(Path(reviews)) if reviews else {}
    rows = tz.read_rows(data, revs)
    img = ck.get("img_size") or 224
    face_idx = np.array([ZONES.index(z) for z in tz.FACE_ZONES])
    for s in ("valid", "test"):
        part = [r for r in rows if r["split"] == s]
        if reviewed_only and any(r["reviewed"] for r in part):
            part = [r for r in part if r["reviewed"]]
        dl = DataLoader(tz.ZoneCrops(part, data, tz.transforms(img, False)), batch_size=batch, num_workers=workers)
        Y, P, Z = tz.predict(model, dl, device, device.type == "cuda")
        np.savez_compressed(run / f"preds_{s}.npz", Y=Y, P=P, Z=Z, face_idx=face_idx)
        print(f"{s}: {len(part)} crops predicted")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="runs_zone/<run> folder")
    ap.add_argument("--data", default=None, help="zone_dataset (only needed to compute ROC/PR for older runs)")
    ap.add_argument("--reviews", default=None)
    ap.add_argument("--eval-reviewed-only", action="store_true")
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    run = Path(args.run)
    if args.data and not (run / "preds_test.npz").exists():
        predict_run(run, Path(args.data), args.reviews, args.eval_reviewed_only, args.batch, args.workers)
    files = plot_all(run, args.target)
    print("figures:", ", ".join(files))


if __name__ == "__main__":
    main()
