"""
compare_reviews.py — วัดความตรงกันของผู้ตรวจ 2 คน + รวมเป็น label สุดท้าย
=========================================================================
Compares two independent zone_reviews files (e.g. yours and a second
reviewer's) made with review_tool.py on the same images, and reports how
much the two reviewers agree:

  * per class: % agreement, Cohen's kappa, and each reviewer's positive count
  * per zone:  % agreement
  * images one reviewer marked "bad" (unusable) and the other didn't

Cohen's kappa corrects % agreement for chance: 1.0 = perfect, >0.8 almost
perfect, 0.6-0.8 substantial, 0.4-0.6 moderate, <0.4 weak (Landis & Koch).
A class with weak kappa means the label definition is unclear or the photos
can't show it; the model's score on that class can't be trusted beyond it.

Outputs (in --out, default review_compare/):
  agreement.md       readable report (Thai + numbers)
  agreement.json     same numbers, machine readable
  disagreements.csv  one row per image/zone/class where A and B differ
  zone_reviews_final.jsonl  (with --final) labels where both agree, plus
                     adjudicated answers for the cells they disagreed on

Typical flow
  1. both reviewers review the test set separately (never look at each other's file)
  2. python compare_reviews.py --a zone_reviews.jsonl --b zone_reviews_claude.jsonl
  3. settle the disagreements:
       python review_tool.py --zones zone_dataset --data-root dataset_clean_v2 ^
           --compare zone_reviews.jsonl zone_reviews_claude.jsonl --reviews zone_reviews_adjudicated.jsonl
  4. python compare_reviews.py --a ... --b ... --adjudicated zone_reviews_adjudicated.jsonl --final
  5. train / evaluate with --reviews review_compare/zone_reviews_final.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import face_zones as fz  # noqa: E402

CLASSES = fz.CLASSES


def load(path):
    """latest review per image (same rule as review_tool / training)."""
    latest = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                latest[r["image"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return latest


def kappa(n11, n10, n01, n00):
    n = n11 + n10 + n01 + n00
    if n == 0:
        return None
    po = (n11 + n00) / n
    pa1, pb1 = (n11 + n10) / n, (n11 + n01) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:  # both reviewers gave the same single answer everywhere
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def verdict(k):
    if k is None:
        return "-"
    return ("ตรงกันเกือบสมบูรณ์" if k > 0.8 else "ตรงกันมาก" if k > 0.6 else
            "ปานกลาง" if k > 0.4 else "พอใช้" if k > 0.2 else "ต่ำ")


def compare(a, b, split=None):
    common = sorted(i for i in a if i in b and (split is None or a[i].get("split") == split))
    table = {c: [0, 0, 0, 0] for c in CLASSES}          # n11 n10 n01 n00 (A,B)
    zone_agree = defaultdict(lambda: [0, 0])            # zone -> [agree, total]
    status_diff, rows = [], []
    for img in common:
        ra, rb = a[img], b[img]
        if ra["status"] != rb["status"]:
            status_diff.append({"image": img, "a": ra["status"], "b": rb["status"]})
            continue
        if ra["status"] != "ok":
            continue
        for z in sorted(set(ra["labels"]) & set(rb["labels"])):
            for c in CLASSES:
                if c not in ra["labels"][z] or c not in rb["labels"][z]:
                    continue
                va, vb = int(ra["labels"][z][c]), int(rb["labels"][z][c])
                table[c][{(1, 1): 0, (1, 0): 1, (0, 1): 2, (0, 0): 3}[(va, vb)]] += 1
                zone_agree[z][0] += va == vb
                zone_agree[z][1] += 1
                if va != vb:
                    rows.append({"image": img, "zone": z, "class": c, "a": va, "b": vb})
    per_class = {}
    tot = [0, 0, 0, 0]
    for c, t in table.items():
        n = sum(t)
        tot = [x + y for x, y in zip(tot, t)]
        per_class[c] = {"cells": n, "agree_pct": round(100 * (t[0] + t[3]) / n, 1) if n else None,
                        "kappa": None if kappa(*t) is None else round(kappa(*t), 3),
                        "pos_a": t[0] + t[1], "pos_b": t[0] + t[2], "both_pos": t[0],
                        "only_a": t[1], "only_b": t[2]}
    n = sum(tot)
    overall = {"images_compared": len(common), "status_mismatch": len(status_diff),
               "cells": n, "agree_pct": round(100 * (tot[0] + tot[3]) / n, 1) if n else None,
               "kappa_pooled": None if kappa(*tot) is None else round(kappa(*tot), 3)}
    ks = [v["kappa"] for v in per_class.values() if v["kappa"] is not None and v["cells"]]
    overall["kappa_mean_over_classes"] = round(sum(ks) / len(ks), 3) if ks else None
    per_zone = {z: round(100 * v[0] / v[1], 1) for z, v in zone_agree.items() if v[1]}
    return {"overall": overall, "per_class": per_class, "per_zone": per_zone,
            "status_mismatch": status_diff}, rows


def write_report(res, rows, out: Path, name_a, name_b):
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "disagreements.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "zone", "class", "a", "b"])
        w.writeheader()
        w.writerows(rows)
    o = res["overall"]
    L = [f"# ความตรงกันของผู้ตรวจ: A = `{name_a}` · B = `{name_b}`", "",
         f"- ภาพที่ทั้งคู่ตรวจ: **{o['images_compared']}** · ตัดสิน \"ภาพใช้ไม่ได้\" ไม่ตรงกัน: {o['status_mismatch']}",
         f"- ช่องที่เทียบ (ภาพ × โซน × ปัญหา): {o['cells']}",
         f"- ตรงกัน {o['agree_pct']}% · Cohen's kappa รวม {o['kappa_pooled']} · เฉลี่ยรายปัญหา {o['kappa_mean_over_classes']}",
         "", "| ปัญหา | ช่อง | ตรงกัน % | kappa | ระดับ | A ว่ามี | B ว่ามี | มีทั้งคู่ | A เท่านั้น | B เท่านั้น |",
         "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for c, v in res["per_class"].items():
        L.append(f"| {fz.CLASS_TH.get(c, c)} ({c}) | {v['cells']} | {v['agree_pct']} | {v['kappa']} | {verdict(v['kappa'])} "
                 f"| {v['pos_a']} | {v['pos_b']} | {v['both_pos']} | {v['only_a']} | {v['only_b']} |")
    L += ["", "| โซน | ตรงกัน % |", "|---|---:|"]
    for z, p in sorted(res["per_zone"].items()):
        L.append(f"| {fz.ZONE_TH.get(z, z)} | {p} |")
    L += ["", "kappa: >0.8 เกือบสมบูรณ์ · 0.6–0.8 มาก · 0.4–0.6 ปานกลาง · <0.4 ต่ำ (Landis & Koch 1977).",
          "ถ้าปัญหาไหน kappa ต่ำ แปลว่าคนสองคนยังเห็นไม่ตรงกัน — คะแนนโมเดลบนปัญหานั้นเชื่อได้ไม่เกินระดับนี้",
          "", "ขั้นต่อไป: ตัดสินช่องที่ไม่ตรงกันด้วย `review_tool.py --compare A B` แล้วรัน `compare_reviews.py ... --final`"]
    (out / "agreement.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def make_final(a, b, adj, out_path: Path):
    """agreed cells from A∩B; cells/status they disagreed on come from the adjudication
    file. Images whose disagreement hasn't been settled yet are left out (reported)."""
    final, pending = [], []
    for img in sorted(i for i in a if i in b):
        ra, rb, rj = a[img], b[img], adj.get(img)
        if rj is not None:
            final.append(dict(rj, reviewer="adjudicated"))
            continue
        same_status = ra["status"] == rb["status"]
        same_labels = ra["status"] != "ok" or ra["labels"] == rb["labels"]
        if same_status and same_labels:
            final.append(dict(ra, reviewer="agreed"))
        else:
            pending.append(img)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(final), pending


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="reviews of reviewer A (e.g. zone_reviews.jsonl)")
    ap.add_argument("--b", required=True, help="reviews of reviewer B (e.g. zone_reviews_claude.jsonl)")
    ap.add_argument("--split", default="test", help="compare only this split ('' = all)")
    ap.add_argument("--out", default="review_compare")
    ap.add_argument("--adjudicated", default=None, help="reviews saved by review_tool.py --compare")
    ap.add_argument("--final", action="store_true", help="also write zone_reviews_final.jsonl")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    res, rows = compare(a, b, args.split or None)
    out = Path(args.out)
    write_report(res, rows, out, Path(args.a).name, Path(args.b).name)
    o = res["overall"]
    print(f"images {o['images_compared']} · status mismatch {o['status_mismatch']} · cells {o['cells']} · "
          f"agree {o['agree_pct']}% · kappa {o['kappa_pooled']} (mean over classes {o['kappa_mean_over_classes']})")
    for c, v in res["per_class"].items():
        print(f"  {c:11s} agree {v['agree_pct']}%  kappa {v['kappa']}  A+ {v['pos_a']}  B+ {v['pos_b']}")
    print(f"report: {out / 'agreement.md'} · {len(rows)} disagreeing cells in {out / 'disagreements.csv'}")
    if args.final:
        adj = load(args.adjudicated) if args.adjudicated and Path(args.adjudicated).exists() else {}
        n, pending = make_final(a, b, adj, out / "zone_reviews_final.jsonl")
        print(f"final labels: {n} images -> {out / 'zone_reviews_final.jsonl'}"
              + (f" · {len(pending)} images still need adjudication (left out)" if pending else ""))


if __name__ == "__main__":
    main()
