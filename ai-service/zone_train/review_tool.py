"""
review_tool.py — ตรวจ/แก้ label รายโซนผ่านหน้าเว็บในเครื่อง
============================================================
Opens a small local web page that shows each face split into its zones with
the labels pre-filled, so a person can confirm or correct them in a few
seconds per image. Every saved image gets COMPLETE labels (all conditions x
all zones known), which is what the dataset is missing.

Pre-filled suggestions come from:
  * the original polygon labels (reliable, shown in purple)
  * optionally a trained zone classifier  (--model best.pt, shown as "AI"),
    pre-ticked only when it is confident (--ai-min-prob, default 0.6).
    AI suggestions are NOT shown on the test split unless --ai-on-test is
    given: people tend to accept pre-ticked answers, and test labels that
    lean towards the model's own answers would inflate its test score.
Unknown cells start unticked and are marked "?", so it's clear which answers
nobody has given yet.

Reviews are appended to zone_reviews.jsonl (the latest save of an image wins).
They are keyed by image name + zone, so they survive re-running
build_zone_dataset.py. Pass the file to train_zone_classifier.py with
--reviews zone_reviews.jsonl.

Usage (from the skinglow-trian folder):
    python <repo>/ai-service/zone_train/review_tool.py --zones zone_dataset --data-root dataset_clean_v2
    python ... --model runs_zone/<run>/best.pt        # AI suggestions from round 1
Then open http://127.0.0.1:8765 (opens automatically). Only reachable from
this computer. Ctrl+C to stop; nothing is lost, every save is on disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import threading
import time
import webbrowser
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import face_zones as fz  # noqa: E402

CLASSES = fz.CLASSES
ZONE_ORDER = fz.ZONE_NAMES + ["patch"]
ZONE_TH = dict(fz.ZONE_TH, patch="ภาพระยะใกล้")
SPLIT_ORDER = {"test": 0, "valid": 1, "train": 2}
RARE = {"redness", "eyebag"}


def load_reviews(path: Path):
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    latest[r["image"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    return latest


class Store:
    def __init__(self, zones_dir: Path, data_root: Path, reviews_path: Path, model_path: str | None,
                 ai_min_prob: float = 0.6, ai_on_test: bool = False):
        self.ai_min_prob, self.ai_on_test = ai_min_prob, ai_on_test
        self.zones_dir, self.data_root, self.reviews_path = zones_dir, data_root, reviews_path
        self.lock = threading.Lock()
        faces_path = zones_dir / "faces.jsonl"
        if not faces_path.exists():
            sys.exit(f"{faces_path} not found — re-run build_zone_dataset.py (this version writes it).")
        self.faces = {}
        for line in faces_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                f = json.loads(line)
                self.faces[f["image"]] = f
        self.rows = defaultdict(dict)  # image -> zone -> row
        with open(zones_dir / "labels.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                self.rows[r["image"]][r["zone"]] = r
        self.reviews = load_reviews(reviews_path)
        self.order = self._make_order()
        self.predictor = None
        self.pred_cache = {}
        if model_path:
            import zone_model  # needs torch
            self.predictor = zone_model.load_predictor(model_path)
            if self.predictor is None:
                sys.exit(f"could not load model {model_path}")
            print(f"AI suggestions from {model_path}")

    def _make_order(self):
        """test first, then valid, then train images most worth reviewing:
        faces with rare classes (redness / eyebag) and with many unknowns."""
        def key(img):
            face = self.faces[img]
            zones = self.rows.get(img, {})
            unknown = sum(1 for r in zones.values() for c in CLASSES if r[c] == "-1")
            rare = any(r[c] == "1" for r in zones.values() for c in RARE)
            is_patch = face["status"] != "ok"
            return (SPLIT_ORDER.get(face["split"], 3), is_patch, not rare, -unknown, img)
        return sorted((i for i in self.faces if i in self.rows), key=key)

    def suggestions(self, img):
        if not self.predictor:
            return {}
        if img in self.pred_cache:
            return self.pred_cache[img]
        import cv2
        zones = [z for z in ZONE_ORDER if z in self.rows[img]]
        crops = [cv2.imread(str(self.zones_dir / self.rows[img][z]["crop"])) for z in zones]
        keep = [(z, c) for z, c in zip(zones, crops) if c is not None]
        out = {}
        if keep:
            with self.lock:
                preds = self.predictor.predict([c for _, c in keep], [z for z, _ in keep])
            for (z, _), p in zip(keep, preds):
                out[z] = {c: v["prob"] for c, v in p.items()}
        self.pred_cache[img] = out
        return out

    def item(self, idx):
        img = self.order[idx]
        face = self.faces[img]
        ai = self.suggestions(img) if (face["split"] != "test" or self.ai_on_test) else {}
        thr = dict(zip(CLASSES, self.predictor.thresholds.tolist())) if self.predictor else {}
        zones = []
        for z in ZONE_ORDER:
            r = self.rows[img].get(z)
            if not r:
                continue
            cells = {}
            for c in CLASSES:
                allowed = fz.class_allowed(c, z)
                orig = int(r[c]) if allowed else None
                p = ai.get(z, {}).get(c)
                cells[c] = {"allowed": allowed, "orig": orig,
                            "ai": None if p is None or not allowed else round(p, 2),
                            "ai_on": bool(p is not None and allowed
                                          and p >= max(thr.get(c, 0.5), self.ai_min_prob))}
            zones.append({"zone": z, "name_th": ZONE_TH.get(z, z), "cells": cells,
                          "polygons": face["zones"].get(z, [])})
        return {"index": idx, "total": len(self.order), "image": img, "split": face["split"],
                "family": face["family"], "status": face["status"], "width": face["width"],
                "height": face["height"], "boxes": face["boxes"], "zones": zones,
                "review": self.reviews.get(img)}

    def save(self, payload):
        img = payload.get("image")
        if img not in self.faces:
            raise ValueError("unknown image")
        status = payload.get("status", "ok")
        if status not in ("ok", "bad"):
            raise ValueError("bad status")
        labels = {}
        if status == "ok":
            for z, cells in (payload.get("labels") or {}).items():
                if z not in self.rows[img]:
                    continue
                labels[z] = {c: 1 if cells.get(c) else 0 for c in CLASSES if fz.class_allowed(c, z)}
        rec = {"image": img, "split": self.faces[img]["split"], "status": status, "labels": labels,
               "reviewer": str(payload.get("reviewer", ""))[:40], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with self.lock:
            with open(self.reviews_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.reviews[img] = rec
        return rec

    def stats(self):
        per_split = Counter(self.faces[i]["split"] for i in self.order)
        done = Counter(self.faces[i]["split"] for i in self.order if i in self.reviews)
        pos = Counter()
        for r in self.reviews.values():
            for cells in r.get("labels", {}).values():
                for c, v in cells.items():
                    pos[c] += v
        first_open = {s: next((k for k, i in enumerate(self.order)
                               if self.faces[i]["split"] == s and i not in self.reviews), None)
                      for s in SPLIT_ORDER}
        return {"total": dict(per_split), "reviewed": dict(done), "positives": dict(pos),
                "first_unreviewed": first_open, "ai": self.predictor is not None}


def make_handler(store: Store, html: bytes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/":
                    return self._send(200, html, "text/html; charset=utf-8")
                if u.path == "/api/stats":
                    return self._json(store.stats())
                if u.path == "/api/item":
                    idx = int(parse_qs(u.query).get("i", ["0"])[0])
                    idx = max(0, min(idx, len(store.order) - 1))
                    return self._json(store.item(idx))
                if u.path.startswith("/img/"):
                    img = unquote(u.path[5:])
                    face = store.faces.get(img)
                    if not face:  # only images from the dataset, never arbitrary paths
                        return self._send(404, b"not found", "text/plain")
                    p = store.data_root / face["split"] / "images" / img
                    if not p.exists():
                        return self._send(404, b"image missing", "text/plain")
                    return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "image/jpeg")
                return self._send(404, b"not found", "text/plain")
            except Exception as exc:
                return self._json({"error": str(exc)}, 500)

        def do_POST(self):
            if urlparse(self.path).path != "/api/review":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                rec = store.save(json.loads(self.rfile.read(n) or b"{}"))
                return self._json({"ok": True, "review": rec})
            except Exception as exc:
                return self._json({"ok": False, "error": str(exc)}, 400)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", default="zone_dataset", help="folder written by build_zone_dataset.py")
    ap.add_argument("--data-root", required=True, help="dataset folder with <split>/images (e.g. dataset_clean_v2)")
    ap.add_argument("--reviews", default="zone_reviews.jsonl", help="where reviews are saved")
    ap.add_argument("--model", default=None, help="optional zone classifier best.pt for AI suggestions")
    ap.add_argument("--ai-min-prob", type=float, default=0.6, help="pre-tick AI suggestions only at/above this probability")
    ap.add_argument("--ai-on-test", action="store_true", help="also show AI suggestions on the test split (not recommended)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    store = Store(Path(args.zones), Path(args.data_root), Path(args.reviews), args.model,
                  args.ai_min_prob, args.ai_on_test)
    html = (HERE / "review_tool.html").read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, html))
    url = f"http://127.0.0.1:{args.port}"
    s = store.stats()
    print(f"{len(store.order)} images · reviewed {sum(s['reviewed'].values())} · saving to {args.reviews}")
    print(f"Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped — reviews are saved in", args.reviews)


if __name__ == "__main__":
    main()
