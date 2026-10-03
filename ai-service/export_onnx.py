"""
export_onnx.py — แปลงโมเดลเป็น ONNX เพื่อให้ ai-service รันได้โดยไม่ต้องใช้ PyTorch
=================================================================================
Run this (needs torch + ultralytics, i.e. on the training machine) whenever a
.pt model in ai-service/models/ changes, then commit the .onnx files:

    python export_onnx.py                      # both models
    python export_onnx.py --zone path\\to\\best.pt   # a newly trained zone model

Produces models/skin_analysis_farmasi.onnx and models/zone_classifier.onnx.
The zone model's thresholds, arch and img_size are stored as ONNX metadata,
so the .onnx file is self-contained. The service uses the .onnx files when
present and falls back to the .pt files (with torch) otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")


def export_yolo(pt: str, out: str):
    from ultralytics import YOLO
    path = YOLO(pt).export(format="onnx", imgsz=640, dynamic=True, simplify=False, opset=17, verbose=False)
    if os.path.abspath(path) != os.path.abspath(out):
        shutil.move(path, out)
    print("YOLO ->", out)


def export_zone(pt: str, out: str):
    import onnx
    import torch
    from zone_model import ZoneClassifier

    ck = torch.load(pt, map_location="cpu", weights_only=False)
    model = ZoneClassifier(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    img = int(ck.get("img_size") or 224)
    x = torch.zeros(2, 3, img, img)
    zi = torch.tensor([0, 1], dtype=torch.long)
    torch.onnx.export(model, (x, zi), out, input_names=["image", "zone"], output_names=["logits"],
                      dynamic_axes={"image": {0: "n"}, "zone": {0: "n"}, "logits": {0: "n"}},
                      opset_version=17, dynamo=False)
    m = onnx.load(out)
    meta = {"thresholds": json.dumps({k: float(v) for k, v in ck["thresholds"].items()}),
            "arch": ck["arch"], "img_size": str(img), "epoch": str(ck.get("epoch")),
            "thresholds_tuned_on": str(ck.get("thresholds_tuned_on", ""))}
    for k, v in meta.items():
        p = m.metadata_props.add(); p.key, p.value = k, v
    onnx.save(m, out)
    print("zone classifier ->", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo", default=os.path.join(MODELS, "skin_analysis_farmasi.pt"))
    ap.add_argument("--zone", default=os.path.join(MODELS, "zone_classifier.pt"))
    ap.add_argument("--only", choices=["yolo", "zone"])
    a = ap.parse_args()
    if a.only in (None, "yolo") and os.path.exists(a.yolo):
        export_yolo(a.yolo, os.path.join(MODELS, "skin_analysis_farmasi.onnx"))
    if a.only in (None, "zone") and os.path.exists(a.zone):
        export_zone(a.zone, os.path.join(MODELS, "zone_classifier.onnx"))


if __name__ == "__main__":
    main()
