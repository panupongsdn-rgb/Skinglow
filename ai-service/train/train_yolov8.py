"""
train_yolov8.py
----------------
Fine-tunes a YOLOv8 object-detection model on a labeled skin-condition
dataset (acne / black_spot / eyebag / oiliness / redness / wrinkle).

This is the real training step described in train/README.md. It requires:
  - A GPU (a few hours on a single consumer GPU for a small dataset;
    CPU-only training will work but is impractically slow)
  - A labeled dataset in YOLO format (images/ + labels/ + data.yaml)
  - `pip install ultralytics`

Usage:
    python train_yolov8.py --data data.yaml --epochs 100 --imgsz 640

After training, the best weights are saved to:
    runs/detect/train/weights/best.pt

Copy that file into ai-service/ and swap main.py's heuristic pipeline for
a call to this model (see the load_trained_model() example at the bottom
of this file for the inference-side code you'd use instead).
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for Skinglow skin-condition detection")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to YOLO dataset config")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    args = parser.parse_args()

    model = YOLO(args.model)  # start from a COCO-pretrained checkpoint (transfer learning)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        # augmentation tuned conservatively for close-up face photos:
        # avoid flips/rotations that would break left/right facial symmetry cues,
        # keep color jitter modest since skin tone/lighting matters for the labels.
        fliplr=0.5,
        flipud=0.0,
        degrees=5,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        mosaic=0.5,
    )

    metrics = model.val()
    print("Validation metrics:", metrics.results_dict)

    model.export(format="onnx")  # optional: for faster/cheaper CPU inference in production


def load_trained_model(weights_path: str = "best.pt") -> YOLO:
    """Example of how to load the trained model for inference in main.py,
    replacing the heuristic analyze_face() pipeline:

        from train.train_yolov8 import load_trained_model
        model = load_trained_model("best.pt")

        results = model.predict(image, conf=0.35)[0]
        detections = [
            {
                "box": [int(x) for x in box.xyxy[0].tolist()],
                "label": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 2),
            }
            for box in results.boxes
        ]
    """
    return YOLO(weights_path)


if __name__ == "__main__":
    main()
