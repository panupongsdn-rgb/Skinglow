# Training a real skin-condition detection model

The `ai-service/main.py` in this project currently ships as a **classical
computer-vision heuristic** (OpenCV color thresholding + blob detection).
It reacts to real pixel data — so it's a step up from the original random
mock — but it is **not a trained model** and its accuracy has a hard
ceiling. This folder contains everything needed to train a real one.

## Why I couldn't just "train it for you"

Training a genuinely more accurate model requires three things this
sandboxed environment doesn't have:

1. **A labeled dataset.** Bounding boxes drawn by someone (ideally with
   dermatology knowledge) around acne/black-spot/eyebag/redness/oiliness/wrinkle
   regions on hundreds–thousands of real face photos. I have no such
   dataset, and I'm not able to browse the open web to source one for you
   inside this environment.
2. **A GPU.** Fine-tuning even a small YOLOv8 model on CPU is impractically
   slow (many hours to days for a tiny dataset).
3. **Consent & licensing.** Face photos of real people are sensitive data.
   Any dataset you train on needs a clear license or the subjects' consent,
   especially since this is a health-adjacent product.

So the honest path forward is: I've upgraded the mock to something that
actually looks at pixels, and I've written the full training pipeline
below for you to run yourself once you have data + a GPU (your own
machine, Colab, or a cloud GPU instance).

## 0. The "Clear" class decision

If you're adding a 7th class for "no skin issues found," **don't add it as
a YOLO detection class.** Reasoning:

- Clear has no bounding-box location — it's a property of the whole face,
  not an object with a position. Forcing a box (e.g. the full face rect)
  makes the model learn to distinguish "big background-ish box" from
  "small specific-problem box," which is a much harder and less useful
  distinction than what YOLO is good at.
- The existing confusion matrix already shows heavy background false
  positives (Wrinkle: 546 FPs, Black Spot: 415 FPs on true background).
  Adding a class that directly competes with "background" is likely to
  make that specific confusion worse, not better.
- It's also unnecessary: **the model already implicitly expresses "clear"
  by simply not firing any detection above the confidence threshold.**
  `main.py`'s `determine_skin_status()` uses exactly that — no retraining
  required.

**What you should still do** (and this is where "Clear" data is actually
useful): add clear/healthy-skin **face photos as negative training
images** — same format as any other training image, just with an *empty*
label `.txt` file (zero bounding boxes). This is standard YOLO negative
mining and directly targets the false-positive problem visible in your
confusion matrix, without touching the class list or model architecture
at all. `scripts/audit_dataset.py` already reports these as "background
images" (not an error) so you can track how many you have per split.

If your thesis specifically requires demonstrating a classification stage
(e.g. for the methodology chapter) rather than pure application logic,
the alternative is a small second-stage classifier (binary: has-issue vs
clear) trained on face crops — but weigh the added complexity against
what's actually a one-line confidence check. I'd only do this if a
reviewer specifically expects to see a distinct model stage.

## 1. Get a dataset

**This project uses "Skin_Analysis" by FarmasiSkinCare** (Roboflow
Universe: https://universe.roboflow.com/farmasiskincare-aan8y/skin_analysis/dataset/1),
for a non-commercial student/thesis project. `data.yaml` in this folder is
already set up to match its classes and confirmed folder layout: `acne`,
`black_spot`, `eyebag`, `oiliness`, `redness`, `wrinkle`.

**✅ Class order confirmed** against the actual `data.yaml` from the
downloaded export (checked 2026-08-12): Roboflow's real order is
`['Acne', 'Black Spot', 'Eyebag', 'Oilness', 'Redness', 'Wrinkle']` —
identical index order to what's already in this folder's `data.yaml`, just
lowercased/renamed to match this codebase's naming convention. If you
re-download or fork a newer version later, re-check this — Roboflow can
reassign IDs if classes are added/removed/reordered in the source project.

**⚖️ Licensing note — read before publishing your thesis repo.** The
downloaded `data.yaml` lists `license: Private`. That's different from the
`CC BY 4.0` badge shown on some sibling "skin analysis" datasets on
Roboflow Universe — "Private" generally means the dataset owner hasn't
granted an open license, even though the project page is publicly
browsable. For a non-commercial university thesis this is very commonly
fine under research/educational use, but to stay safe:
- **Don't redistribute the raw dataset itself** (e.g. don't commit the
  images/labels into a public GitHub repo alongside your code) — keep it
  local or in a private/institutional storage location.
- **Do cite it** in your thesis methodology chapter using the `roboflow:`
  block in `data.yaml` (workspace, project, version, url).
- If your thesis or its code repo will be made public, consider messaging
  the dataset owner (workspace `farmasiskincare-aan8y`) through Roboflow
  to confirm you're OK to reference/use it — a quick message is usually
  enough for a student project and creates a paper trail if anyone asks.
- Trained model *weights* (`best.pt`) are generally fine to share even
  under a private-dataset license, since the weights alone don't
  reproduce the original images — but double check your university's
  academic integrity policy on this if in doubt.

Other options if you need more data or a different starting point:

**A. Use another existing labeled dataset.** Search Roboflow Universe or
academic sources for public acne/skin-condition detection datasets (e.g.
search "acne detection dataset" or "acne04"). Check the license — many
academic ones are for research use only.

**B. Collect and label your own.** With proper consent from photo subjects:
1. Collect close-up, well-lit face photos (consistent lighting reduces
   false positives from the color-threshold-trained model too).
2. Label them with [Roboflow](https://roboflow.com) or
   [CVAT](https://www.cvat.ai) — draw a box around each blemish and tag it
   with one of: `acne`, `black_spot`, `eyebag`, `redness`, `oiliness`, `wrinkle`.
3. Export in **YOLOv8 format** — both tools support this natively.
4. Aim for at least a few hundred images per class as a starting point;
   more (thousands) will meaningfully improve accuracy.

Drop the export into this folder so the structure looks like:
`train/images`, `train/labels`, `valid/images`, `valid/labels`,
`test/images`, `test/labels` — matching `data.yaml`'s `path`/`train`/`val`/`test` keys.

## 2. Install training dependencies

```bash
pip install ultralytics
```

## 3. Train

```bash
cd ai-service/train
python train_yolov8.py --data data.yaml --epochs 100 --imgsz 640 --batch 16
```

- Start from `yolov8n.pt` (nano) for fast iteration; move to `yolov8s.pt`
  or `yolov8m.pt` once your pipeline works, for better accuracy.
- Watch the validation mAP in `runs/detect/train/`. If it's not improving
  after ~50 epochs, the usual culprits are: too little data, inconsistent
  labeling, or class imbalance (way more `acne` boxes than `wrinkle`, etc.)
- `--patience 20` stops training early if validation metrics plateau.

## 4. Evaluate honestly before shipping

Don't just trust the training-set mAP. Hold out a validation set the
model never saw, and ideally have someone review a sample of predictions
against dermatology judgment. Track false positives (flagging clear skin
as acne) separately from false negatives — for a consumer product, a
noisy false-positive rate erodes trust fast.

## 5. Swap it into the API

Once you have `runs/detect/train/weights/best.pt`:

1. Copy `best.pt` into `ai-service/`.
2. In `ai-service/main.py`, replace the call to `analyze_face(bgr)` inside
   the `/analyze` endpoint with a YOLO inference call. Example:

```python
from ultralytics import YOLO

MODEL = YOLO("best.pt")

def analyze_face_ml(bgr):
    results = MODEL.predict(bgr, conf=0.35)[0]
    detections = [
        Detection(
            box=[int(x) for x in box.xyxy[0].tolist()],
            label=MODEL.names[int(box.cls[0])],
            confidence=round(float(box.conf[0]), 2),
        )
        for box in results.boxes
    ]
    penalty = sum(d.confidence for d in detections) * 6
    skin_score = round(max(0.0, 100 - penalty), 1)
    return AnalyzeResponse(skin_score=skin_score, detections=detections, model_type="yolov8_trained")
```

3. Keep the heuristic `analyze_face()` function around as a fallback (e.g.
   if the model file is missing) so the service doesn't hard-crash.

## 6. Combining multiple models trained on different datasets

If you train separate models on each of several datasets (e.g. the
FarmasiSkinCare Skin_Analysis set plus Skin-Problem-Detection-Relabel-Clean3
and face_skin_condition), you have two real options — pick deliberately,
don't default to whichever sounds fancier:

**Option A — Ensemble at inference time (`ai-service/ensemble.py`).**
Run all N models per request and fuse their outputs with Weighted Boxes
Fusion. Already implemented and tested in this repo — drop each trained
`.pt` into `ai-service/models/`, fill in that model's class-name mapping
in `ensemble.py`'s `ENSEMBLE_CONFIG`, and the service automatically scales
from 1 model (passthrough) to N models (fused) with no other code changes.
Costs roughly Nx inference time/memory — matters on a free-tier CPU host.

**Option B — Merge the raw datasets, train once.** Combine all three
datasets' images+labels into one unified YOLO dataset (map each source's
class names to this project's 6 canonical labels, run
`scripts/audit_dataset.py` on the merged result to catch duplicates and
leakage), then train a single model. Generally gives better results than
ensembling for this kind of task since the model learns from all the data
jointly rather than voting between separately-trained opinions, and it's
much cheaper to deploy (1x inference cost).

**My recommendation:** B first. Use A if you specifically want a
methodology-chapter comparison between the two approaches, or if the three
datasets turn out to have very different label conventions that are hard
to reconcile into one taxonomy.

Before combining either way: inspect each source dataset's actual class
list with `YOLO('path/to/model.pt').names` (for a model someone already
trained) or by reading its `data.yaml`/`classes.txt` directly (for the raw
dataset) — Roboflow class names are almost never identical across
different projects (e.g. "Black Spot" vs "Dark-Spots" vs "darkspot"), so
this mapping step is manual no matter which option you pick.

## 7. Keep improving it

Model accuracy in production is a loop, not a one-time step:
- Log low-confidence predictions and periodically review/relabel them.
- Retrain every few months as your labeled dataset grows.
- Consider per-skin-tone validation — color-threshold heuristics (and
  even trained models on unbalanced data) can perform unevenly across
  skin tones, which is a fairness issue worth testing for explicitly.
