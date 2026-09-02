# Skinglow — AI Skin Analysis & Skincare Recommendation

## 1. Architecture Overview

```
┌─────────────┐   multipart/form-data    ┌───────────────┐   multipart    ┌──────────────────┐
│  Frontend   │ ───────────────────────► │  PHP REST API │ ─────────────► │  Python AI Service │
│ HTML/CSS/JS │ ◄─────────────────────── │  (PDO+MySQL)  │ ◄───────────── │  FastAPI (mock CV) │
└─────────────┘        JSON              └───────┬────────┘     JSON      └──────────────────┘
                                                   │
                                                   ▼
                                             ┌───────────┐
                                             │  MySQL DB │
                                             │ users /   │
                                             │ history / │
                                             │ products  │
                                             └───────────┘
```

**Flow for a skin scan:**
1. User uploads/captures a face photo in the browser.
2. Frontend `POST`s the image to `backend-php/api/analyze.php`.
3. PHP validates the file (real MIME check, size limit), stores the original,
   then forwards it via **cURL** to the Python AI service (`/analyze`).
4. The AI service returns bounding boxes + labels (`acne`, `black_spot`,
   `eyebag`, `redness`, `oiliness`, `wrinkle`) and an overall skin score.
5. PHP draws the bounding boxes onto a copy of the image using **GD**,
   saves both image versions, and writes a row to `analysis_history`.
6. PHP queries `products` for items matching the detected issues.
7. PHP returns one JSON payload (score, image URLs, detections, products)
   which the frontend renders — including the scan panel animation and
   the annotated image.

## 2. Directory Structure

```
skinglow/
├── frontend/
│   ├── login.html               # login + signup, stores session token
│   └── index.html               # analysis UI (upload, camera, results, Insight/Compare/Daily tabs) — requires login
├── backend-php/
│   ├── config/
│   │   ├── config.php           # app constants, CORS, upload limits
│   │   └── database.php         # PDO connection (prepared statements only)
│   ├── src/
│   │   ├── Auth.php             # issues/verifies session tokens
│   │   └── BoundingBoxDrawer.php  # GD-based box drawing
│   ├── api/
│   │   ├── register.php         # returns a token — auto-logs in after signup
│   │   ├── login.php
│   │   ├── analyze.php          # requires auth; upload -> AI -> draw -> save flow
│   │   └── history.php          # requires auth
│   └── uploads/
│       ├── original/
│       └── processed/
├── ai-service/
│   ├── main.py                  # FastAPI service — trained model(s) + heuristic fallback
│   ├── ensemble.py              # multi-model loading/fusion (Weighted Boxes Fusion)
│   ├── models/                  # trained .pt weights live here
│   ├── scripts/                 # dataset audit + multi-config training comparison
│   └── train/                   # training pipeline + dataset notes
├── database/
│   └── schema.sql               # users / analysis_history / products
├── DEPLOY_STEPS.md               # free-tier cloud deployment walkthrough
├── LOCAL_DEPLOY.md               # XAMPP local deployment walkthrough
└── README.md
```

## 3. Setup

### Database
```bash
mysql -u root -p < database/schema.sql
```

### AI Service (Python)
```bash
cd ai-service
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### PHP Backend
```bash
cd backend-php
# set env vars: DB_HOST, DB_NAME, DB_USER, DB_PASS, AI_SERVICE_URL, JWT_SECRET
php -S 0.0.0.0:8080
```

### Frontend
Open `frontend/login.html` first (not `index.html` directly) — register or
log in there, which stores a session token and redirects to `index.html`.
Opening `index.html` without a session just bounces you back to the login
page.

**For full step-by-step instructions** (XAMPP local setup or free-tier
cloud hosting, including common gotchas we hit while testing this exact
flow), see `LOCAL_DEPLOY.md` or `DEPLOY_STEPS.md` instead of the quick
commands above.

### Frontend
Open `frontend/index.html` in a browser, or serve it with any static file
server. Update `API_ANALYZE_URL` in the `<script>` block to match your PHP
backend's address. If the API is unreachable, the page falls back to demo
data so the UI can still be previewed standalone.

## 4. Security Notes / Next Steps
- All SQL uses PDO prepared statements (`PDO::ATTR_EMULATE_PREPARES => false`).
- Uploaded file type is verified server-side via `finfo`, not the client-supplied MIME type.
- Passwords are hashed with `password_hash()` (bcrypt).
- **Auth is fully wired end-to-end**, not just backend endpoints: `login.html`
  calls `register.php`/`login.php`, stores the returned token in
  `localStorage`, and `index.html` requires that token to load at all
  (redirects to `login.html` otherwise). `analyze.php` and `history.php`
  derive `user_id` from the verified token (`src/Auth.php`) rather than
  trusting a client-submitted value — see `Auth::requireAuth()`.
- `Auth.php` issues a simple HMAC-signed token, not a standards-compliant
  JWT; swap in `firebase/php-jwt` if this ever needs to interoperate with
  other systems/services.
- **AI model:** `ai-service/main.py` now supports 1+ trained YOLOv8 models
  dropped into `ai-service/models/` (currently: one model fine-tuned on
  FarmasiSkinCare's Skin_Analysis dataset), and automatically falls back
  to the OpenCV heuristic pipeline if none load successfully — check the
  `/health` endpoint's `model_type` field (`yolov8_ensemble` /
  `yolov8_trained` / `heuristic_cv_v2`) and `model_count` to see what's
  actually active. Adding a second or third model (trained on a different
  dataset) automatically switches to Weighted-Boxes-Fusion ensembling —
  see `ai-service/ensemble.py` and `ai-service/train/README.md` section 6.
  Per-class accuracy is currently uneven (strong on redness/black_spot,
  weak on oiliness/wrinkle) — see `ai-service/train/README.md` for the
  training/evaluation notes and ideas for improving it further.
- **Dataset/training tooling:** `ai-service/scripts/audit_dataset.py` checks
  a YOLO dataset for duplicates, corrupted files, invalid boxes, and
  train/val/test leakage before you train on it.
  `ai-service/scripts/train_compare.py` runs and compares multiple training
  configurations (model size, dataset version, augmentation) side by side
  instead of tuning on a single run. Both require a local GPU + dataset —
  see comments at the top of each file for usage.
- **Deployment:** see `DEPLOY_STEPS.md` for a full free-tier cloud
  deployment walkthrough (Render + InfinityFree + Cloudflare Pages),
  including an explicit checklist to avoid ever being asked for payment
  info — or `LOCAL_DEPLOY.md` to run the whole stack on your own machine
  with XAMPP instead (no accounts, no free-tier limits).
- Add rate limiting and auth middleware to `analyze.php` before production use.
