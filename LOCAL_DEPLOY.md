# Running Skinglow locally

This is the local/offline equivalent of `DEPLOY_STEPS.md` — no accounts,
no free-tier limits, no cold starts. Good for development and thesis
demos where you control the machine. Everything below assumes Windows,
macOS, or Linux with XAMPP; adjust paths for your OS where noted.

## Part 1 — PHP + MySQL (XAMPP)

1. Install [XAMPP](https://www.apachefriends.org/) — gives you PHP, MySQL,
   and phpMyAdmin in one installer, no separate setup needed.
2. Open the XAMPP Control Panel, click **Start** next to Apache and MySQL.
3. Copy the entire `skinglow/` project folder into XAMPP's web root:
   - Windows: `C:\xampp\htdocs\skinglow`
   - macOS: `/Applications/XAMPP/htdocs/skinglow`
   - Linux: `/opt/lampp/htdocs/skinglow`
4. Open `http://localhost/phpmyadmin` → **Import** → choose
   `database/schema.sql` → Go. This creates `skinglow_db` with the
   `users`, `analysis_history`, and `products` tables, plus sample products.
5. **Check `backend-php/config/database.php` and `config.php`.** The
   defaults already match a fresh XAMPP install (`DB_HOST=127.0.0.1`,
   `DB_USER=root`, `DB_PASS=''` empty, `APP_BASE_URL=http://localhost/skinglow/backend-php`)
   — if your XAMPP MySQL has a root password set, override it via an
   environment variable or edit the default directly in `database.php`.
6. Create the upload folders (PHP won't auto-create parent dirs on all
   setups): inside `backend-php/`, make sure `uploads/original/` and
   `uploads/processed/` exist. `analyze.php` creates them at runtime if
   missing, but if you hit a permissions error, create them manually and
   make sure the web server user can write to them.

## Part 2 — Python AI service

1. Make sure you have Python 3.10+ installed (`python --version`).
2. Open a terminal in `ai-service/`:
   ```bash
   cd skinglow/ai-service
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows (cmd): `venv\Scripts\activate`
   - Windows (PowerShell): `venv\Scripts\Activate.ps1`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies. **Do the torch CPU install first** (same reason
   as the cloud deploy — the default PyPI torch wheel is huge with CUDA
   support you don't need for local CPU inference on a laptop):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
   If you actually have an NVIDIA GPU on this machine (like the RTX 3070
   used for training) and want the AI service itself to use it for faster
   inference, skip the CPU-only line and just `pip install torch` normally
   — but this is optional, inference (unlike training) is fast enough on
   CPU for a single request at a time.
5. Confirm your trained model is in place: `ai-service/models/` should
   contain at least `skin_analysis_farmasi.pt`. If you've since renamed or
   added files here, double-check `ensemble.py`'s `ENSEMBLE_CONFIG` still
   points at the right filenames.
6. Run the service:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```
7. Check `http://127.0.0.1:8000/health` in a browser. You should see:
   ```json
   {"status":"ok","model_type":"yolov8_trained","model_count":1,"models":[...]}
   ```
   If `model_type` says `heuristic_cv_v2` instead, the model file didn't
   load — check the terminal output from step 6 for the actual load error.

## Part 3 — Point the PHP backend at the local AI service

`backend-php/config/config.php` already defaults `AI_SERVICE_URL` to
`http://127.0.0.1:8000/analyze`, which matches step 6 above exactly — for
a fully local setup you shouldn't need to change this at all.

## Part 4 — Frontend

Open **`http://localhost/skinglow/frontend/login.html`** in your browser
— not `file://` directly, and not `index.html` directly either (it now
requires a logged-in session and will redirect you straight back to
`login.html` if you try). This matters for `file://`: opening the HTML
file straight from disk means its `fetch()` calls to
`backend-php/api/*.php` are cross-origin from a `file://` context, which
browsers block. Going through Apache (the `http://localhost/...` URL)
puts both the frontend and PHP backend on the same origin, so it just
works.

Register a new account (or log in if you already have one) — this stores
a session token in `localStorage` and takes you to `index.html`
automatically.

`API_ANALYZE_URL` in `index.html` is a **relative** path
(`../backend-php/api/analyze.php`), resolved relative to
`frontend/index.html` itself — this works correctly whether the project
sits at your web server's root or in a subfolder (like
`htdocs/skinglow/` here). An earlier version of this file used an
absolute path (`/backend-php/...`), which silently broke under any
subfolder deployment since the browser resolves a leading `/` against
the domain root, not the project folder — worth knowing if you ever see
this revert during a merge/rebase.

## Sanity checklist

- [ ] `http://localhost/phpmyadmin` shows the `skinglow_db` database with data in `products`
- [ ] `http://127.0.0.1:8000/health` shows `model_type: yolov8_trained`
- [ ] Opening `http://localhost/skinglow/frontend/index.html` directly redirects you to `login.html` (no session yet — this is correct)
- [ ] Registering a new account on `login.html` logs you straight in and lands on the analysis page with your name in the top-right
- [ ] Logging out (top-right button) sends you back to `login.html`, and `index.html` redirects to login again if you try to open it directly afterward
- [ ] Uploading a photo returns real bounding boxes, not a JS error

## Common local-only gotchas

- **"Failed to fetch" / CORS error in browser console:** almost always
  means you opened `index.html` via `file://` instead of through Apache —
  see Part 4.
- **PHP can't reach the AI service:** confirm uvicorn (Part 2, step 6) is
  still running in its terminal — closing that terminal window kills it.
  Unlike Render, nothing restarts it for you locally.
- **"Database connection failed":** check XAMPP's MySQL is actually
  running (Control Panel should show it green/started), and that
  `database.php`'s credentials match your XAMPP MySQL setup.
- **GD library errors when drawing bounding boxes:** XAMPP's default PHP
  build usually includes GD, but if `BoundingBoxDrawer.php` errors out,
  check `php.ini` (via XAMPP Control Panel → Apache → Config) has the
  `gd` extension uncommented/enabled, then restart Apache.
- **"กรุณาเข้าสู่ระบบก่อนใช้งาน" (401) even right after logging in:** the
  frontend sends your session as an `Authorization: Bearer <token>` header.
  A few Apache/PHP configs strip this header before PHP ever sees it —
  `Auth.php` already checks the common fallback
  (`REDIRECT_HTTP_AUTHORIZATION`), but if you still hit this, add
  `CGIPassAuth On` inside the `<Directory>` block for your project in
  Apache's config, or confirm `mod_headers` is enabled.
- **Port 8000 already in use:** something else on your machine is using
  it — either stop that process or run uvicorn on a different port
  (`--port 8001`) and update `AI_SERVICE_URL` in `config.php` to match.
