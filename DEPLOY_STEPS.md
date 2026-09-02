# Deploying Skinglow (free hosting, zero payment risk)

**Read this first:** I (Claude) cannot actually click through hosting
dashboards, create accounts, or deploy this for you — I don't have browser
control or the ability to sign up for third-party services on your behalf
(that would need your email, and creating accounts without you present
isn't something I should do anyway). What I *did* do: integrated your
trained model (`best.pt`) into the AI service, tested it end-to-end
locally, and prepared every config file so the actual deploy is mostly
copy-paste. This doc is the exact sequence of clicks/commands for you to
run yourself.

## The one hard rule: if anything asks for a card, stop

Render's free web service tier does **not** require a credit card — but
some users have reported the dashboard prompting for payment info if you
accidentally pick a paid instance type instead of "Free" during setup.
**If any screen in any step below asks for payment/card details, stop and
don't enter anything.** That means you've drifted off the free path.
Screenshot it and message me, or just re-check that "Free" is selected as
the instance type. None of the services below should ever need a card for
what this project needs.

---

## Part 1 — AI service (Render)

1. Push this project to a GitHub repo (public or private, either works).
2. Go to render.com → sign up (email or GitHub login, **no card entry
   screen should appear for account creation itself**).
3. Dashboard → "New +" → "Blueprint".
4. Connect the GitHub repo. Render will detect `render.yaml` at the repo
   root and pre-fill the service config (already set to `plan: free`).
5. Before clicking deploy, double-check the instance type shown is
   **"Free"** — not "Starter" or anything with a $ next to it.
6. Deploy. First build takes a while (~5-10 min) because of the
   `torch` CPU wheel install — this is normal, not a hang.
7. Once live, note the URL Render gives you, e.g.
   `https://skinglow-ai-service.onrender.com` — you'll need it in Part 2.
8. Test it: visit `https://<your-url>.onrender.com/health` in a browser.
   You should see `{"status":"ok","model_type":"yolov8_trained"}`.
   If it says `"heuristic_cv_v2"` instead, `best.pt` didn't make it into
   the deploy — check it's actually committed to the repo (it's a binary
   file, `git add` it explicitly, don't rely on a wildcard `.gitignore`).

**Cold start note:** free tier sleeps after 15 min idle. The first request
after that takes 30-60s to wake up — this is expected Render free-tier
behavior, not a bug in the code.

## Part 2 — PHP + MySQL backend (InfinityFree)

1. Sign up at infinityfree.com (no card required for their free plan).
2. Create a new hosting account, get your MySQL credentials from the
   control panel (phpMyAdmin is included).
3. Import `database/schema.sql` via phpMyAdmin.
4. Upload the contents of `backend-php/` via the file manager or FTP.
5. Edit `backend-php/config/config.php`:
   - Set `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS` env vars (or hardcode
     temporarily, but env vars are safer) to match InfinityFree's MySQL info.
   - Set `AI_SERVICE_URL` to the Render URL from Part 1, e.g.
     `https://skinglow-ai-service.onrender.com/analyze`.
   - Update `APP_BASE_URL` to your InfinityFree domain.
6. **Test the cURL connection specifically** — some free PHP hosts block
   outbound requests to external domains. Upload a tiny test script that
   just does `curl_exec` against your Render `/health` URL and check it
   returns something instead of failing silently. If it's blocked, that's
   a host-level restriction you'd need to raise with InfinityFree support
   or switch to GoogieHost.

## Part 3 — Frontend (Cloudflare Pages)

1. Sign up at Cloudflare (free, no card required for Pages).
2. Pages → "Create a project" → connect the same GitHub repo.
3. Build settings: none needed (it's static HTML) — set the output
   directory to `frontend/`.
4. Deploy. You'll get a `*.pages.dev` URL.
5. In `frontend/index.html`, update `API_ANALYZE_URL` to point at your
   InfinityFree domain's `backend-php/api/analyze.php`.
6. In `backend-php/config/config.php`, update the CORS header
   (`Access-Control-Allow-Origin`) from `*` to your actual `*.pages.dev`
   domain — leaving it as `*` works but is looser than you want once
   you have a real domain to lock it to.

---

## After deploy: sanity checklist

- [ ] `GET /health` on Render returns `model_type: yolov8_trained`
- [ ] Uploading a photo on the live frontend returns real bounding boxes,
      not an error
- [ ] No step above ever asked you to enter a credit card
- [ ] `analyze.php` successfully reaches the Render AI service (test with
      a real upload, not just checking the endpoint loads)

If something breaks partway through, the most common causes are: (1) the
InfinityFree→Render cURL block mentioned above, (2) `best.pt` not actually
committed to the repo, or (3) CORS mismatch between the Pages domain and
what's set in `config.php`. Come back with the specific error and I can
help debug it.
