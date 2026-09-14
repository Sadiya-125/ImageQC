# Deployment Guide

This walks through deploying ImageQC for real: Neon (database) → Render
(backend, via the `render.yaml` Blueprint) → Vercel (frontend). Follow the
steps in order — each one produces a value the next step needs.

## 1. Create the Neon project and get both connection strings

1. Sign up / log in at [neon.tech](https://neon.tech) and create a new project (any region close to your Render region is fine — this repo defaults Render to `oregon`, see `render.yaml`).
2. In the Neon dashboard, open **Connection Details** for the project's default database. Neon gives you two connection strings — you need both:
   - **Pooled connection** (host contains `-pooler`, routes through PgBouncer) → this is `DATABASE_URL`. The app's actual runtime traffic uses this.
   - **Direct connection** (no `-pooler` in the host) → this is `DIRECT_URL`. Alembic migrations use this only — PgBouncer's pooled connections don't reliably support the session-level features Alembic's DDL transactions rely on (see `backend/app/alembic/env.py`).
3. Both strings look like `postgresql://user:password@host/dbname?sslmode=require&channel_binding=require`. This project's SQLAlchemy setup expects the `asyncpg` driver explicitly, and asyncpg does **not** understand libpq-style `sslmode=`/`channel_binding=` query params (confirmed by testing directly — the connection fails with those params, and succeeds with `ssl=require`). Rewrite both strings before using them anywhere in this app:
   ```
   postgresql://user:pass@host/db?sslmode=require&channel_binding=require
     ->
   postgresql+asyncpg://user:pass@host/db?ssl=require
   ```
4. Save both rewritten strings somewhere private (a password manager, not a committed file) — you'll paste them into `backend/.env` in step 2 and into Render's dashboard in step 3.

## 2. Run Alembic migrations against `DIRECT_URL`, before the first deploy

Do this from your local machine, once, before Render ever serves traffic — a freshly created Neon database has no tables yet, and the backend will fail every request until they exist.

```bash
cd backend
cp .env.example .env
# edit .env: paste your real (asyncpg-scheme) DATABASE_URL and DIRECT_URL from step 1
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
alembic upgrade head
```

Confirm it worked: `alembic current` should print the latest revision id, and connecting to the Neon database (e.g. via `psql` or Neon's SQL editor) should show the `analyses` and `analysis_issues` tables plus `alembic_version`.

Re-run `alembic upgrade head` the same way after any future migration is added — always against `DIRECT_URL`, never the pooled one.

## 3. Deploy the backend to Render via the Blueprint

1. Push this repo to GitHub (Render's Blueprint flow reads `render.yaml` from a connected git repo).
2. In the Render dashboard: **New** → **Blueprint**, connect the repo, and Render will detect `render.yaml` at the root and propose the `imageqc-backend` Docker web service defined there.
3. Render will prompt for every `sync: false` env var declared in `render.yaml` (nothing secret is stored in that file). Fill in:
   | Variable | Value |
   | --- | --- |
   | `DATABASE_URL` | Neon **pooled** connection string, `postgresql+asyncpg://...` |
   | `DIRECT_URL` | Neon **direct** connection string, `postgresql+asyncpg://...` |
   | `CORS_ORIGINS` | Your Vercel frontend's URL, e.g. `https://imageqc.vercel.app` (comma-separate if you have more than one, e.g. a preview + production domain) |
   | `MODEL_WEIGHTS_PATH` | `app/ml/weights/mobilenetv3_iqa.pt` (default; only change if you relocated the file) |
   | `ANOMALY_MODEL_PATH` | `app/ml/weights/anomaly_iforest.joblib` (default) |
   | `MAX_UPLOAD_MB` | `10` (or your preferred limit) |
4. Deploy. Render builds `backend/Dockerfile` (CPU-only PyTorch, ~500MB image) and starts the container — first build takes several minutes given PyTorch/OpenCV's size.
5. Once live, note the backend's `https://imageqc-backend.onrender.com`-style URL — the frontend needs it next.

**Free tier note:** Render's free web services spin down after ~15 minutes of no traffic and cold-start on the next request (10-30+ seconds before the first response, since the container has to boot Python, load PyTorch, and load the model weights before `/health` goes green). This is expected free-tier behavior, not a bug — the frontend's loading state explicitly acknowledges this ("a little longer if the server has been idle").

## 4. Deploy the frontend to Vercel

1. In the Vercel dashboard: **Add New** → **Project**, import this repo, set the **Root Directory** to `frontend/` (this is a monorepo — Vercel needs to know the Next.js app isn't at the repo root).
2. Add one environment variable: `NEXT_PUBLIC_API_URL` = the Render backend URL from step 3.5 (no trailing slash).
3. Deploy. Vercel auto-detects Next.js and needs no other configuration — this repo intentionally has no `frontend/vercel.json`, since there's no custom header/redirect/rewrite behavior to declare (see the frontend build's notes).
4. Once live, take the Vercel URL and go back to Render's dashboard to set `CORS_ORIGINS` to match it exactly, if you hadn't already got the final domain in step 3.

## 5. Post-deploy smoke test

Run through this checklist against the *live* URLs (not localhost) before calling the deploy done:

- [ ] `curl https://<your-render-url>/health` returns `{"status":"ok","model_loaded":true}` (a `503` with `model_loaded: false` means the container came up but the weights failed to load — check Render's logs).
- [ ] Open the live Vercel URL, upload a real photo (e.g. one from `sample_images/`), and confirm the result renders: image preview, quality score gauge, issues list, image stats panel.
- [ ] Click "View Grad-CAM heatmap" on that result and confirm an overlay image loads.
- [ ] Navigate to `/history` on the live frontend and confirm the analysis you just ran appears in the list.
- [ ] Click into that analysis's detail page (`/analyses/<id>`) and confirm it renders the same result, including the original image (served from `GET /api/analyses/{id}/image`).
- [ ] Upload a non-image file (e.g. a `.txt` renamed to `.png`) and confirm the frontend shows a clean validation error, not a crash.

If the backend was asleep (free tier), expect the *first* request in this checklist to take noticeably longer — that's the cold start described in step 3, not a failure.
