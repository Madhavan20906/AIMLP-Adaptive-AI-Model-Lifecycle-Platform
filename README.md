# AIMLP — Adaptive AI Model Lifecycle Platform

An MLOps platform with two modes: train a model from scratch on any
dataset (Mode 1), and continuously evaluate/evolve a production model as
new data arrives (Mode 2) — retraining the same algorithm when it's still
healthy, or replacing it with a better-performing algorithm when it isn't.

## Architecture

```
                    ┌─────────────────────────────┐
  CSV in ────────▶  │  core/dataset_analysis.py    │  target/problem-type
                    │  core/preprocessing.py       │  detection, impute/
                    └──────────────┬───────────────┘  encode/scale/SMOTE
                                   │
                    ┌──────────────▼───────────────┐
                    │  core/candidate_selector.py   │  shortlist algorithms
                    └──────────────┬───────────────┘  from dataset traits
                                   │
                    ┌──────────────▼───────────────┐
                    │  core/trainer.py               │  leaderboard: 12
                    └──────────────┬───────────────┘  algorithms, ranked
                                   │
        MODE 1 ─────────┬──────────┴──────────┬───────── MODE 2
   (best of leaderboard) │                     │  (health + drift on
                          │                     │   the ACTIVE model)
                ┌─────────▼─────────┐ ┌─────────▼─────────┐
                │ registry/          │ │ core/health_score.py│
                │ model_registry.py   │ │ core/drift_detector.py│
                │ (SQLite/Postgres,   │ │ core/decision_engine.py│
                │  versions, rollback)│ │                        │
                └─────────┬─────────┘ └─────────┬─────────┘
                          │                     │
                ┌─────────▼─────────────────────▼─────────┐
                │      api/main.py  (FastAPI, both modes)   │
                │      frontend/  (React + Tailwind, /app)  │
                │      dashboard/index.html  (fallback UI)  │
                │      deployment/generate_package.py        │
                │      MLflow tracking (per leaderboard run) │
                └───────────────────────────────────────────┘
```

## Setup

**Backend:**
```bash
pip install -r requirements.txt   # fastapi, uvicorn, scikit-learn, xgboost,
                                   # lightgbm, catboost, imbalanced-learn,
                                   # psycopg2-binary, mlflow, etc.
```

**Frontend (React console):**
```bash
cd frontend
npm install
npm run build      # produces frontend/dist, served by the backend at /app
```

**Database — SQLite (zero config) or PostgreSQL:**
The registry (`registry/model_registry.py`) uses SQLite by default. To use
PostgreSQL instead, set `DATABASE_URL`:
```bash
export DATABASE_URL="postgresql://aimlp:aimlp@localhost:5432/aimlp_registry"
```
This was tested against a real local PostgreSQL instance during
development (registered models, rolled back, and independently confirmed
via `psql` directly — not just reviewed as SQL).

**Experiment tracking — MLflow:**
Every Mode 1 leaderboard run logs each candidate as an MLflow run
(algorithm, metrics, overall score) under an experiment named
`aimlp_<project_name>`. No setup needed — MLflow defaults to a local
`mlflow.db` SQLite file. To view the tracking UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```
(`--workers 1` matters — the default of 4 workers repeatedly failed to
bind its port during development; see limitations below.)

## Run

```bash
uvicorn api.main:app --reload --port 8000
```

Then open one of:
- **`http://localhost:8000/app/`** — the React console (primary UI)
- **`http://localhost:8000/dashboard/`** — the original dependency-free HTML/JS console (kept as a lighter-weight fallback that needs no build step)

Or drive it directly via HTTP:
```bash
# Mode 1: train from scratch
curl -X POST http://localhost:8000/mode1/train \
  -F "file=@your_data.csv" -F "project_name=my_project"

# Download the generated PDF report or deployment package
curl -O http://localhost:8000/mode1/download-report/my_project
curl -O http://localhost:8000/mode1/download-deployment/my_project

# Mode 2: evaluate + evolve the active model against new data
curl -X POST http://localhost:8000/mode2/evolve \
  -F "project_name=my_project" -F "latest_file=@latest_data.csv"

# Registry
curl http://localhost:8000/registry/my_project/versions
curl -X POST http://localhost:8000/registry/my_project/rollback/2
```

**MLflow tracking UI:**
```bash
python run_mlflow_ui.py --port 5000
```
Uses a direct-Flask-app launcher instead of the `mlflow ui` CLI -- the
CLI's multi-worker wrapper repeatedly failed to bind its port during
development; calling the underlying Flask app directly works. Verified
with a real screenshot showing the home page and experiment list
rendering actual logged run data. One caveat found by testing, not
resolved: drilling into an individual run's *detail* page hits a
client-side error in this MLflow version with no failed network request
behind it -- looks like an MLflow 3.x UI bug independent of the launch
method, not something fixable from this project.

**Docker:** written correctly (multi-stage build, 3 services), but
**genuinely not run** -- see limitations below for exactly why, including
a real attempt that hit a hard wall.

## What's genuinely tested vs. not

Everything below was tested against real behavior (not just "the code runs
without an exception") — including headless-browser-driven passes through
both consoles, live PostgreSQL, and MLflow's own client API for
independent verification:

- Dataset analysis on 3 different dataset shapes (fraud, churn, synthetic
  regression) — caught and fixed a real target-column detection bug
- Preprocessing + SMOTE — confirmed SMOTE triggers only on genuine
  imbalance, confirmed it correctly skips when not needed
- Candidate selection — confirmed it produces different shortlists (and
  different reasoning) for large/imbalanced vs. small/balanced datasets,
  and correctly falls back to evaluating everything when no rule fires
- Leaderboard across 12 algorithms including XGBoost/LightGBM/CatBoost —
  caught and fixed a metric-averaging bug that let a fraud-blind model look
  fine under "weighted" averaging
- Health score + decision engine — tested both branches (retrain-same vs.
  replace-model) on real fraud data, confirmed both fire correctly
- Model registry — versioning, rollback, tested against SQLite **and**
  against a live local PostgreSQL instance (registered, rolled back,
  confirmed via a raw `psql` query independent of this codebase)
- MLflow tracking (data layer) — every leaderboard run logs real
  params/metrics; retrieved 12 real logged runs back via `MlflowClient`
  independently of the logging code itself
- **MLflow tracking UI** — root-caused the CLI wrapper's port-binding
  failures and fixed by calling `mlflow.server.app` directly; verified
  with a real headless-browser screenshot showing actual logged
  experiments. Also found and fixed a real consistency bug in the
  process: the default tracking URI was relative to the *server process's*
  working directory, so runs logged while running the app from one
  directory were invisible when queried from another. Now anchored to an
  absolute path at the project root.
- **Health score / drift weighting** — the tension described in earlier
  passes (High drift could be masked by strong performance) is now fixed:
  drift severity caps the category (High forces at most "Average") rather
  than just contributing 20% of a blend that performance could dominate.
  Verified against the exact scenario that originally exposed the issue:
  the same inputs that previously produced "Good" health despite 45% of
  features drifting now correctly produce "Average," which correctly
  routes through candidate evaluation instead of blindly trusting the
  current algorithm.
- **Prediction confidence** — previously a stand-in equal to the
  performance score; now computed independently from real `predict_proba`
  output (mean probability of the predicted class). Verified the number is
  now genuinely different from performance rather than mirroring it.
- **PDF report generation** — built using reportlab, wired into
  `/mode1/train` and a new `/mode1/download-report/{project}` endpoint.
  Verified by converting the actual generated PDF to an image and visually
  confirming the tables, headings, and pagination render correctly --not
  just checking the file was created.
- Deployment package generator — the generated FastAPI app was started
  and it served real predictions, not just files that look plausible
- Platform API — Mode 1 and Mode 2 both tested over real HTTP, including a
  drift-comparison bug caught specifically because this was tested
  end-to-end rather than assumed correct
- **React console** — built with Vite (`npm run build` succeeds cleanly),
  mounted into the FastAPI backend, and driven through a real headless
  browser (Playwright): filled in a project name, uploaded a CSV, clicked
  Train, waited for real results, uploaded a second CSV, clicked Evolve,
  clicked Rollback — all against the live backend, screenshotted at each
  step. Caught and fixed a real bug in the process (Vite's asset paths
  didn't account for being served from `/app/` instead of `/`, which
  produced a blank page until fixed).
- Original HTML/JS console — same browser-driven testing, still included
  as a zero-build-step fallback

**Not done, with the specific reason for each:**
- **Docker Compose was never run.** Worth being precise here: a real
  Docker daemon WAS installed and successfully started in the development
  sandbox (`dockerd` ran, `docker ps` worked) -- new progress from earlier
  passes. But `docker build` failed at the very first step: pulling the
  base image (`node:20-slim`) returned a 403 Forbidden, because this
  sandbox's network allowlist only permits specific domains (pypi, npm,
  GitHub, a few others) with no container registry on it. This is a
  network policy wall, not something fixable with a different flag or
  config from inside the sandbox. The Dockerfile and compose file are
  correct and should build fine in a normal environment with unrestricted
  registry access.
- MLflow run-detail page (see above) -- tracking data and the experiment
  list both work; that one specific view doesn't, and it isn't clear the
  fix is on this project's side.
- CatBoost is installed and used, but adds meaningful time (~3-15s
  depending on dataset size) to leaderboard runs — noticeable in the
  per-candidate timing you'll see.

## File structure

```
aimlp/
├── api/main.py                  # FastAPI: Mode 1, Mode 2, registry endpoints
├── core/
│   ├── dataset_analysis.py      # profile any CSV: target, type, quality score
│   ├── preprocessing.py         # impute/encode/scale/SMOTE
│   ├── candidate_selector.py    # intelligent algorithm shortlisting
│   ├── trainer.py               # leaderboard across 12 algorithms + MLflow logging
│   ├── health_score.py          # composite health scoring
│   ├── drift_detector.py        # DDM + KS-test, generalized from drift_pipeline
│   ├── decision_engine.py       # Mode 2: retrain-same vs. replace decision
│   └── mode1_pipeline.py        # ties it all together for Mode 1
├── registry/model_registry.py   # SQLite or PostgreSQL version registry, rollback
├── deployment/
│   ├── generate_package.py   # model.pkl + preprocessing.pkl + FastAPI + Docker
│   └── report_generator.py    # PDF training report (reportlab)
├── run_mlflow_ui.py            # working MLflow UI launcher (bypasses the flaky CLI)
├── frontend/                     # React + Vite + Tailwind console (build with npm)
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── components/           # HealthGauge, Leaderboard, ProfilePanel, etc.
├── dashboard/index.html          # fallback console, no build step needed
├── infra/
│   ├── Dockerfile                # multi-stage: builds frontend, bundles into API image
│   └── docker-compose.yml        # api + postgres + mlflow
├── sample_data/make_samples.py   # synthetic churn + house-price test datasets
└── requirements.txt
```

"# AIMLP-Adaptive-AI-Model-Lifecycle-Platform" 
