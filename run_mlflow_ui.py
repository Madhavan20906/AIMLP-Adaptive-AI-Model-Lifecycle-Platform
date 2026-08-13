"""
run_mlflow_ui.py

Launches the MLflow tracking UI directly against mlflow.server.app
(a Flask app), bypassing the `mlflow ui` / `mlflow server` CLI's
subprocess-spawning wrapper (gunicorn/uvicorn multi-worker model).

Why this exists: in the sandbox this project was developed in, `mlflow ui`
consistently failed to bind its port across several worker configurations
(4 workers default, 1 worker, different host bindings) -- the CLI wrapper
spawns worker processes in a way that didn't work there. Calling the
underlying Flask app directly with `app.run()` works reliably.

Verified: the home page and experiment list render correctly with real
logged run data (confirmed via a real headless-browser screenshot, not
just an HTTP 200). One caveat found the same way: drilling into an
individual experiment's run-detail page hits a client-side error in this
MLflow version ("Something went wrong") with no failed network request
behind it -- looks like an MLflow 3.x UI bug independent of this launch
method, not something fixable from here. The experiment list, metrics,
and run data are all genuinely queryable via the API regardless (see
core/trainer.py's MLflow integration, or query the API directly:
POST /api/2.0/mlflow/experiments/search).

Usage:
    python run_mlflow_ui.py [--port 5000]
"""
import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    db_path = project_root / "mlflow.db"
    artifacts_path = project_root / "mlartifacts"

    os.environ["_MLFLOW_SERVER_FILE_STORE"] = f"sqlite:///{db_path}"
    os.environ["_MLFLOW_SERVER_REGISTRY_STORE"] = f"sqlite:///{db_path}"
    os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = str(artifacts_path)

    import mlflow.server as server
    print(f"Serving MLflow UI at http://{args.host}:{args.port}/  (tracking db: {db_path})")
    server.app.run(host=args.host, port=args.port, debug=False)
