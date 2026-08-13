"""
core/trainer.py

Trains each candidate model, evaluates it on a full metric set (not just
accuracy), and produces a ranked leaderboard with an explainable overall
score. This is what powers both Mode 1 (pick the best model from scratch)
and Mode 2 (compare a replacement candidate against the current model).
"""

import time
import tempfile
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)


def _model_size_kb(model) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        joblib.dump(model, tmp.name)
        size = os.path.getsize(tmp.name) / 1024
    os.unlink(tmp.name)
    return round(size, 1)


def _classification_metrics(model, X_test, y_test):
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    inference_time_ms = (time.perf_counter() - t0) * 1000 / max(len(X_test), 1)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        # macro (not weighted) average: weighted average lets a model that
        # completely ignores a rare minority class (e.g. 0.17% fraud) still
        # score ~99% just by being right on the majority class. Macro
        # treats every class equally, so ignoring the minority class
        # actually tanks the score -- which is the whole point when this
        # feeds a health score meant to catch exactly that failure mode.
        "precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "f1": f1_score(y_test, preds, average="macro", zero_division=0),
        "roc_auc": None,
    }

    n_classes = len(np.unique(y_test))
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                metrics["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
        except (ValueError, IndexError):
            pass

    return metrics, inference_time_ms


def _regression_metrics(model, X_test, y_test):
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    inference_time_ms = (time.perf_counter() - t0) * 1000 / max(len(X_test), 1)

    mse = mean_squared_error(y_test, preds)
    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }
    return metrics, inference_time_ms


def train_one(name, estimator, X_train, y_train, X_test, y_test, problem_type):
    """Trains + evaluates a single candidate. Never raises -- a candidate
    that fails to train (e.g. incompatible data) is recorded as failed
    rather than crashing the whole leaderboard run."""
    try:
        t0 = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0

        if problem_type == "classification":
            metrics, inference_ms = _classification_metrics(estimator, X_test, y_test)
        else:
            metrics, inference_ms = _regression_metrics(estimator, X_test, y_test)

        return {
            "name": name,
            "status": "ok",
            "model": estimator,
            "metrics": metrics,
            "train_time_s": round(train_time_s, 3),
            "inference_time_ms": round(inference_ms, 4),
            "model_size_kb": _model_size_kb(estimator),
        }
    except Exception as e:
        return {"name": name, "status": "failed", "error": str(e)}


def _overall_score(result: dict, problem_type: str) -> float:
    """
    Composite 0-100 score used to rank the leaderboard. Performance
    dominates; efficiency (train time, model size) contributes a small
    penalty so two similarly-accurate models don't rank identically
    regardless of cost.
    """
    m = result["metrics"]
    if problem_type == "classification":
        parts = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
        if m["roc_auc"] is not None:
            parts.append(m["roc_auc"])
        perf = float(np.mean(parts)) * 100
    else:
        # r2 can be negative for a bad model; clip so it doesn't blow up the score
        perf = max(0.0, min(1.0, m["r2"])) * 100

    efficiency_penalty = min(15.0, (result["train_time_s"] * 0.5) + (result["model_size_kb"] / 2000))
    return round(max(0.0, perf - efficiency_penalty), 2)


def run_leaderboard(candidates: dict, X_train, y_train, X_test, y_test, problem_type,
                     mlflow_experiment: str = None) -> list:
    """Trains every candidate, scores them, returns a leaderboard sorted best-first.

    If `mlflow_experiment` is given, each candidate's params/metrics are
    logged as a separate MLflow run under that experiment -- this is what
    lets you open the MLflow UI and compare candidates the way you'd
    compare any other experiment, not just read the leaderboard as a
    one-off table.
    """
    mlflow_enabled = False
    if mlflow_experiment:
        try:
            import mlflow
            # Explicit absolute path, not MLflow's cwd-relative default. The
            # default caused a real inconsistency during development: runs
            # logged while the server ran from one working directory were
            # invisible when queried from another -- separate untracked
            # mlflow.db files. Anchoring to the project root fixes that.
            project_root = Path(__file__).resolve().parents[1]
            mlflow.set_tracking_uri(f"sqlite:///{project_root / 'mlflow.db'}")
            mlflow.set_experiment(mlflow_experiment)
            mlflow_enabled = True
        except Exception:
            pass  # MLflow tracking is a nice-to-have; never let it block training

    results = []
    for name, estimator in candidates.items():
        r = train_one(name, estimator, X_train, y_train, X_test, y_test, problem_type)
        if r["status"] == "ok":
            r["overall_score"] = _overall_score(r, problem_type)

            if mlflow_enabled:
                try:
                    import mlflow
                    with mlflow.start_run(run_name=name):
                        mlflow.set_tag("problem_type", problem_type)
                        mlflow.log_param("algorithm", name)
                        for k, v in r["metrics"].items():
                            if v is not None:
                                mlflow.log_metric(k, float(v))
                        mlflow.log_metric("overall_score", r["overall_score"])
                        mlflow.log_metric("train_time_s", r["train_time_s"])
                        mlflow.log_metric("model_size_kb", r["model_size_kb"])
                except Exception:
                    pass
        results.append(r)

    ok_results = [r for r in results if r["status"] == "ok"]
    failed_results = [r for r in results if r["status"] != "ok"]

    ok_results.sort(key=lambda r: r["overall_score"], reverse=True)
    for rank, r in enumerate(ok_results, start=1):
        r["rank"] = rank

    return ok_results + failed_results


def print_leaderboard(leaderboard: list, problem_type: str):
    print("=" * 90)
    print("LEADERBOARD")
    print("=" * 90)
    for r in leaderboard:
        if r["status"] != "ok":
            print(f"  {r['name']:20} FAILED: {r['error']}")
            continue
        m = r["metrics"]
        if problem_type == "classification":
            metric_str = f"acc={m['accuracy']:.3f} f1={m['f1']:.3f} roc_auc={m['roc_auc']}"
        else:
            metric_str = f"r2={m['r2']:.3f} rmse={m['rmse']:.3f}"
        print(f"  #{r['rank']:<2} {r['name']:20} score={r['overall_score']:6.2f}  {metric_str}  "
              f"train={r['train_time_s']}s  size={r['model_size_kb']}KB")
