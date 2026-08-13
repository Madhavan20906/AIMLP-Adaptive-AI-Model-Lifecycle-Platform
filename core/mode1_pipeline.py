"""
core/mode1_pipeline.py

Mode 1: Initial Model Creation.
Ties dataset_analysis -> preprocessing -> candidate_selector -> trainer ->
model_registry into one call. This is "upload a CSV, get a trained
production model back."
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from core.dataset_analysis import analyze_dataset
from core.preprocessing import preprocess
from core.candidate_selector import select_candidates
from core.trainer import run_leaderboard
import registry.model_registry as reg


def run_mode1(csv_path: str, project_name: str, target_column: str = None, max_candidates: int = 5):
    t_start = time.time()

    df = pd.read_csv(csv_path)
    profile = analyze_dataset(df, target_column=target_column)

    X_train, X_test, y_train, y_test, preprocessor, label_encoder = preprocess(df, profile)

    candidates, selection_reasons, confidence, evaluate_all = select_candidates(
        profile, max_candidates=max_candidates
    )

    leaderboard = run_leaderboard(candidates, X_train, y_train, X_test, y_test, profile["problem_type"],
                                   mlflow_experiment=f"aimlp_{project_name}")
    ok_results = [r for r in leaderboard if r["status"] == "ok"]

    if not ok_results:
        raise RuntimeError("Every candidate model failed to train -- check dataset quality/format.")

    best = ok_results[0]
    total_time = round(time.time() - t_start, 2)

    reg.init_registry()
    version_id = reg.register_model(
        project_name=project_name,
        algorithm=best["name"],
        problem_type=profile["problem_type"],
        metrics=best["metrics"],
        overall_score=best["overall_score"],
        dataset_metadata={
            "csv_path": str(csv_path),
            "n_rows": profile["n_rows"],
            "n_columns": profile["n_columns"],
            "target_column": profile["target_column"],
        },
        training_time=total_time,
        model_object=best["model"],
        source="mode1_initial_training",
        notes=f"Best of {len(ok_results)} evaluated candidates",
        extra={
            "preprocessor": preprocessor,
            "label_encoder": label_encoder,
            # X_test/y_test, not X_train: this split was NEVER touched by
            # SMOTE, so it's a clean, real sample of the actual data
            # distribution -- the correct baseline for Mode 2 drift
            # comparison later. Using a SMOTE-augmented split as the
            # "reference" was the bug that made a dataset look 100%
            # drifted against its own untouched sibling split.
            "X_ref": X_test,
            "y_ref": y_test,
        },
    )

    return {
        "project_name": project_name,
        "version_id": version_id,
        "profile": profile,
        "candidate_selection": {
            "candidates_evaluated": list(candidates.keys()),
            "reasons": selection_reasons,
            "confidence": confidence,
            "evaluated_all": evaluate_all,
        },
        "leaderboard": [
            {"rank": r.get("rank"), "name": r["name"], "overall_score": r.get("overall_score"),
             "metrics": r.get("metrics"), "train_time_s": r.get("train_time_s"),
             "model_size_kb": r.get("model_size_kb")}
            for r in leaderboard if r["status"] == "ok"
        ],
        "failed_candidates": [{"name": r["name"], "error": r["error"]} for r in leaderboard if r["status"] != "ok"],
        "best_model": {
            "name": best["name"],
            "metrics": best["metrics"],
            "overall_score": best["overall_score"],
        },
        "best_model_object": best["model"],
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "reference_sample": {"X_ref": X_test, "y_ref": y_test},
        "total_time_s": total_time,
    }


if __name__ == "__main__":
    import json
    result = run_mode1("sample_data/churn_sample.csv", project_name="churn_demo")
    printable = {k: v for k, v in result.items()
                 if k not in ("preprocessor", "label_encoder", "best_model_object", "reference_sample")}
    print(json.dumps(printable, indent=2, default=str))
