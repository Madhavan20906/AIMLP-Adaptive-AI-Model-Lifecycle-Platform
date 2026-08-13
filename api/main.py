"""
api/main.py

The platform's main API. Wraps Mode 1 (train from scratch) and Mode 2
(evaluate + evolve a production model) as HTTP endpoints, backed by the
model registry.

Run:
    uvicorn api.main:app --reload --port 8000
"""

import sys
import shutil
import tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

from core.mode1_pipeline import run_mode1
from core.dataset_analysis import analyze_dataset
from core.preprocessing import preprocess
from core.candidate_selector import select_candidates
from core.decision_engine import evolve_model
from deployment.generate_package import generate_deployment_package
from deployment.report_generator import generate_mode1_report
import registry.model_registry as reg

app = FastAPI(title="AIMLP — Adaptive AI Model Lifecycle Platform", version="1.0.0")

reg.init_registry()

UPLOAD_DIR = Path(tempfile.gettempdir()) / "aimlp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


@app.get("/")
def root():
    return {"message": "AIMLP is running.", "modes": ["/mode1/train", "/mode2/evolve"], "docs": "/docs"}


@app.post("/mode1/analyze")
async def analyze(file: UploadFile = File(...), target_column: Optional[str] = Form(None)):
    """Dataset analysis only -- lets a frontend show the profile before committing to training."""
    path = UPLOAD_DIR / file.filename
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    df = pd.read_csv(path)
    profile = analyze_dataset(df, target_column=target_column)
    candidates, reasons, confidence, evaluate_all = select_candidates(profile)
    return {
        "profile": profile,
        "candidate_preview": {
            "candidates": list(candidates.keys()),
            "reasons": reasons,
            "confidence": confidence,
            "evaluate_all": evaluate_all,
        },
    }


@app.post("/mode1/train")
async def train(
    file: UploadFile = File(...),
    project_name: str = Form(...),
    target_column: Optional[str] = Form(None),
):
    """Mode 1: full pipeline, dataset in -> best model registered + deployment package out."""
    path = UPLOAD_DIR / file.filename
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = run_mode1(str(path), project_name=project_name, target_column=target_column)
    best = result["best_model"]

    zip_path = generate_deployment_package(
        project_name=project_name,
        algorithm=best["name"],
        problem_type=result["profile"]["problem_type"],
        model_object=result["best_model_object"],
        preprocessor=result["preprocessor"],
        metrics=best["metrics"],
        overall_score=best["overall_score"],
        label_encoder=result["label_encoder"],
    )

    report_path = generate_mode1_report(
        project_name=project_name,
        profile=result["profile"],
        candidate_selection=result["candidate_selection"],
        leaderboard=result["leaderboard"],
        best_model=result["best_model"],
        total_time_s=result["total_time_s"],
    )

    return {
        "project_name": project_name,
        "version_id": result["version_id"],
        "profile": result["profile"],
        "candidate_selection": result["candidate_selection"],
        "leaderboard": result["leaderboard"],
        "best_model": result["best_model"],
        "deployment_package_path": str(zip_path),
        "report_path": str(report_path),
        "total_time_s": result["total_time_s"],
    }


@app.get("/mode1/download-deployment/{project_name}")
def download_deployment(project_name: str):
    zip_path = Path(f"./deployment_output/{project_name}.zip")
    if not zip_path.exists():
        return {"error": f"No deployment package found for '{project_name}'. Run /mode1/train first."}
    return FileResponse(zip_path, filename=f"{project_name}_deployment.zip")


@app.get("/mode1/download-report/{project_name}")
def download_report(project_name: str):
    pdf_path = Path(f"./deployment_output/{project_name}_report.pdf")
    if not pdf_path.exists():
        return {"error": f"No report found for '{project_name}'. Run /mode1/train first."}
    return FileResponse(pdf_path, filename=f"{project_name}_report.pdf")


@app.post("/mode2/evolve")
async def evolve(
    project_name: str = Form(...),
    latest_file: UploadFile = File(...),
    switch_threshold: float = Form(3.0),
):
    """
    Mode 2: loads the project's currently-active model AND the exact
    preprocessor/reference-sample saved alongside it at training time,
    then evaluates against the newly uploaded data using that SAME
    preprocessor -- not a freshly-fit one. This matters: comparing two
    independently-fit splits of the same new file (an earlier version of
    this endpoint did that) isn't measuring drift at all, since a fresh
    train/test split of one file looks arbitrarily different once one
    half has been SMOTE-augmented and the other hasn't. Real drift
    comparison needs a fixed, untouched reference from the past.
    """
    import json as _json

    current_model, active_meta = reg.load_active_model(project_name)
    if current_model is None:
        return {"error": f"No active model found for project '{project_name}'. Run /mode1/train first."}

    extra = reg.load_active_extra(project_name)
    if extra is None:
        return {"error": f"No stored preprocessor/reference sample for '{project_name}' "
                          f"(model was registered before this feature existed). Re-run /mode1/train."}

    dataset_meta = _json.loads(active_meta["dataset_metadata_json"])
    target_column = dataset_meta.get("target_column")

    path = UPLOAD_DIR / latest_file.filename
    with open(path, "wb") as f:
        shutil.copyfileobj(latest_file.file, f)
    df_latest = pd.read_csv(path)

    profile = analyze_dataset(df_latest, target_column=target_column)

    y_latest_raw = df_latest[target_column]
    X_latest_raw = df_latest.drop(columns=[target_column])

    preprocessor = extra["preprocessor"]
    label_encoder = extra.get("label_encoder")
    X_latest = preprocessor.transform(X_latest_raw)
    if label_encoder is not None and not pd.api.types.is_numeric_dtype(y_latest_raw):
        y_latest = label_encoder.transform(y_latest_raw)
    else:
        y_latest = y_latest_raw.values

    X_ref, y_ref = extra["X_ref"], extra["y_ref"]

    candidates, reasons, confidence, evaluate_all = select_candidates(profile)

    algorithm_name = active_meta["algorithm"]
    from core.candidate_selector import _all_classifiers, _all_regressors
    pool = _all_classifiers() if profile["problem_type"] == "classification" else _all_regressors()
    estimator_class = (lambda name=algorithm_name: type(pool[name])()) if algorithm_name in pool else (lambda: type(current_model)())

    explanation = evolve_model(
        current_model=current_model,
        model_name=algorithm_name,
        estimator_class=estimator_class,
        X_ref=X_ref, y_ref=y_ref,
        X_latest=X_latest, y_latest=y_latest,
        profile=profile,
        candidates=candidates,
        switch_threshold=switch_threshold,
    )

    new_model = explanation.pop("new_model_object", None)
    version_id = None
    if new_model is not None:
        version_id = reg.register_model(
            project_name=project_name,
            algorithm=explanation["chosen_model"],
            problem_type=profile["problem_type"],
            metrics=explanation["chosen_model_metrics"],
            overall_score=explanation["chosen_overall_score"],
            dataset_metadata={"n_rows": profile["n_rows"], "source_file": latest_file.filename,
                               "target_column": target_column},
            training_time=0,
            model_object=new_model,
            source="mode2_adaptive_evolution",
            notes=explanation["decision"],
            activate=True,
            # carry the SAME preprocessor forward (retrained model still
            # expects the same input feature space), but refresh the
            # reference sample to the data just evaluated -- that becomes
            # the new baseline for the NEXT evolve() call.
            extra={"preprocessor": preprocessor, "label_encoder": label_encoder, "X_ref": X_latest, "y_ref": y_latest},
        )

    explanation["new_version_id"] = version_id
    return explanation


@app.get("/registry/{project_name}/versions")
def get_versions(project_name: str):
    return {"versions": reg.list_versions(project_name)}


@app.post("/registry/{project_name}/rollback/{version_id}")
def rollback(project_name: str, version_id: int):
    active = reg.rollback(project_name, version_id)
    return {"message": f"Rolled back to version {version_id}", "active_version": active}
