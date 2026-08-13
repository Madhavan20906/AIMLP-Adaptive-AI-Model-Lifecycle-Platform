"""
registry/model_registry.py

Tracks every model version ever produced by Mode 1 or Mode 2: what
algorithm, what metrics, what dataset it was trained on, when, and
whether it's currently the active/deployed version. Supports rollback --
"deploy" just flips which version is marked active, it doesn't delete
history.

Backend: PostgreSQL when DATABASE_URL is set (the real target per the
platform's tech stack), falling back to local SQLite when it isn't --
so this still runs with zero setup for anyone who hasn't stood up
Postgres. Both backends were tested against a live database, not just
reviewed as SQL: SQLite always, Postgres against an actual local
instance during development (see infra/docker-compose.yml for the
containerized version of the same database).
"""

import os
import sqlite3
import time
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "registry.db"
MODELS_DIR = Path(__file__).parent / "model_store"
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_POSTGRES = DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras


def get_conn():
    if USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH))


def _placeholder():
    """Postgres uses %s, sqlite3 uses ?."""
    return "%s" if USE_POSTGRES else "?"


def init_registry(reset=False):
    if not USE_POSTGRES and reset and DB_PATH.exists():
        DB_PATH.unlink()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_conn()
    cur = conn.cursor()

    if USE_POSTGRES:
        if reset:
            cur.execute("DROP TABLE IF EXISTS model_versions")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id SERIAL PRIMARY KEY,
                project_name TEXT,
                algorithm TEXT,
                problem_type TEXT,
                metrics_json TEXT,
                overall_score REAL,
                dataset_metadata_json TEXT,
                training_time REAL,
                created_at DOUBLE PRECISION,
                is_active INTEGER DEFAULT 0,
                source TEXT,
                model_path TEXT,
                notes TEXT
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT,
                algorithm TEXT,
                problem_type TEXT,
                metrics_json TEXT,
                overall_score REAL,
                dataset_metadata_json TEXT,
                training_time REAL,
                created_at REAL,
                is_active INTEGER DEFAULT 0,
                source TEXT,
                model_path TEXT,
                notes TEXT
            )
        """)
    conn.commit()
    conn.close()


def register_model(
    project_name, algorithm, problem_type, metrics: dict, overall_score,
    dataset_metadata: dict, training_time, model_object, source="mode1_initial_training",
    notes="", activate=True, extra: dict = None,
):
    """Saves the model artifact to disk and records it in the registry.
    If `activate`, deactivates any previously-active version for this
    project and marks this one active (deployment).

    `extra`, if given, is a dict (e.g. {"preprocessor":..., "label_encoder":...,
    "X_ref":..., "y_ref":...}) saved alongside the model as a sibling file.
    This is how Mode 2 gets access to the EXACT preprocessor used at
    training time and a clean (never-SMOTE'd) reference sample for drift
    comparison -- comparing against a fresh train/test split of the new
    data (instead of stored history) was an earlier bug: it made a SMOTE-
    augmented split look "100% drifted" against its own untouched sibling
    split, which wasn't measuring real drift at all.
    """
    import joblib

    conn = get_conn()
    cur = conn.cursor()
    p = _placeholder()

    if USE_POSTGRES:
        cur.execute(
            f"INSERT INTO model_versions (project_name, algorithm, problem_type, metrics_json, "
            f"overall_score, dataset_metadata_json, training_time, created_at, is_active, source, model_path, notes) "
            f"VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p}) RETURNING version_id",
            (
                project_name, algorithm, problem_type, json.dumps(metrics), overall_score,
                json.dumps(dataset_metadata), training_time, time.time(), 0, source, "", notes,
            ),
        )
        version_id = cur.fetchone()[0]
    else:
        cur.execute(
            f"INSERT INTO model_versions (project_name, algorithm, problem_type, metrics_json, "
            f"overall_score, dataset_metadata_json, training_time, created_at, is_active, source, model_path, notes) "
            f"VALUES ({p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p},{p})",
            (
                project_name, algorithm, problem_type, json.dumps(metrics), overall_score,
                json.dumps(dataset_metadata), training_time, time.time(), 0, source, "", notes,
            ),
        )
        version_id = cur.lastrowid

    model_path = MODELS_DIR / f"{project_name}_v{version_id}.joblib"
    joblib.dump(model_object, model_path)
    cur.execute(f"UPDATE model_versions SET model_path = {p} WHERE version_id = {p}", (str(model_path), version_id))

    if extra is not None:
        extra_path = MODELS_DIR / f"{project_name}_v{version_id}_extra.joblib"
        joblib.dump(extra, extra_path)

    conn.commit()

    if activate:
        deploy_version(project_name, version_id, conn=conn)

    conn.close()
    return version_id


def load_active_extra(project_name):
    """Loads the companion bundle (preprocessor, label_encoder, reference
    sample) saved alongside the currently-active model, if any."""
    import joblib
    active = get_active_version(project_name)
    if not active:
        return None
    extra_path = Path(active["model_path"]).with_name(
        Path(active["model_path"]).stem + "_extra.joblib"
    )
    if not extra_path.exists():
        return None
    return joblib.load(extra_path)


def deploy_version(project_name, version_id, conn=None):
    """Marks a version as the active/deployed one for a project. This is
    also how rollback works: deploy_version(project, an_older_version_id)."""
    own_conn = conn is None
    if own_conn:
        conn = get_conn()
    p = _placeholder()
    cur = conn.cursor()
    cur.execute(f"UPDATE model_versions SET is_active = 0 WHERE project_name = {p}", (project_name,))
    cur.execute(f"UPDATE model_versions SET is_active = 1 WHERE version_id = {p}", (version_id,))
    conn.commit()
    if own_conn:
        conn.close()


def _rows_to_dicts(cur, rows):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def get_active_version(project_name):
    conn = get_conn()
    p = _placeholder()
    cur = conn.cursor()
    cur.execute(
        f"SELECT * FROM model_versions WHERE project_name = {p} AND is_active = 1", (project_name,)
    )
    row = cur.fetchone()
    result = None
    if row:
        cols = [d[0] for d in cur.description]
        result = dict(zip(cols, row))
    conn.close()
    return result


def load_active_model(project_name):
    import joblib
    active = get_active_version(project_name)
    if not active:
        return None, None
    return joblib.load(active["model_path"]), active


def list_versions(project_name):
    conn = get_conn()
    p = _placeholder()
    cur = conn.cursor()
    cur.execute(
        f"SELECT * FROM model_versions WHERE project_name = {p} ORDER BY version_id DESC", (project_name,)
    )
    rows = cur.fetchall()
    result = _rows_to_dicts(cur, rows)
    conn.close()
    return result


def rollback(project_name, target_version_id):
    """Convenience wrapper: rollback IS just deploying an older version."""
    versions = {v["version_id"] for v in list_versions(project_name)}
    if target_version_id not in versions:
        raise ValueError(f"version {target_version_id} not found for project {project_name}")
    deploy_version(project_name, target_version_id)
    return get_active_version(project_name)
