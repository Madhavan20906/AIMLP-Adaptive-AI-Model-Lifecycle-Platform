"""
core/dataset_analysis.py

Given ANY csv (the platform must work on datasets it's never seen before,
not just fraud), this module figures out:
    - target column (if not specified)
    - problem type (classification vs regression)
    - categorical vs numerical features
    - missing values, duplicates, class imbalance, correlation
    - a single "data quality score" summarizing all of the above

This profile is what candidate_selector.py reads to decide which
algorithms are even worth trying -- so the accuracy of this module
directly determines how smart the "intelligent shortlisting" claim is.
"""

import numpy as np
import pandas as pd


COMMON_TARGET_NAMES = [
    "target", "label", "class", "y", "outcome", "result",
    "churn", "fraud", "is_fraud", "default", "diagnosis", "response",
]


def _guess_target_column(df: pd.DataFrame) -> str:
    """
    Heuristic target detection when the user doesn't specify one:
    1. Exact match (case-insensitive) against common target-column names --
       checked first and takes priority.
    2. Substring match, but ONLY for names 4+ characters long. Short names
       like "y" or "class" as plain substrings would false-positive against
       almost any column ("y" matches "monthly_charges", "type", etc.) --
       this bit its own author during testing, which is why the guard exists.
    3. Otherwise fall back to the last column (common CSV convention).
    """
    lower_cols = {c.lower(): c for c in df.columns}

    for name in COMMON_TARGET_NAMES:
        if name in lower_cols:
            return lower_cols[name]

    for name in COMMON_TARGET_NAMES:
        if len(name) < 4:
            continue
        for lc, original in lower_cols.items():
            if name in lc:
                return original

    return df.columns[-1]


def _detect_problem_type(y: pd.Series) -> str:
    """
    Classification if the target is non-numeric, or numeric with few
    distinct values relative to the number of rows (typical label
    encoding). Regression otherwise.
    """
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    n_unique = y.nunique(dropna=True)
    ratio = n_unique / max(len(y), 1)

    if n_unique <= 20 or ratio < 0.05:
        return "classification"
    return "regression"


def _data_quality_score(missing_ratio, duplicate_ratio, imbalance_ratio, n_rows) -> float:
    """
    0-100 composite score. Penalizes missingness, duplication, class
    imbalance, and very small sample sizes. This is intentionally simple
    and explainable rather than a black-box heuristic -- every penalty
    below is traceable to a specific, named data problem.
    """
    score = 100.0
    score -= min(40, missing_ratio * 100 * 0.8)       # up to -40 for missing data
    score -= min(20, duplicate_ratio * 100 * 0.5)      # up to -20 for duplicates
    if imbalance_ratio is not None:
        score -= min(20, max(0, (imbalance_ratio - 1) * 2))  # up to -20 for imbalance
    if n_rows < 500:
        score -= 15
    elif n_rows < 2000:
        score -= 5
    return round(max(0.0, min(100.0, score)), 1)


def analyze_dataset(df: pd.DataFrame, target_column: str = None) -> dict:
    """
    Returns a full profile dict used by candidate_selector, preprocessing,
    and the dashboard's "Dataset Analysis" panel.
    """
    df = df.copy()
    n_rows, n_cols = df.shape

    target_column = target_column or _guess_target_column(df)
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataset columns: {list(df.columns)}")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    problem_type = _detect_problem_type(y)

    categorical_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numerical_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

    missing_per_col = X.isna().sum()
    total_missing = int(missing_per_col.sum())
    missing_ratio = total_missing / (n_rows * max(n_cols - 1, 1))

    n_duplicates = int(df.duplicated().sum())
    duplicate_ratio = n_duplicates / max(n_rows, 1)

    class_imbalance = None
    imbalance_ratio = None
    if problem_type == "classification":
        counts = y.value_counts()
        class_imbalance = counts.to_dict()
        if len(counts) >= 2:
            imbalance_ratio = float(counts.max() / max(counts.min(), 1))

    # correlation among numeric features (informs candidate selection: high
    # correlation -> linear models degrade, tree ensembles handle it fine)
    high_corr_pairs = []
    if len(numerical_features) >= 2:
        corr = X[numerical_features].corr(numeric_only=True).abs()
        for i, c1 in enumerate(numerical_features):
            for c2 in numerical_features[i + 1:]:
                v = corr.loc[c1, c2]
                if pd.notna(v) and v > 0.85:
                    high_corr_pairs.append((c1, c2, round(float(v), 3)))

    quality_score = _data_quality_score(missing_ratio, duplicate_ratio, imbalance_ratio, n_rows)

    profile = {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "target_column": target_column,
        "problem_type": problem_type,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "n_categorical_features": len(categorical_features),
        "n_numerical_features": len(numerical_features),
        "categorical_feature_pct": round(len(categorical_features) / max(len(X.columns), 1) * 100, 1),
        "missing_values_total": total_missing,
        "missing_ratio": round(missing_ratio, 4),
        "duplicates": n_duplicates,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "class_imbalance": class_imbalance,
        "imbalance_ratio": round(imbalance_ratio, 2) if imbalance_ratio else None,
        "high_correlation_pairs": high_corr_pairs,
        "dataset_size_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        "data_quality_score": quality_score,
    }
    return profile


def print_profile(profile: dict):
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    for k, v in profile.items():
        print(f"{k:28}: {v}")
