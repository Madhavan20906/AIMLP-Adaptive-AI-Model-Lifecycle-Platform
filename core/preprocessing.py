"""
core/preprocessing.py

Turns a raw, messy CSV into model-ready train/test arrays, automatically,
based on the profile produced by dataset_analysis.py. No manual column
specification needed beyond (optionally) the target column.

Steps: duplicate removal -> train/test split -> impute -> encode -> scale
-> outlier clipping -> SMOTE (classification + imbalance only, train set
only, never on test set -- SMOTE-ing the test set would leak synthetic
samples into evaluation and inflate every metric).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


def _clip_outliers(X: pd.DataFrame, numerical_features, iqr_multiplier=3.0) -> pd.DataFrame:
    """Clips (not drops) extreme outliers in numeric columns using the IQR rule.
    Clipping instead of dropping preserves row count / alignment with y."""
    X = X.copy()
    for col in numerical_features:
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        X[col] = X[col].clip(lower, upper)
    return X


def build_pipeline(profile: dict) -> ColumnTransformer:
    """Builds the sklearn ColumnTransformer for impute+encode+scale, reusable
    at inference time on new data (this is what gets pickled as preprocessing.pkl)."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if profile["numerical_features"]:
        transformers.append(("num", numeric_pipe, profile["numerical_features"]))
    if profile["categorical_features"]:
        transformers.append(("cat", categorical_pipe, profile["categorical_features"]))

    return ColumnTransformer(transformers, remainder="drop")


def preprocess(df: pd.DataFrame, profile: dict, test_size=0.2, seed=42, use_smote="auto"):
    """
    Returns: X_train, X_test, y_train, y_test, fitted_preprocessor, label_encoder (or None)

    use_smote: "auto" applies SMOTE only for classification with
    imbalance_ratio > 3; True/False force it on/off.
    """
    df = df.drop_duplicates().reset_index(drop=True)

    target_col = profile["target_column"]
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    X = _clip_outliers(X, profile["numerical_features"])

    label_encoder = None
    if profile["problem_type"] == "classification" and not pd.api.types.is_numeric_dtype(y_raw):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = y_raw.values

    stratify = y if profile["problem_type"] == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    preprocessor = build_pipeline(profile)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    should_smote = use_smote
    if use_smote == "auto":
        should_smote = (
            profile["problem_type"] == "classification"
            and profile.get("imbalance_ratio") is not None
            and profile["imbalance_ratio"] > 3
        )

    if should_smote:
        from imblearn.over_sampling import SMOTE
        try:
            sm = SMOTE(random_state=seed)
            X_train_proc, y_train = sm.fit_resample(X_train_proc, y_train)
        except ValueError:
            # SMOTE needs at least k_neighbors+1 samples in the minority
            # class -- if the dataset is too small/imbalanced for that,
            # skip it rather than crash the whole pipeline.
            pass

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, label_encoder
