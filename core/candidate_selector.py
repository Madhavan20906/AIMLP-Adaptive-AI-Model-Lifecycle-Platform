"""
core/candidate_selector.py

The stated novelty of this platform: DON'T always train all 12 supported
algorithms. Look at what's actually in the dataset -- size, imbalance,
categorical %, missingness, feature count -- and shortlist a small set of
algorithms likely to do well, with a plain-English reason for each pick.

If nothing in the dataset strongly favors a subset (confidence stays low),
fall back to evaluating everything -- that's the safety net that keeps
"intelligent shortlisting" honest rather than just guessing.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


SEED = 42


def _all_classifiers() -> dict:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=SEED),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=150, random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "SVM": SVC(probability=True, random_state=SEED),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=SEED, verbosity=0)
    if _HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(random_state=SEED, verbosity=-1)
    if _HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(random_state=SEED, verbose=False)
    return models


def _all_regressors() -> dict:
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=SEED),
        "DecisionTree": DecisionTreeRegressor(random_state=SEED),
        "RandomForest": RandomForestRegressor(n_estimators=150, random_state=SEED),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=150, random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
        "AdaBoost": AdaBoostRegressor(random_state=SEED),
        "SVM": SVR(),
        "KNN": KNeighborsRegressor(),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(random_state=SEED, verbosity=0)
    if _HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(random_state=SEED, verbosity=-1)
    if _HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(random_state=SEED, verbose=False)
    return models


def select_candidates(profile: dict, max_candidates=5):
    """
    Returns (candidates: dict[name -> estimator], reasons: list[str],
    confidence: float 0-1, evaluate_all: bool)
    """
    problem_type = profile["problem_type"]
    all_models = _all_classifiers() if problem_type == "classification" else _all_regressors()

    n_rows = profile["n_rows"]
    imbalance_ratio = profile.get("imbalance_ratio")
    cat_pct = profile["categorical_feature_pct"]
    n_features = profile["n_numerical_features"] + profile["n_categorical_features"]
    missing_ratio = profile["missing_ratio"]

    scores = {name: 0 for name in all_models}
    reasons = []

    def boost(names, reason, weight=1):
        added = False
        for n in names:
            if n in scores:
                scores[n] += weight
                added = True
        if added:
            reasons.append(reason)

    # --- Rule 1: high imbalance -> boosted trees handle this natively better
    if problem_type == "classification" and imbalance_ratio and imbalance_ratio > 4:
        boost(["CatBoost", "LightGBM", "XGBoost"],
              f"High class imbalance (ratio {imbalance_ratio:.1f}:1) -> gradient boosting models "
              f"handle imbalance natively better than linear/distance-based models.", weight=3)

    # --- Rule 2: large dataset -> favor models that scale, avoid SVM/KNN (slow at scale)
    if n_rows > 20000:
        boost(["LightGBM", "RandomForest", "XGBoost"],
              f"Large dataset ({n_rows:,} rows) -> LightGBM/RandomForest/XGBoost scale efficiently; "
              f"SVM and KNN become slow at this size.", weight=2)
        scores["SVM"] -= 2
        scores["KNN"] -= 1

    # --- Rule 3: small dataset -> simpler models generalize better, avoid deep boosting
    elif n_rows < 2000:
        boost(["RandomForest", "LogisticRegression"] if problem_type == "classification"
              else ["RandomForest", "Ridge", "LinearRegression"],
              f"Small dataset ({n_rows:,} rows) -> simpler models (RandomForest/Logistic-or-Linear) "
              f"generalize better than deep boosting ensembles, which tend to overfit with few samples.",
              weight=2)

    # --- Rule 4: high categorical percentage -> CatBoost handles raw categoricals best
    if cat_pct > 40:
        boost(["CatBoost", "LightGBM"],
              f"High proportion of categorical features ({cat_pct:.0f}%) -> CatBoost has native "
              f"categorical handling and typically needs less encoding-related tuning.", weight=2)

    # --- Rule 5: high dimensionality -> tree ensembles handle irrelevant features better
    if n_features > 50:
        boost(["RandomForest", "ExtraTrees", "LightGBM"],
              f"High feature count ({n_features}) -> tree ensembles are naturally robust to "
              f"irrelevant/redundant features without manual selection.", weight=1)

    # --- Rule 6: high missingness -> boosting models with native NaN handling
    if missing_ratio > 0.05:
        boost(["LightGBM", "XGBoost", "CatBoost"],
              f"Notable missing data ({missing_ratio*100:.1f}%) -> gradient boosting libraries "
              f"handle missing values natively, reducing sensitivity to imputation choices.", weight=1)

    # --- Rule 7: high feature correlation -> penalize plain linear models
    if profile.get("high_correlation_pairs"):
        scores["LogisticRegression"] = scores.get("LogisticRegression", 0) - 1
        scores["LinearRegression"] = scores.get("LinearRegression", 0) - 1
        reasons.append(
            f"{len(profile['high_correlation_pairs'])} highly correlated feature pair(s) found -> "
            f"plain linear models are sensitive to multicollinearity; tree-based models are not."
        )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_score = ranked[0][1] if ranked else 0

    # Confidence: how much the top candidates actually stand out from the pack
    positive_signals = [s for _, s in ranked if s > 0]
    confidence = min(1.0, (top_score / 6.0)) if positive_signals else 0.0

    evaluate_all = confidence < 0.3 or len(positive_signals) < 2

    if evaluate_all:
        candidates = all_models
        reasons.append(
            "No dataset characteristic strongly favored a subset of algorithms "
            "(confidence too low) -> evaluating the full model pool instead of guessing."
        )
    else:
        chosen_names = [name for name, s in ranked[:max_candidates] if s > 0]
        # always keep at least one simple baseline in the mix for comparison
        baseline = "LogisticRegression" if problem_type == "classification" else "LinearRegression"
        if baseline not in chosen_names and baseline in all_models:
            chosen_names.append(baseline)
        candidates = {name: all_models[name] for name in chosen_names}

    return candidates, reasons, round(confidence, 2), evaluate_all
