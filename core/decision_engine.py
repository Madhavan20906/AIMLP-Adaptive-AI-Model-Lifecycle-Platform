"""
core/decision_engine.py

Mode 2: Adaptive Production Model Evolution.

    load current_model.pkl
        -> evaluate on latest_dataset.csv
        -> drift detection (data drift severity)
        -> health score
        -> DECISION:
             health Excellent/Good  -> retrain SAME algorithm, return updated_model
             health Average/Poor    -> evaluate candidate algorithms via the
                                        leaderboard; replace ONLY if the best
                                        candidate beats the current model's
                                        score by more than `switch_threshold`
                                        (avoids unnecessary model switching,
                                        exactly as the spec requires) --
                                        otherwise fall back to retraining the
                                        same algorithm anyway.

Every decision returns a plain-English explanation, because "the platform
should never only tell the user which model is best" applies here too:
users get the reasoning, not just a verdict.
"""

import numpy as np

from core.trainer import train_one, run_leaderboard, _overall_score
from core.health_score import compute_health_score
from core.drift_detector import data_drift_severity


def evaluate_current_model(model, X_latest, y_latest, problem_type):
    """Evaluates the currently-deployed model on the latest data, same
    metric set the leaderboard uses so comparisons are apples-to-apples.
    Also computes real mean prediction confidence from predict_proba when
    available, instead of the health score silently substituting the
    performance score in its place."""
    from core.trainer import _classification_metrics, _regression_metrics
    if problem_type == "classification":
        metrics, inference_ms = _classification_metrics(model, X_latest, y_latest)
    else:
        metrics, inference_ms = _regression_metrics(model, X_latest, y_latest)

    confidence = None
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_latest)
            # mean of the winning class's probability per prediction -- how
            # confident the model was in whichever class it actually picked
            confidence = float(np.mean(np.max(proba, axis=1)))
        except Exception:
            pass

    return metrics, inference_ms, confidence


def evolve_model(
    current_model,
    model_name: str,
    estimator_class,
    X_ref, y_ref,
    X_latest, y_latest,
    profile: dict,
    candidates: dict,
    switch_threshold: float = 3.0,
):
    """
    Runs the full Mode 2 decision flow.

    - current_model: the fitted model currently in production
    - model_name / estimator_class: identity of the current model's algorithm,
      needed to retrain "the same algorithm" on fresh data
    - X_ref, y_ref: reference window (older data) for drift comparison
    - X_latest, y_latest: the new data the current model is evaluated against
    - candidates: dict[name -> unfit estimator] from candidate_selector,
      used only if health is Average/Poor
    """
    problem_type = profile["problem_type"]

    current_metrics, inference_ms, prediction_confidence = evaluate_current_model(
        current_model, X_latest, y_latest, problem_type
    )
    current_score = _overall_score(
        {"metrics": current_metrics, "train_time_s": 0, "model_size_kb": 0}, problem_type
    )

    drift_info = data_drift_severity(np.asarray(X_ref), np.asarray(X_latest))

    health = compute_health_score(
        metrics=current_metrics,
        problem_type=problem_type,
        drift_severity=drift_info["severity"],
        prediction_confidence=prediction_confidence,
        latency_ms=inference_ms,
    )

    explanation = {
        "current_model": model_name,
        "current_metrics": current_metrics,
        "current_overall_score": current_score,
        "drift": drift_info,
        "health": health,
        "decision": None,
        "action": None,
        "reasoning": [],
        "chosen_model": None,
        "chosen_model_metrics": None,
        "chosen_overall_score": None,
    }

    if health["category"] in ("Excellent", "Good"):
        # Case 1: current model is healthy -> retrain SAME algorithm on fresh data
        retrained = estimator_class()
        result = train_one(model_name, retrained, X_ref, y_ref, X_latest, y_latest, problem_type)
        result["overall_score"] = _overall_score(result, problem_type) if result["status"] == "ok" else None

        explanation["decision"] = "retrain_same_algorithm"
        explanation["action"] = "updated_model.pkl"
        explanation["chosen_model"] = model_name
        explanation["chosen_model_metrics"] = result.get("metrics")
        explanation["chosen_overall_score"] = result.get("overall_score")
        explanation["reasoning"].append(
            f"Current model health is '{health['category']}' ({health['health_score']}/100) -> "
            f"still performing well, so we retrain the SAME algorithm ({model_name}) on the latest "
            f"data rather than risk an unnecessary model switch."
        )
        explanation["new_model_object"] = result.get("model")
        return explanation

    # Case 2: health is Average/Poor -> evaluate candidate algorithms
    leaderboard = run_leaderboard(candidates, X_ref, y_ref, X_latest, y_latest, problem_type)
    ok_candidates = [r for r in leaderboard if r["status"] == "ok"]

    explanation["decision_path"] = "candidate_evaluation"
    explanation["leaderboard"] = [
        {"name": r["name"], "overall_score": r["overall_score"], "metrics": r["metrics"]}
        for r in ok_candidates
    ]

    if not ok_candidates:
        # nothing trained successfully -- fall back to retraining the same algorithm
        retrained = estimator_class()
        result = train_one(model_name, retrained, X_ref, y_ref, X_latest, y_latest, problem_type)
        explanation["decision"] = "retrain_same_algorithm"
        explanation["action"] = "updated_model.pkl"
        explanation["chosen_model"] = model_name
        explanation["chosen_model_metrics"] = result.get("metrics")
        explanation["chosen_overall_score"] = result.get("overall_score") if result["status"] == "ok" else None
        explanation["new_model_object"] = result.get("model")
        explanation["reasoning"].append(
            "Health was Average/Poor and no candidate algorithm trained successfully -> "
            "falling back to retraining the current algorithm rather than shipping nothing."
        )
        return explanation

    best = ok_candidates[0]
    improvement = best["overall_score"] - current_score

    if improvement > switch_threshold:
        explanation["decision"] = "replace_model"
        explanation["action"] = "replacement_model.pkl"
        explanation["chosen_model"] = best["name"]
        explanation["chosen_model_metrics"] = best["metrics"]
        explanation["chosen_overall_score"] = best["overall_score"]
        explanation["new_model_object"] = best["model"]
        explanation["reasoning"].append(
            f"Current model health is '{health['category']}' ({health['health_score']}/100). "
            f"Best candidate '{best['name']}' scores {best['overall_score']} vs. current "
            f"{current_score} (+{improvement:.2f}), exceeding the {switch_threshold}-point switch "
            f"threshold -> replacing the model family."
        )
    else:
        # health poor, but no candidate meaningfully better -> retrain same algorithm anyway
        retrained = estimator_class()
        result = train_one(model_name, retrained, X_ref, y_ref, X_latest, y_latest, problem_type)
        explanation["decision"] = "retrain_same_algorithm"
        explanation["action"] = "updated_model.pkl"
        explanation["chosen_model"] = model_name
        explanation["chosen_model_metrics"] = result.get("metrics")
        explanation["chosen_overall_score"] = result.get("overall_score") if result["status"] == "ok" else None
        explanation["new_model_object"] = result.get("model")
        explanation["reasoning"].append(
            f"Current model health is '{health['category']}' ({health['health_score']}/100), but best "
            f"candidate '{best['name']}' only scores {best['overall_score']} vs. current {current_score} "
            f"(+{improvement:.2f}) -- below the {switch_threshold}-point switch threshold, so we retrain "
            f"the same algorithm instead of switching models for a marginal (or no) gain."
        )

    return explanation
