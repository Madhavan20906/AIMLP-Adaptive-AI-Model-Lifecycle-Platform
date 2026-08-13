"""
core/health_score.py

A model's "health" is not just its accuracy. This composite score is what
Mode 2's decision engine actually acts on: whether to retrain the same
algorithm, or evaluate replacements.

Weights (must sum to 1.0):
    performance   0.50  -- accuracy/f1/roc_auc (or r2 for regression)
    drift         0.20  -- how much the input distribution/error rate has shifted
    confidence    0.15  -- mean prediction confidence (classification only)
    efficiency    0.15  -- latency + memory + model size, normalized

Every number here is explainable -- the breakdown is returned alongside
the score specifically so the dashboard/report can show WHY a model
scored what it did, not just the number.
"""

DRIFT_PENALTY = {"Low": 0.0, "Medium": 0.4, "High": 1.0}


def _performance_component(metrics: dict, problem_type: str) -> float:
    if problem_type == "classification":
        parts = [metrics.get("accuracy", 0), metrics.get("f1", 0)]
        if metrics.get("roc_auc") is not None:
            parts.append(metrics["roc_auc"])
        return sum(parts) / len(parts)
    else:
        return max(0.0, min(1.0, metrics.get("r2", 0)))


def _efficiency_component(latency_ms: float, memory_mb: float, model_size_kb: float) -> float:
    """
    Normalizes efficiency into a 0-1 "good" score using soft caps -- these
    thresholds are reasonable defaults for a lightweight production model,
    not hard requirements. A model well within them scores ~1.0; one far
    beyond them approaches 0.
    """
    latency_score = max(0.0, 1 - (latency_ms / 100))       # 100ms/sample = fully penalized
    memory_score = max(0.0, 1 - (memory_mb / 500))          # 500MB = fully penalized
    size_score = max(0.0, 1 - (model_size_kb / 50000))      # 50MB model = fully penalized
    return (latency_score + memory_score + size_score) / 3


def compute_health_score(
    metrics: dict,
    problem_type: str,
    drift_severity: str,
    prediction_confidence: float = None,
    latency_ms: float = 5.0,
    memory_mb: float = 50.0,
    model_size_kb: float = 500.0,
) -> dict:
    performance = _performance_component(metrics, problem_type)
    drift_component = 1 - DRIFT_PENALTY.get(drift_severity, 0.5)
    confidence_component = prediction_confidence if prediction_confidence is not None else performance
    efficiency = _efficiency_component(latency_ms, memory_mb, model_size_kb)

    weighted = (
        performance * 0.50
        + drift_component * 0.20
        + confidence_component * 0.15
        + efficiency * 0.15
    )
    score_100 = round(weighted * 100, 1)

    if score_100 >= 85:
        category = "Excellent"
    elif score_100 >= 70:
        category = "Good"
    elif score_100 >= 50:
        category = "Average"
    else:
        category = "Poor"

    # Drift severity can override an otherwise-strong category. Found by
    # testing, not assumed upfront: with performance weighted at 50%, a
    # model with "High" drift (e.g. 45% of features shifted) could still
    # land in "Good" if its current-window accuracy hadn't degraded yet --
    # which defeats the point of detecting drift as an EARLY warning,
    # before performance craters. Capping the category here means Mode 2's
    # decision engine can no longer be lulled into "retrain same algorithm"
    # by a performance metric that simply hasn't caught up to the drift yet.
    if drift_severity == "High" and category in ("Excellent", "Good"):
        category = "Average"
    elif drift_severity == "Medium" and category == "Excellent":
        category = "Good"

    return {
        "health_score": score_100,
        "category": category,
        "breakdown": {
            "performance": round(performance * 100, 1),
            "drift": round(drift_component * 100, 1),
            "confidence": round(confidence_component * 100, 1),
            "efficiency": round(efficiency * 100, 1),
        },
        "weights": {"performance": 0.50, "drift": 0.20, "confidence": 0.15, "efficiency": 0.15},
        "drift_severity": drift_severity,
    }
