"""
core/drift_detector.py

Same DDM + KS-test approach validated in the standalone drift_pipeline
project, generalized here to output a Low/Medium/High severity (rather
than a binary flag) since that's what health_score.py and the decision
engine need.
"""

import numpy as np
from scipy.stats import ks_2samp


class PerformanceDriftDetector:
    """DDM: streaming error-rate drift detector (see drift_pipeline/drift/detector.py
    for the original derivation and the warm-up bug fix this already includes)."""

    def __init__(self, warning_level=3.0, drift_level=6.0, min_instances=100):
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_instances = min_instances
        self.reset()

    def reset(self):
        self.n = 0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.p = 0.0
        self.s = 0.0

    def update(self, is_error: int) -> str:
        self.n += 1
        self.p += (is_error - self.p) / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / self.n) if self.n > 0 else 0.0

        if self.n >= self.min_instances and self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s

        if self.n < self.min_instances:
            return "in_control"
        if self.p + self.s > self.p_min + self.drift_level * self.s_min:
            return "drift"
        elif self.p + self.s > self.p_min + self.warning_level * self.s_min:
            return "warning"
        return "in_control"


def data_drift_severity(X_ref: np.ndarray, X_current: np.ndarray, alpha=0.01) -> dict:
    """
    Runs a KS test per feature, comparing reference vs. current distributions.
    Returns a severity level based on what fraction of features drifted
    significantly, plus the per-feature detail for the dashboard.
    """
    n_features = X_ref.shape[1]
    drifted = []
    p_values = []

    for f in range(n_features):
        stat, p = ks_2samp(X_ref[:, f], X_current[:, f])
        p_values.append(p)
        if p < alpha and stat > 0.1:  # require both statistical significance AND effect size
            drifted.append({"feature_index": f, "ks_stat": round(float(stat), 4), "p_value": round(float(p), 5)})

    fraction_drifted = len(drifted) / max(n_features, 1)

    if fraction_drifted >= 0.4:
        severity = "High"
    elif fraction_drifted >= 0.15:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "severity": severity,
        "fraction_features_drifted": round(fraction_drifted, 3),
        "drifted_features": drifted,
        "mean_p_value": round(float(np.mean(p_values)), 5),
    }
