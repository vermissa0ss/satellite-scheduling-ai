import numpy as np


class HeuristicPriorityModel:
    """Lightweight baseline priority scorer used to test the MIS plumbing."""

    def score(self, component, cand_mask):
        dwell = np.maximum(component.dwell, 1.0)
        base = component.weights / dwell
        deg = component.degree.astype(np.float64, copy=False)
        max_deg = float(deg.max()) if deg.size else 0.0
        if max_deg > 0.0:
            base = base + 0.1 * deg / max_deg
        scores = np.full(component.n, -1e12, dtype=np.float64)
        remaining = cand_mask
        while remaining:
            bit = remaining & -remaining
            v = (bit.bit_length() - 1)
            scores[v] = base[v]
            remaining &= remaining - 1
        return scores


class DegreeAwarePriorityModel:
    """
    Priority model for MIS that prefers heavy, low-degree vertices.

    score(v) ≈ weight(v) / dwell(v) - alpha * normalized_degree(v)
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)

    def score(self, component, cand_mask):
        w = component.weights
        dwell = np.maximum(component.dwell, 1.0)
        deg = component.degree.astype(np.float64, copy=False)

        base = w / dwell
        max_deg = float(deg.max()) if deg.size else 1.0
        norm_deg = deg / max_deg
        full_scores = base - self.alpha * norm_deg

        scores = np.full(component.n, -1e12, dtype=np.float64)
        remaining = cand_mask
        while remaining:
            bit = remaining & -remaining
            v = bit.bit_length() - 1
            scores[v] = full_scores[v]
            remaining &= remaining - 1
        return scores
