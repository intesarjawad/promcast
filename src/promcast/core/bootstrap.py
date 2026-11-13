from collections.abc import Callable
from typing import Literal

import numpy as np

from promcast.core.series import FloatArray


def bootstrap_ci(
    per_series: list[FloatArray],
    stat: Callable[[FloatArray], float],
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a 95% percentile interval by resampling whole series clusters.

    Empty clusters remain in the sampling population. Replicates containing no
    incidents are undefined and omitted; an entirely empty input is an error.
    """
    if n <= 0 or not per_series or not any(item.size for item in per_series):
        raise ValueError("bootstrap requires incidents, series and positive resample count")
    if any(item.ndim != 1 or not np.isfinite(item).all() for item in per_series):
        raise ValueError("bootstrap clusters must be finite one-dimensional arrays")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n):
        indices = rng.integers(0, len(per_series), size=len(per_series))
        sample = np.concatenate([per_series[int(index)] for index in indices])
        if sample.size:
            estimate = stat(sample)
            if not np.isfinite(estimate):
                raise ValueError("bootstrap statistic must be finite")
            estimates.append(estimate)
    if not estimates:
        raise ValueError("no bootstrap replicate contained incidents")
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def paired_warning_difference(early: FloatArray, linear: FloatArray) -> FloatArray:
    """Subtract incident-aligned warning times; missed incidents must be zero."""
    if early.shape != linear.shape or early.ndim != 1:
        raise ValueError("paired warning times must have identical one-dimensional shapes")
    if not np.isfinite(early).all() or not np.isfinite(linear).all():
        raise ValueError("paired warning times must be finite, with zero for misses")
    if np.any(early < 0) or np.any(linear < 0):
        raise ValueError("warning times must be nonnegative")
    return early - linear


def decision_rule(
    interval: tuple[float, float], false_alerts_per_week: float, budget: float
) -> Literal["beats", "ties", "loses"]:
    """Apply the pre-registered paired warning-time and test false-alert rule."""
    lower, upper = interval
    if not np.isfinite([lower, upper, false_alerts_per_week, budget]).all():
        raise ValueError("decision inputs must be finite")
    if lower > upper or min(false_alerts_per_week, budget) < 0:
        raise ValueError("decision requires an ordered interval and nonnegative rates")
    if lower <= 0 <= upper:
        return "ties"
    if lower > 0 and false_alerts_per_week <= 1.5 * budget:
        return "beats"
    return "loses"
