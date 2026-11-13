from typing import Literal

import numpy as np

from promcast.core.series import FloatArray

Direction = Literal["above", "below"]
_QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9)
_MEDIAN_INDEX = 4


def quantile_row(forecast: FloatArray, quantile: float, direction: Direction) -> FloatArray:
    """Select the upper risk quantile for above, or its complement for below."""
    if quantile not in _QUANTILES:
        raise ValueError("quantile must be one of 0.5, 0.6, 0.7, 0.8, 0.9")
    offset = _QUANTILES.index(quantile)
    index = _MEDIAN_INDEX + offset if direction == "above" else _MEDIAN_INDEX - offset
    return forecast[index]


def time_to_breach_s(
    row: FloatArray, step_s: int, threshold: float, direction: Direction
) -> float | None:
    """Return seconds to the first inclusive crossing, starting one step after origin."""
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    crossings = np.flatnonzero(row >= threshold if direction == "above" else row <= threshold)
    return float((int(crossings[0]) + 1) * step_s) if crossings.size else None


def already_breached(last_value: float, threshold: float, direction: Direction) -> bool:
    """Include equality when checking whether the actual has breached at origin."""
    return last_value >= threshold if direction == "above" else last_value <= threshold
