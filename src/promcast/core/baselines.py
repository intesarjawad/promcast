import augurs
import numpy as np

from promcast.core.alerting import Direction
from promcast.core.series import FloatArray

_QUANTILE_COUNT = 9
_ETS_SEASONS = 3
# ADR 0015: MSTL decomposition requires at least two complete periods.
_MSTL_MIN_SEASONS = 2
_ETS_INTERVAL_LEVEL = 0.8


def _observed_indices(values: FloatArray) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    if values.ndim != 1 or np.any(np.isinf(values)):
        raise ValueError("values must be a one-dimensional array without infinities")
    observed = np.flatnonzero(~np.isnan(values))
    if not observed.size:
        raise ValueError("values must contain an observed value")
    return observed


def _validate_steps(step_s: int, horizon_steps: int, period_steps: int) -> None:
    if step_s <= 0 or horizon_steps <= 0 or period_steps <= 0:
        raise ValueError("step_s, horizon_steps and window or season steps must be positive")


def _identical_rows(row: FloatArray) -> FloatArray:
    return np.tile(row, (_QUANTILE_COUNT, 1))


def static_margin(
    values: FloatArray, horizon_steps: int, margin: float, direction: Direction
) -> FloatArray:
    """Shift the last observation toward the threshold for replay's static comparator."""
    if horizon_steps <= 0 or not np.isfinite(margin) or margin < 0:
        raise ValueError("horizon_steps must be positive and margin finite and nonnegative")
    last = values[_observed_indices(values)[-1]]
    shifted = last + margin if direction == "above" else last - margin
    return np.full((_QUANTILE_COUNT, horizon_steps), shifted, dtype=np.float64)


def linear(values: FloatArray, step_s: int, horizon_steps: int, window_steps: int) -> FloatArray:
    """Fit observed slots in the trailing window, keeping elapsed time across gaps."""
    _validate_steps(step_s, horizon_steps, window_steps)
    window = values[-window_steps:]
    observed = _observed_indices(window)
    if observed.size < 2:
        raise ValueError("linear requires at least two observed points in its window")
    times = (observed.astype(np.float64) - (window.size - 1)) * step_s
    samples = window[observed]
    mean_time = float(np.mean(times))
    mean_value = float(np.mean(samples))
    centered = times - mean_time
    slope = float(np.sum(centered * (samples - mean_value)) / np.sum(centered**2))
    leads = np.arange(1, horizon_steps + 1, dtype=np.float64) * step_s
    return _identical_rows(mean_value + slope * (leads - mean_time))


def seasonal_naive(
    history: FloatArray, step_s: int, horizon_steps: int, season_steps: int
) -> FloatArray:
    """Repeat the last season; callers record fallback=last_value for short history."""
    _validate_steps(step_s, horizon_steps, season_steps)
    observed = _observed_indices(history)
    if history.size < season_steps:
        row = np.full(horizon_steps, history[observed[-1]], dtype=np.float64)
    else:
        row = history[-season_steps:][np.arange(horizon_steps) % season_steps]
    return _identical_rows(row)


def ets(values: FloatArray, step_s: int, horizon_steps: int, season_steps: int) -> FloatArray:
    """Expand ETS's 80% band to nine rows; use AutoETS below two full seasons.

    Callers identify the fallback from the history span after leading missing slots.
    """
    _validate_steps(step_s, horizon_steps, season_steps)
    observed = _observed_indices(values)
    # Leading missing slots are outside the observed history, not a seasonal cycle.
    window = values[int(observed[0]) :][-(_ETS_SEASONS * season_steps) :]
    slots = np.arange(window.size)
    present = ~np.isnan(window)
    filled = np.interp(slots, slots[present], window[present])
    model = (
        augurs.MSTL.ets([season_steps])
        if filled.size >= _MSTL_MIN_SEASONS * season_steps
        else augurs.AutoETS(1, "ZZN")
    )
    model.fit(filled)
    forecast = model.predict(horizon_steps, level=_ETS_INTERVAL_LEVEL)
    point, lower, upper = forecast.point(), forecast.lower(), forecast.upper()
    if lower is None or upper is None:
        raise ValueError("ETS did not return the requested prediction intervals")
    weights = np.linspace(0, 1, 5)[:, None]
    return np.concatenate(
        [lower + weights * (point - lower), (point + weights * (upper - point))[1:]], axis=0
    )
