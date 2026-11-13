from dataclasses import dataclass
from math import ceil

import numpy as np

from promcast.core.baselines import linear
from promcast.core.config import TargetConfig
from promcast.core.forecaster import Forecaster
from promcast.core.series import (
    FloatArray,
    Labels,
    Series,
    Skipped,
    apply_tail_rule,
    end_at_origin,
    is_monotone_counter,
)

_MIN_OBSERVATIONS = 64
_MAX_MISSING_FRACTION = 0.5
_QUANTILE_COUNT = 9


@dataclass(frozen=True)
class PreparationPolicy:
    """Observation minimum and whether metric semantics establish a gauge."""

    minimum_observations: int = _MIN_OBSERVATIONS
    known_gauge: bool = False


@dataclass(frozen=True)
class Forecast:
    """Quantiles at successive steps after the last observed timestamp."""

    target: str
    labels: Labels
    origin_ts: float
    step_s: int
    quantiles: FloatArray
    missing_fraction: float
    model: str
    naive_mae: float | None = None


def _trim_future(series: Series, now_ts: float) -> Series:
    timestamps = series.end_ts - np.arange(series.values.size - 1, -1, -1) * series.step_s
    keep = timestamps <= now_ts
    values = series.values[keep].copy()
    end_ts = float(timestamps[keep][-1]) if values.size else now_ts
    return Series(series.labels, series.step_s, end_ts, values)


def _prepare(
    source: Series, now_ts: float, window_steps: int, policy: PreparationPolicy
) -> tuple[Series, Series, float] | Skipped:
    series = end_at_origin(_trim_future(source, now_ts))
    if series is None or np.any(np.isinf(series.values)):
        return Skipped(source.labels, "missing")
    missing_fraction = float(np.mean(np.isnan(series.values)))
    if missing_fraction > _MAX_MISSING_FRACTION:
        return Skipped(series.labels, "missing")
    if is_monotone_counter(series.values, known_gauge=policy.known_gauge):
        return Skipped(series.labels, "counter")
    if (
        np.count_nonzero(~np.isnan(series.values)) < policy.minimum_observations
        or np.count_nonzero(~np.isnan(series.values[-window_steps:])) < 2
    ):
        return Skipped(series.labels, "too_short")
    tail = apply_tail_rule(series.values)
    if tail is None:
        return Skipped(series.labels, "tail_missing")
    filled = Series(series.labels, series.step_s, series.end_ts, tail[0])
    return series, filled, missing_fraction


def run_cycle(
    target: TargetConfig,
    series_list: list[Series],
    forecaster: Forecaster,
    now_ts: float,
    *,
    policy: PreparationPolicy | None = None,
) -> tuple[list[Forecast], list[Skipped]]:
    """Prepare eligible contexts, cap in input order and forecast without I/O."""
    policy = policy or PreparationPolicy(known_gauge=target.known_gauge)
    if not np.isfinite(now_ts):
        raise ValueError("now_ts must be finite")
    horizon_steps = int(target.horizon / target.step)
    window_steps = ceil(target.linear_window / target.step)
    prepared: list[Series] = []
    originals: list[Series] = []
    missing_fractions: list[float] = []
    skipped: list[Skipped] = []
    for source in series_list:
        candidate = _prepare(source, now_ts, window_steps, policy)
        if isinstance(candidate, Skipped):
            skipped.append(candidate)
        elif len(prepared) >= target.max_series:
            skipped.append(Skipped(source.labels, "capped"))
        else:
            original, filled, missing_fraction = candidate
            originals.append(original)
            prepared.append(filled)
            missing_fractions.append(missing_fraction)
    if not prepared:
        return [], skipped
    baseline = [linear(item.values, item.step_s, horizon_steps, window_steps) for item in originals]
    outputs = forecaster.forecast(prepared, horizon_steps) if target.method == "toto" else baseline
    if len(outputs) != len(prepared):
        raise ValueError("forecaster must return one output per input series")
    forecasts: list[Forecast] = []
    for original, series, missing, output, comparator in zip(
        originals, prepared, missing_fractions, outputs, baseline, strict=True
    ):
        if output.shape != (_QUANTILE_COUNT, horizon_steps):
            raise ValueError("forecaster output must have shape (9, horizon_steps)")
        if not np.all(np.isfinite(output)) or not np.all(np.isfinite(comparator)):
            skipped.append(Skipped(series.labels, "model_output"))
            continue
        models = [(forecaster.name, output), ("linear", comparator)]
        if target.method == "linear":
            models = [("linear", output)]
        differences = np.abs(np.diff(original.values))
        observed_differences = differences[np.isfinite(differences)]
        naive_mae = float(np.mean(observed_differences)) if observed_differences.size else None
        for model, quantiles in models:
            forecasts.append(
                Forecast(
                    target.name,
                    series.labels,
                    series.end_ts,
                    series.step_s,
                    np.maximum.accumulate(quantiles, axis=0),
                    missing,
                    model,
                    naive_mae,
                )
            )
    return forecasts, skipped
