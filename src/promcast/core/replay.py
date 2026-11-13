from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil
from typing import TYPE_CHECKING

import numpy as np

from promcast.core import baselines
from promcast.core.alerting import Direction, already_breached, quantile_row, time_to_breach_s
from promcast.core.operating import MethodScore, OperatingPoint, choose_operating_point
from promcast.core.series import BoolArray, FloatArray, Series, apply_tail_rule

if TYPE_CHECKING:
    from promcast.core.forecaster import Forecaster

_SECONDS_PER_WEEK = 7 * 24 * 3600


@dataclass(frozen=True)
class ReplaySpec:
    """Sampling, alert persistence and calibration protocol shared by all methods."""

    direction: Direction
    context_steps: int
    horizon_steps: int
    step_s: int
    cycle_steps: int
    season_steps: int
    for_cycles: int = 2
    calibration_fraction: float = 0.3
    min_incident_steps: int = 3
    budget_per_week: float = 1.0
    threshold: float | None = None
    auto_percentile: float = 99.0

    def __post_init__(self) -> None:
        if (
            min(
                self.context_steps,
                self.horizon_steps,
                self.step_s,
                self.cycle_steps,
                self.for_cycles,
                self.min_incident_steps,
            )
            <= 0
            or self.season_steps < 0
        ):
            raise ValueError("replay step counts must be positive; season may be zero")
        if not 0 < self.calibration_fraction < 1 or not 0 < self.auto_percentile < 100:
            raise ValueError("calibration fraction and percentile must be inside their ranges")
        if not np.isfinite(self.budget_per_week) or self.budget_per_week < 0:
            raise ValueError("budget must be finite and nonnegative")
        if self.threshold is not None and not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")


@dataclass(frozen=True)
class Incident:
    """Inclusive first and last observed breach timestamps, after merging."""

    start_ts: float
    end_ts: float


@dataclass(frozen=True)
class Episode:
    """Persistent warning with end at the first inactive evaluation."""

    condition_start_ts: float
    firing_ts: float
    end_ts: float


@dataclass(frozen=True)
class ForecastTrace:
    """One cached prediction, reused across quantile and warning-time sweeps."""

    origin: int
    quantiles: FloatArray
    fallback: str | None = None


@dataclass(frozen=True)
class MethodResult:
    """Calibration sweep and test score at its locked operating point."""

    point: OperatingPoint
    calibration: MethodScore
    test: MethodScore
    sweep: dict[OperatingPoint, MethodScore] = field(
        default_factory=dict[OperatingPoint, MethodScore]
    )
    fallbacks: tuple[str, ...] = ()
    test_sweep: dict[OperatingPoint, MethodScore] = field(
        default_factory=dict[OperatingPoint, MethodScore]
    )
    labelled_test: MethodScore | None = None


@dataclass(frozen=True)
class ReplayResult:
    """Pure replay output; callers own serialization and filesystem access."""

    spec: ReplaySpec
    threshold: float
    split: int
    methods: dict[str, MethodResult]

    @property
    def calibration_winner(self) -> str:
        """Rank locked method points using only their calibration scores."""
        return choose_operating_point(
            {method.point: method.calibration for method in self.methods.values()},
            self.spec.budget_per_week,
        ).method


def split_index(n: int, spec: ReplaySpec) -> int:
    """Split the timeline before forecasting, keeping both partitions nonempty."""
    split = int(n * spec.calibration_fraction)
    if not 0 < split < n:
        raise ValueError("timeline must have nonempty calibration and test splits")
    return split


def auto_threshold(values: FloatArray, split: int, spec: ReplaySpec) -> float:
    """Use observed calibration samples only for the threshold percentile."""
    if not 0 < split < values.size:
        raise ValueError("split must be inside the timeline")
    if spec.threshold is not None:
        return spec.threshold
    observed = values[:split][np.isfinite(values[:split])]
    if not observed.size:
        raise ValueError("auto threshold requires observed calibration samples")
    percentile = spec.auto_percentile if spec.direction == "above" else 100 - spec.auto_percentile
    return float(np.percentile(observed, percentile))


def detect_incidents(
    ts: FloatArray, values: FloatArray, threshold: float, spec: ReplaySpec
) -> list[Incident]:
    """Find sustained breaches and merge gaps strictly shorter than one horizon."""
    if ts.shape != values.shape or ts.ndim != 1 or np.any(np.diff(ts) != spec.step_s):
        raise ValueError("incident timestamps must match values on the replay grid")
    breached = values >= threshold if spec.direction == "above" else values <= threshold
    transitions = np.diff(np.concatenate(([False], breached, [False])).astype(int))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    incidents: list[Incident] = []
    for start, end in zip(starts, ends, strict=True):
        if end - start < spec.min_incident_steps:
            continue
        incident = Incident(float(ts[start]), float(ts[end - 1]))
        if (
            incidents
            and incident.start_ts - incidents[-1].end_ts < spec.horizon_steps * spec.step_s
        ):
            incident = Incident(incidents.pop().start_ts, incident.end_ts)
        incidents.append(incident)
    return incidents


def rolling_origins(n: int, spec: ReplaySpec) -> list[int]:
    """Start after a complete context and evaluate each effective cycle once."""
    return list(range(spec.context_steps - 1, n, spec.cycle_steps))


def episodes(
    origin_ts: FloatArray, condition: BoolArray, in_breach: BoolArray, spec: ReplaySpec
) -> list[Episode]:
    """Fire on the required consecutive origin; reset on gaps or breached actuals."""
    if (
        origin_ts.ndim != 1
        or origin_ts.shape != condition.shape
        or condition.shape != in_breach.shape
    ):
        raise ValueError("episode arrays must have identical one-dimensional shapes")
    if not np.isfinite(origin_ts).all() or np.any(np.diff(origin_ts) <= 0):
        raise ValueError("episode origins must be finite and strictly increasing")
    cycle_s = spec.cycle_steps * spec.step_s
    result: list[Episode] = []
    start: float | None = None
    firing: float | None = None
    count = 0
    for i, timestamp in enumerate(origin_ts):
        gap = i > 0 and timestamp - origin_ts[i - 1] != cycle_s
        if gap or not condition[i] or in_breach[i]:
            if start is not None and firing is not None:
                end = float(origin_ts[i - 1] + cycle_s) if gap else float(timestamp)
                result.append(Episode(start, firing, end))
            start, firing, count = None, None, 0
        if condition[i] and not in_breach[i]:
            start = float(timestamp) if start is None else start
            count += 1
            if count == spec.for_cycles:
                firing = float(timestamp)
    if start is not None and firing is not None:
        result.append(Episode(start, firing, float(origin_ts[-1] + cycle_s)))
    return result


def score(
    eps: list[Episode], incidents: list[Incident], weeks: float, spec: ReplaySpec
) -> MethodScore:
    """Match by firing time, preserving zero warning times for every missed incident."""
    if not np.isfinite(weeks) or weeks <= 0:
        raise ValueError("scoring exposure must be positive and finite")
    warnings = [0.0] * len(incidents)
    late: set[int] = set()
    matched = 0
    resets: list[float] = []
    horizon_s = spec.horizon_steps * spec.step_s
    for episode in eps:
        for i, incident in enumerate(incidents):
            if incident.start_ts - horizon_s <= episode.firing_ts <= incident.end_ts:
                matched += 1
                resets.append(episode.end_ts - incident.end_ts)
                if episode.firing_ts < incident.start_ts:
                    warnings[i] = max(warnings[i], incident.start_ts - episode.firing_ts)
                else:
                    late.add(i)
                break
    leads = [lead for lead in warnings if lead > 0]
    return MethodScore(
        len(incidents),
        len(leads),
        len(late),
        len(eps),
        matched,
        weeks,
        leads,
        resets,
        warnings,
        series_warning_times_s=[warnings],
    )


def forecast_at(
    values: FloatArray, origin: int, spec: ReplaySpec, model: Forecaster
) -> FloatArray | None:
    """Give a model only a copied context ending at origin, with the shared tail rule."""
    if origin < spec.context_steps - 1 or origin >= values.size:
        raise ValueError("origin must have a full context and lie inside the timeline")
    context = values[origin - spec.context_steps + 1 : origin + 1].copy()
    if not np.isfinite(context[-1]) or np.isinf(context).any():
        return None
    tail = apply_tail_rule(context)
    if tail is None:
        return None
    outputs = model.forecast(
        [Series((), spec.step_s, float(origin * spec.step_s), tail[0])], spec.horizon_steps
    )
    if len(outputs) != 1 or outputs[0].shape != (9, spec.horizon_steps):
        raise ValueError("replay model must return one (9, horizon_steps) forecast")
    output = outputs[0]
    return output if np.isfinite(output).all() else None


def _trace(
    values: FloatArray,
    spec: ReplaySpec,
    point: OperatingPoint,
    model: Forecaster | None,
    calibration_sd: float,
) -> list[ForecastTrace]:
    traces: list[ForecastTrace] = []
    for origin in rolling_origins(values.size, spec):
        history = values[: origin + 1]
        if not np.isfinite(history[-1]):
            continue
        fallback = None
        if point.method == "linear":
            window_steps = max(2, ceil(point.window_s / spec.step_s))
            if np.count_nonzero(np.isfinite(history[-window_steps:])) < 2:
                continue
            output = baselines.linear(history, spec.step_s, spec.horizon_steps, window_steps)
        elif point.method == "static-margin":
            output = baselines.static_margin(
                history, spec.horizon_steps, point.margin_k * calibration_sd, spec.direction
            )
        elif point.method == "seasonal-naive":
            season = max(1, spec.season_steps)
            output = baselines.seasonal_naive(history, spec.step_s, spec.horizon_steps, season)
            if not spec.season_steps or history.size < season:
                fallback = "last_value"
        elif point.method == "ets":
            season = spec.season_steps or (history.size + 1)
            output = baselines.ets(history, spec.step_s, spec.horizon_steps, season)
            if not spec.season_steps or history.size < 2 * season:
                fallback = "nonseasonal_ets"
        elif model is not None:
            output = forecast_at(values, origin, spec, model)
        else:
            raise ValueError(f"missing forecaster for {point.method}")
        if output is not None and np.isfinite(output).all():
            traces.append(ForecastTrace(origin, output, fallback))
    return traces


def _quality(
    traces: list[ForecastTrace], values: FloatArray, spec: ReplaySpec, stop: int
) -> tuple[float | None, list[float | None]]:
    covered = total = 0
    losses = np.zeros(spec.horizon_steps)
    denominators = np.zeros(spec.horizon_steps)
    counts = np.zeros(spec.horizon_steps, dtype=int)
    levels: FloatArray = np.arange(1, 10, dtype=np.float64) / 10
    for trace in traces:
        size = min(spec.horizon_steps, stop - trace.origin - 1)
        truth = values[trace.origin + 1 : trace.origin + 1 + size]
        for lead in np.flatnonzero(np.isfinite(truth)):
            actual = float(truth[lead])
            prediction: FloatArray = trace.quantiles[:, lead]
            covered += 1 if prediction[0] <= actual <= prediction[8] else 0
            total += 1
            error = actual - prediction
            losses[lead] += float(2 * np.mean(np.maximum(levels * error, (levels - 1) * error)))
            denominators[lead] += abs(actual)
            counts[lead] += 1
    wql = [
        float(loss / denominator) if count and denominator else None
        for loss, denominator, count in zip(losses, denominators, counts, strict=True)
    ]
    return covered / total if total else None, wql


def _split_quality(
    traces: list[ForecastTrace], values: FloatArray, spec: ReplaySpec, split: int
) -> tuple[tuple[float | None, list[float | None]], tuple[float | None, list[float | None]]]:
    """Reuse forecast quality across points that only change alert thresholds."""
    return (
        _quality([trace for trace in traces if trace.origin < split], values, spec, split),
        _quality([trace for trace in traces if trace.origin >= split], values, spec, values.size),
    )


@dataclass(frozen=True)
class _Evaluation:
    start: int
    stop: int
    labels: list[Incident] | None = None


def _evaluate(
    values: FloatArray,
    spec: ReplaySpec,
    point: OperatingPoint,
    traces: list[ForecastTrace],
    evaluation: _Evaluation,
) -> MethodScore:
    start, stop = evaluation.start, evaluation.stop
    labels = evaluation.labels
    selected = [trace for trace in traces if start <= trace.origin < stop]
    times = np.array([trace.origin * spec.step_s for trace in selected], dtype=np.float64)
    condition: list[bool] = []
    in_breach: list[bool] = []
    threshold = spec.threshold
    if threshold is None:
        raise ValueError("evaluation requires a calibration-fixed threshold")
    for trace in selected:
        breach = time_to_breach_s(
            quantile_row(trace.quantiles, point.quantile, spec.direction),
            spec.step_s,
            threshold,
            spec.direction,
        )
        condition.append(breach is not None and breach < point.warn_s)
        in_breach.append(already_breached(float(values[trace.origin]), threshold, spec.direction))
    eps = episodes(
        times, np.array(condition, dtype=np.bool_), np.array(in_breach, dtype=np.bool_), spec
    )
    ts = np.arange(start, stop, dtype=np.float64) * spec.step_s
    incidents = (
        detect_incidents(ts, values[start:stop], threshold, spec)
        if labels is None
        else [
            incident
            for incident in labels
            if start * spec.step_s <= incident.start_ts < stop * spec.step_s
        ]
    )
    return score(eps, incidents, (stop - start) * spec.step_s / _SECONDS_PER_WEEK, spec)


@dataclass(frozen=True)
class PointScores:
    """One series' calibration and test scores at a single candidate operating point."""

    point: OperatingPoint
    calibration: MethodScore
    test: MethodScore
    labelled_test: MethodScore | None = None


@dataclass(frozen=True)
class _Calibrated:
    spec: ReplaySpec
    threshold: float
    split: int
    cache: dict[tuple[str, int, float], list[ForecastTrace]]
    quality: dict[
        tuple[str, int, float],
        tuple[tuple[float | None, list[float | None]], tuple[float | None, list[float | None]]],
    ]
    sweeps: dict[str, dict[OperatingPoint, MethodScore]]

    def key(self, point: OperatingPoint) -> tuple[str, int, float]:
        return (point.method, point.window_s, point.margin_k)

    def test(self, values: FloatArray, point: OperatingPoint) -> MethodScore:
        coverage, wql = self.quality[self.key(point)][1]
        return replace(
            _evaluate(
                values,
                self.spec,
                point,
                self.cache[self.key(point)],
                _Evaluation(self.split, values.size),
            ),
            coverage_80=coverage,
            wql_by_lead=wql,
        )

    def labelled(
        self,
        values: FloatArray,
        point: OperatingPoint,
        incidents: list[Incident],
        test: MethodScore,
    ) -> MethodScore:
        return replace(
            _evaluate(
                values,
                self.spec,
                point,
                self.cache[self.key(point)],
                _Evaluation(self.split, values.size, incidents),
            ),
            coverage_80=test.coverage_80,
            wql_by_lead=test.wql_by_lead,
        )


def _calibrate(
    values: FloatArray,
    spec: ReplaySpec,
    points: list[OperatingPoint],
    models: dict[str, Forecaster] | None,
    labelled_incidents: list[Incident] | None,
) -> _Calibrated:
    if values.ndim != 1 or np.isinf(values).any() or not points:
        raise ValueError("replay requires a one-dimensional series without infinities and points")
    if labelled_incidents is not None and any(
        not np.isfinite(incident.start_ts)
        or not np.isfinite(incident.end_ts)
        or incident.end_ts < incident.start_ts
        for incident in labelled_incidents
    ):
        raise ValueError("labelled incidents must have finite, ordered bounds")
    split = split_index(values.size, spec)
    threshold = auto_threshold(values, split, spec)
    fixed = replace(spec, threshold=threshold)
    observed = values[:split][np.isfinite(values[:split])]
    if not observed.size:
        raise ValueError("replay requires observed calibration samples")
    calibration_sd = float(np.std(observed))
    prepared = _Calibrated(fixed, threshold, split, {}, {}, {})
    for point in sorted(set(points)):
        if not 0 < point.warn_s <= spec.horizon_steps * spec.step_s:
            raise ValueError("warning time must be positive and at most the horizon")
        key = prepared.key(point)
        if key not in prepared.cache:
            prepared.cache[key] = _trace(
                values, fixed, point, (models or {}).get(point.method), calibration_sd
            )
            prepared.quality[key] = _split_quality(prepared.cache[key], values, fixed, split)
        calibration_quality = prepared.quality[key][0]
        prepared.sweeps.setdefault(point.method, {})[point] = replace(
            _evaluate(values, fixed, point, prepared.cache[key], _Evaluation(0, split)),
            coverage_80=calibration_quality[0],
            wql_by_lead=calibration_quality[1],
        )
    return prepared


def replay_sweep(
    values: FloatArray,
    spec: ReplaySpec,
    points: list[OperatingPoint],
    models: dict[str, Forecaster] | None = None,
    *,
    labelled_incidents: list[Incident] | None = None,
) -> dict[str, list[PointScores]]:
    """Score every candidate point on both splits without choosing one.

    A corpus run pools these per-series scores before selecting, so no series picks its own.
    """
    prepared = _calibrate(values, spec, points, models, labelled_incidents)
    result: dict[str, list[PointScores]] = {}
    for name, calibration in sorted(prepared.sweeps.items()):
        for point, calibrated in calibration.items():
            test = prepared.test(values, point)
            labelled = (
                None
                if labelled_incidents is None
                else prepared.labelled(values, point, labelled_incidents, test)
            )
            result.setdefault(name, []).append(PointScores(point, calibrated, test, labelled))
    return result


def replay(
    values: FloatArray,
    spec: ReplaySpec,
    points: list[OperatingPoint],
    models: dict[str, Forecaster] | None = None,
    *,
    labelled_incidents: list[Incident] | None = None,
) -> ReplayResult:
    """Cache predictions once, sweep calibration, then evaluate each locked test point."""
    prepared = _calibrate(values, spec, points, models, labelled_incidents)
    methods: dict[str, MethodResult] = {}
    for name, calibration in sorted(prepared.sweeps.items()):
        point = choose_operating_point(calibration, spec.budget_per_week)
        test_sweep = {candidate: prepared.test(values, candidate) for candidate in calibration}
        test = test_sweep[point]
        labelled_test = (
            prepared.labelled(values, point, labelled_incidents, test)
            if labelled_incidents is not None
            else None
        )
        fallbacks = tuple(
            sorted(
                {
                    trace.fallback
                    for trace in prepared.cache[prepared.key(point)]
                    if trace.fallback is not None
                }
            )
        )
        methods[name] = MethodResult(
            point, calibration[point], test, calibration, fallbacks, test_sweep, labelled_test
        )
    return ReplayResult(spec, prepared.threshold, prepared.split, methods)
