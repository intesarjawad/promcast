from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite

from promcast.core.forecast import Forecast
from promcast.core.series import Labels, Series


@dataclass(frozen=True)
class AuditPoint:
    """One matured lead, scored once against an observed sample."""

    target: str
    lead_s: int
    method: str
    timestamp: float
    covered: bool | None
    scaled_error: float | None


@dataclass(frozen=True)
class AuditSummary:
    """Rolling results pooled across series within a target and method."""

    target: str
    lead_s: int
    method: str
    samples: int
    coverage_ratio: float | None
    mase: float | None


@dataclass
class _Record:
    target: str
    origin_ts: float
    step_s: int
    method: str
    naive_mae: float | None
    predictions: dict[int, tuple[float, float, float]]
    scored: set[int] = field(default_factory=set[int])


class SelfAudit:
    """Score compact forecast leads without I/O or a hidden clock.

    Due times define the rolling window so late observations cannot extend it.
    Zero-scale contexts have undefined MASE and still contribute coverage.
    Call summaries each cycle to expire history even when actuals are missing.
    """

    def __init__(self, leads_s: Sequence[int], window_s: float) -> None:
        if not leads_s or any(lead <= 0 for lead in leads_s):
            raise ValueError("leads_s must contain positive seconds")
        if not isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s must be finite and positive")
        self._leads = tuple(sorted(set(leads_s)))
        self._window_s = window_s
        self._records: dict[Labels, dict[tuple[str, str, float], _Record]] = {}
        self._points: list[AuditPoint] = []

    def record(self, forecast: Forecast) -> None:
        """Capture only the requested quantiles; repeated origins keep their first result."""
        predictions = {
            lead: (
                float(forecast.quantiles[0, lead // forecast.step_s - 1]),
                float(forecast.quantiles[4, lead // forecast.step_s - 1]),
                float(forecast.quantiles[8, lead // forecast.step_s - 1]),
            )
            for lead in self._leads
            if lead % forecast.step_s == 0 and lead <= forecast.quantiles.shape[1] * forecast.step_s
        }
        method = "linear" if forecast.model == "linear" else "toto"
        records = self._records.setdefault(forecast.labels, {})
        records.setdefault(
            (forecast.target, method, forecast.origin_ts),
            _Record(
                forecast.target,
                forecast.origin_ts,
                forecast.step_s,
                method,
                forecast.naive_mae,
                predictions,
            ),
        )

    def score(self, actual_ts: float, actual_value: float, labels: Labels) -> list[AuditPoint]:
        """Match inclusive half-step tolerances and ignore missing or repeated actuals."""
        if not isfinite(actual_ts) or not isfinite(actual_value):
            return []
        points: list[AuditPoint] = []
        for record in self._records.get(labels, {}).values():
            for lead, (lower, median, upper) in record.predictions.items():
                due = record.origin_ts + lead
                if lead in record.scored or abs(actual_ts - due) > record.step_s / 2:
                    continue
                error = abs(actual_value - median)
                scale = record.naive_mae
                scaled = error / scale if scale is not None and scale > 0 else None
                point = AuditPoint(
                    record.target,
                    lead,
                    record.method,
                    due,
                    lower <= actual_value <= upper if record.method == "toto" else None,
                    scaled,
                )
                points.append(point)
                record.scored.add(lead)
        self._points.extend(points)
        return points

    def score_series(self, series: Series) -> None:
        """Match retained deadlines to the nearest slot in a newly fetched history."""
        deadlines = {
            record.origin_ts + lead
            for record in self._records.get(series.labels, {}).values()
            for lead in record.predictions
            if lead not in record.scored
        }
        start = series.end_ts - (series.values.size - 1) * series.step_s
        for due in sorted(deadlines):
            index = round((due - start) / series.step_s)
            if 0 <= index < series.values.size:
                self.score(
                    start + index * series.step_s, float(series.values[index]), series.labels
                )

    def summaries(self, now_ts: float) -> tuple[AuditSummary, ...]:
        """Expire state by the supplied time and aggregate the rolling window."""
        if not isfinite(now_ts):
            raise ValueError("now_ts must be finite")
        cutoff = now_ts - self._window_s
        for labels in tuple(self._records):
            self._records[labels] = {
                key: record
                for key, record in self._records[labels].items()
                if record.origin_ts >= cutoff - max(self._leads)
            }
            if not self._records[labels]:
                del self._records[labels]
        self._points = [point for point in self._points if point.timestamp >= cutoff]
        grouped: dict[tuple[str, int, str], list[AuditPoint]] = {}
        for point in self._points:
            if point.timestamp <= now_ts:
                grouped.setdefault((point.target, point.lead_s, point.method), []).append(point)
        results: list[AuditSummary] = []
        for (target, lead, method), points in sorted(grouped.items()):
            hits = [point.covered for point in points if point.covered is not None]
            errors = [point.scaled_error for point in points if point.scaled_error is not None]
            results.append(
                AuditSummary(
                    target,
                    lead,
                    method,
                    len(points),
                    sum(hits) / len(hits) if hits else None,
                    sum(errors) / len(errors) if errors else None,
                )
            )
        return tuple(results)
