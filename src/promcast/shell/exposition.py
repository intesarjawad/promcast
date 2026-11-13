from collections.abc import Iterator
from dataclasses import dataclass
from threading import Lock

from prometheus_client.core import HistogramMetricFamily, Metric

from promcast.core.alerting import already_breached, quantile_row, time_to_breach_s
from promcast.core.forecast import Forecast
from promcast.shell.store import Store, TargetSnapshot

_GAUGES = {
    "forecast_origin_timestamp_seconds": "Last observed sample used by the forecast.",
    "forecast_value": "Forecast quantile at a fixed lead.",
    "breach_timestamp_seconds": "Unix time of the first forecast breach.",
    "last_success_timestamp_seconds": "End of the last cycle producing forecasts.",
    "series_missing_fraction": "Share of missing samples in context.",
    "series_skipped": "Series skipped in the latest cycle.",
    "selfaudit_coverage_ratio": "Rolling coverage of the q10 to q90 interval.",
    "selfaudit_mase": "Rolling mean absolute scaled error.",
    "selfaudit_samples": "Number of scored forecasts in the rolling window.",
    "build_info": "Build and model identity.",
}
_FIXED_LEADS = (("10m", 600), ("30m", 1800), ("1h", 3600))
# Prometheus client default duration buckets, in seconds.
_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    float("inf"),
)


@dataclass(frozen=True)
class AuditSample:
    """A precomputed rolling self-audit result supplied by the scoring shell."""

    target: str
    lead: str
    method: str
    mase: float | None
    samples: int
    coverage_ratio: float | None = None


def _series_labels(target: str, labels: tuple[tuple[str, str], ...]) -> dict[str, str]:
    reserved = {"target", "quantile", "lead", "method", "reason", "__name__"}
    source = dict(labels)
    result = {name: value for name, value in labels if name not in reserved}
    for name, value in labels:
        if name in reserved:
            exported = "exported_" + name
            while exported in source or exported in result:
                exported = "exported_" + exported
            result[exported] = value
    return {"target": target, **result}


def _add(metric: Metric, labels: dict[str, str], value: float) -> None:
    metric.add_sample(metric.name, labels, value)


def _forecast_metrics(
    metrics: dict[str, Metric], state: TargetSnapshot, forecast: Forecast
) -> None:
    labels = _series_labels(state.config.name, forecast.labels)
    is_primary = (forecast.model == "linear") == (state.config.method == "linear")
    if is_primary:
        _add(metrics["forecast_origin_timestamp_seconds"], labels, forecast.origin_ts)
        _add(metrics["series_missing_fraction"], labels, forecast.missing_fraction)
        horizon_steps = forecast.quantiles.shape[1]
        leads = [
            (name, seconds // forecast.step_s - 1)
            for name, seconds in _FIXED_LEADS
            if seconds % forecast.step_s == 0 and seconds <= horizon_steps * forecast.step_s
        ]
        leads.append(("end", horizon_steps - 1))
        for lead, index in leads:
            for row in range(9):
                _add(
                    metrics["forecast_value"],
                    {**labels, "quantile": f"{(row + 1) / 10:.1f}", "lead": lead},
                    float(forecast.quantiles[row, index]),
                )
    cfg = state.config
    if already_breached(state.last_values[forecast.labels], cfg.threshold, cfg.direction):
        return
    seconds = time_to_breach_s(
        quantile_row(forecast.quantiles, cfg.quantile, cfg.direction),
        forecast.step_s,
        cfg.threshold,
        cfg.direction,
    )
    if seconds is not None:
        _add(
            metrics["breach_timestamp_seconds"],
            {**labels, "method": "linear" if forecast.model == "linear" else "toto"},
            forecast.origin_ts + seconds,
        )


class Collector:
    """Read current target snapshots on each scrape, without persistent series gauges."""

    def __init__(self, store: Store, version: str, model: str, model_sha: str) -> None:
        self._store = store
        self._build = {"version": version, "model": model, "model_sha": model_sha}
        self._lock = Lock()
        self._durations: dict[str, tuple[tuple[int, ...], float]] = {}
        self._audit: tuple[AuditSample, ...] = ()

    def observe_cycle(self, model: str, duration_s: float) -> None:
        """Accumulate a completed model cycle's nonnegative elapsed seconds."""
        if not 0 <= duration_s < float("inf"):
            raise ValueError("duration_s must be finite and nonnegative")
        with self._lock:
            counts, total = self._durations.get(model, ((0,) * len(_DURATION_BUCKETS), 0.0))
            self._durations[model] = (
                tuple(
                    count + (duration_s <= bound)
                    for count, bound in zip(counts, _DURATION_BUCKETS, strict=True)
                ),
                total + duration_s,
            )

    def replace_audit(self, samples: tuple[AuditSample, ...]) -> None:
        """Replace rolling results so expired audit series disappear on the next scrape."""
        with self._lock:
            self._audit = samples

    def collect(self) -> Iterator[Metric]:
        """Emit only the contracted families, including cumulative counter and histogram."""
        metrics = {
            name: Metric(f"promcast_{name}", help_text, "gauge")
            for name, help_text in _GAUGES.items()
        }
        dropped = Metric(
            "promcast_series_dropped", "Series dropped by the cardinality cap.", "counter"
        )
        for state in self._store.snapshot():
            target_labels = {"target": state.config.name}
            if state.last_success_ts is not None:
                _add(
                    metrics["last_success_timestamp_seconds"], target_labels, state.last_success_ts
                )
            dropped.add_sample("promcast_series_dropped_total", target_labels, state.dropped_total)
            for forecast in state.forecasts:
                _forecast_metrics(metrics, state, forecast)
            for skipped in state.skipped:
                _add(
                    metrics["series_skipped"],
                    {**_series_labels(state.config.name, skipped.labels), "reason": skipped.reason},
                    1,
                )
        _add(metrics["build_info"], self._build, 1)
        with self._lock:
            audit = self._audit
            durations = dict(self._durations)
        sample_counts: dict[tuple[str, str], int] = {}
        for sample in audit:
            labels = {"target": sample.target, "lead": sample.lead, "method": sample.method}
            if sample.mase is not None:
                _add(metrics["selfaudit_mase"], labels, sample.mase)
            if sample.method == "toto" and sample.coverage_ratio is not None:
                _add(metrics["selfaudit_coverage_ratio"], labels, sample.coverage_ratio)
            key = (sample.target, sample.lead)
            sample_counts[key] = max(sample_counts.get(key, 0), sample.samples)
        for (target, lead), count in sample_counts.items():
            _add(metrics["selfaudit_samples"], {"target": target, "lead": lead}, count)
        yield from metrics.values()
        yield dropped
        histogram = HistogramMetricFamily(
            "promcast_cycle_duration_seconds", "Model cycle duration in seconds.", labels=["model"]
        )
        for model, (counts, total) in durations.items():
            buckets = [
                ("+Inf" if bound == float("inf") else str(bound), count)
                for bound, count in zip(_DURATION_BUCKETS, counts, strict=True)
            ]
            histogram.add_metric([model], buckets, total)
        yield histogram
