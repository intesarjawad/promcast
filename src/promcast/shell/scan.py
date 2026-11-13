import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from promcast.core.alerting import already_breached, quantile_row, time_to_breach_s
from promcast.core.config import Config, TargetConfig
from promcast.core.diagnostics import safe_urls
from promcast.core.forecast import Forecast, PreparationPolicy, run_cycle
from promcast.core.forecaster import Forecaster
from promcast.core.recipes import Recipe
from promcast.core.series import end_at_origin
from promcast.shell.discover import History, read_history


class ScanRow(BaseModel):
    """One forecast and its comparator in the public scan document."""

    model_config = ConfigDict(extra="forbid")

    target: str
    labels: dict[str, str]
    value: float
    unit: str
    threshold: float
    promcast_breach_ts: float | None
    linear_breach_ts: float | None
    missing_fraction: float
    short_history_min: float | None
    horizon_s: int
    show_labels: list[str] = Field(default=[], exclude=True)
    forecast_values: list[float] = Field(default=[], exclude=True)


class ScanSkipped(BaseModel):
    """One unforecast series and an actionable warmup estimate when known."""

    model_config = ConfigDict(extra="forbid")

    target: str
    labels: dict[str, str]
    reason: str
    warmup_minutes: int | None = Field(default=None, exclude=True)


class ScanDocument(BaseModel):
    """Versioned JSON surface shared with the countdown board."""

    model_config = ConfigDict(extra="forbid")

    url: str
    model: str
    origin_ts: int
    rows: list[ScanRow]
    skipped: list[ScanSkipped]

    @field_validator("url")
    @classmethod
    def _safe_url(cls, value: str) -> str:
        return safe_urls(value)


@dataclass(frozen=True)
class ScanResult:
    """Scan output and operational metadata kept outside its JSON contract."""

    document: ScanDocument
    targets: list[TargetConfig]
    dropped: int
    total: int
    forecasts_per_s: float
    replay_forecasts_needed: int = 0


@dataclass(frozen=True)
class ScanOptions:
    """One fixed range end and candidate limit for an entire scan."""

    end_ts: int
    max_series: int = 50
    progress: Callable[[int, int], None] | None = None


def _breach(forecast: Forecast, target: TargetConfig, value: float) -> float | None:
    if already_breached(value, target.threshold, target.direction):
        return forecast.origin_ts
    row = quantile_row(forecast.quantiles, target.quantile, target.direction)
    lead = time_to_breach_s(row, forecast.step_s, target.threshold, target.direction)
    return forecast.origin_ts + lead if lead is not None else None


def _forecast_rows(
    recipe: Recipe,
    history: History,
    forecaster: Forecaster,
    *,
    end_ts: int,
    advance: Callable[[], None] = lambda: None,
) -> tuple[list[ScanRow], list[ScanSkipped]]:
    target, series = history.target, history.series
    intervals, counts = history.intervals, history.counts
    rows: list[ScanRow] = []
    skipped: list[ScanSkipped] = []
    for item in series:
        advance()
        origin = end_at_origin(item)
        observed = int(np.count_nonzero(~np.isnan(item.values)))
        real = counts.get(item.labels, 0)
        interval = intervals.get(item.labels)
        if origin is None or interval is None or min(real, observed) < 32:
            minutes = ceil(max(0, 32 - real) * interval / 60) if interval else None
            skipped.append(
                ScanSkipped(
                    target=target.name,
                    labels=dict(item.labels),
                    reason="too_short",
                    warmup_minutes=minutes,
                )
            )
            continue
        short = min(real, observed) < 64
        horizon_s = int(target.horizon.total_seconds())
        if short:
            horizon_s = min(horizon_s, observed * item.step_s)
        limited = target.model_copy(update={"horizon": target.step * (horizon_s // item.step_s)})
        forecasts, rejected = run_cycle(
            limited,
            [item],
            forecaster,
            end_ts,
            policy=PreparationPolicy(32, target.known_gauge),
        )
        skipped.extend(
            ScanSkipped(target=target.name, labels=dict(s.labels), reason=s.reason)
            for s in rejected
        )
        if not forecasts:
            continue
        learned, linear = forecasts
        value = float(origin.values[-1])
        rows.append(
            ScanRow(
                target=target.name,
                labels=dict(item.labels),
                value=value,
                unit=recipe.unit,
                threshold=target.threshold,
                promcast_breach_ts=_breach(learned, limited, value),
                linear_breach_ts=_breach(linear, limited, value),
                missing_fraction=learned.missing_fraction,
                short_history_min=observed * item.step_s / 60 if short else None,
                horizon_s=horizon_s,
                show_labels=recipe.show,
                forecast_values=quantile_row(
                    learned.quantiles, target.quantile, target.direction
                ).tolist(),
            )
        )
    return rows, skipped


async def scan(
    client: httpx.AsyncClient,
    config: Config,
    recipes: list[Recipe],
    forecaster: Forecaster,
    options: ScanOptions,
) -> ScanResult:
    """Discover history and use the shared forecast cycle with real-sample gating."""
    end_ts, max_series = options.end_ts, options.max_series
    rows: list[ScanRow] = []
    skipped: list[ScanSkipped] = []
    targets: list[TargetConfig] = []
    dropped = total = forecast_count = 0
    inference_s = 0.0
    replay_needed = 0
    histories = [
        (recipe, await read_history(client, config, recipe, end_ts, max_series))
        for recipe in recipes
    ]
    series_total = sum(len(history.series) for _, history in histories)
    done = 0

    def advance() -> None:
        nonlocal done
        done += 1
        if options.progress is not None:
            options.progress(done, series_total)

    for recipe, history in histories:
        targets.append(history.target)
        total += history.total
        dropped += max(0, history.total - max_series)
        started = time.perf_counter()
        found, rejected = _forecast_rows(
            recipe, history, forecaster, end_ts=end_ts, advance=advance
        )
        inference_s += time.perf_counter() - started
        forecast_count += len(found)
        replay_steps = 30 * 86400 // int(history.target.step.total_seconds())
        context_steps = int(history.target.context / history.target.step)
        horizon_steps = int(history.target.horizon / history.target.step)
        cycle_steps = max(1, int(config.cycle / history.target.step))
        replay_needed += (
            len(found) * max(0, replay_steps - context_steps - horizon_steps) // cycle_steps
        )
        rows.extend(found)
        skipped.extend(rejected)
    rows.sort(
        key=lambda row: (
            row.promcast_breach_ts if row.promcast_breach_ts is not None else float("inf"),
            row.target,
            sorted(row.labels.items()),
        )
    )
    document = ScanDocument(
        url=str(config.tsdb.url).rstrip("/"),
        model=forecaster.name,
        origin_ts=end_ts,
        rows=rows,
        skipped=skipped,
    )
    return ScanResult(
        document,
        targets,
        dropped,
        total,
        forecast_count / inference_s if inference_s else 0,
        replay_needed,
    )


def redact(document: ScanDocument) -> ScanDocument:
    """Use stable label digests independent of the discovered population; hide the TSDB."""

    def labels(values: dict[str, str]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for key, value in values.items():
            prefix = "host" if key == "instance" else "mount" if key == "mountpoint" else key
            digest = hashlib.sha256(f"{key}={value}".encode()).hexdigest()[:12]
            aliases[key] = f"{prefix}-{digest}"
        return aliases

    return document.model_copy(
        update={
            "url": "<redacted>",
            "rows": [
                row.model_copy(update={"labels": labels(row.labels)}) for row in document.rows
            ],
            "skipped": [
                row.model_copy(update={"labels": labels(row.labels)}) for row in document.skipped
            ],
        }
    )
