"""Shared target cycles with candidate selection anchored at the integer range end.

Each cycle fetches candidate history and an instant count of the original query
at that same end to keep selection and dropped-series counts consistent.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from promcast.core.config import Config, TargetConfig
from promcast.core.forecast import Forecast, run_cycle
from promcast.core.forecaster import Forecaster
from promcast.core.recipes import candidate_query
from promcast.core.selfaudit import SelfAudit
from promcast.core.series import Labels, Series, Skipped, align, end_at_origin
from promcast.shell.exposition import AuditSample, Collector
from promcast.shell.store import CycleCompletion, Store
from promcast.shell.tsdb import RangeGrid, TsdbError, range_query, series_count

LOGGER = logging.getLogger(__name__)
_FAILED_CYCLES_UNHEALTHY = 3
_AUDIT_WINDOW_S = 24 * 3600
_AUDIT_LEADS = (("10m", 600), ("30m", 1800), ("1h", 3600))


@dataclass(frozen=True)
class CycleResources:
    """Shared HTTP transport, inference and publication boundaries."""

    client: httpx.AsyncClient
    forecaster: Forecaster
    store: Store
    collector: Collector


@dataclass(frozen=True)
class Clock:
    """Wall time for query origins and monotonic time for cycle scheduling."""

    wall: Callable[[], float] = time.time
    monotonic: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class _TargetResult:
    forecasts: list[Forecast]
    skipped: list[Skipped]
    last_values: dict[Labels, float]
    actuals: list[Series]


def _forecast_target(
    target: TargetConfig, matrix: dict[str, object], forecaster: Forecaster, end_ts: float
) -> _TargetResult:
    series, rejected = align(
        matrix, int(target.step.total_seconds()), end_ts, int(target.context / target.step)
    )
    forecasts, skipped = run_cycle(target, series, forecaster, end_ts)
    origins = [end_at_origin(item) for item in series]
    last_values = {item.labels: float(item.values[-1]) for item in origins if item is not None}
    return _TargetResult(forecasts, rejected + skipped, last_values, series)


class Scheduler:
    """Run serial target cycles, retaining published forecasts on transport failure."""

    def __init__(
        self, config: Config, resources: CycleResources, clock: Clock | None = None
    ) -> None:
        self.config = config
        self.resources = resources
        self._clock = clock or Clock()
        self._lock = asyncio.Lock()
        self._succeeded: set[str] = set()
        self._failures = dict.fromkeys((target.name for target in config.targets), 0)
        self._running = True
        self._audit_leads = {
            target.name: (
                *(
                    (name, seconds)
                    for name, seconds in _AUDIT_LEADS
                    if seconds <= target.horizon.total_seconds()
                    and seconds % int(target.step.total_seconds()) == 0
                ),
                ("end", int(target.horizon.total_seconds())),
            )
            for target in config.targets
        }
        self._audits = {
            name: SelfAudit([seconds for _, seconds in leads], _AUDIT_WINDOW_S)
            for name, leads in self._audit_leads.items()
        }

    @property
    def healthy(self) -> bool:
        """Require a successful forecast for every target and fewer than three failures."""
        return (
            self._running
            and bool(self._failures)
            and all(
                name in self._succeeded and count < _FAILED_CYCLES_UNHEALTHY
                for name, count in self._failures.items()
            )
        )

    async def _target(self, target: TargetConfig, now_ts: float) -> None:
        started = self._clock.monotonic()
        record: dict[str, object] = {"event": "target_cycle", "target": target.name}
        try:
            step_s = int(target.step.total_seconds())
            end_ts = int(now_ts // step_s * step_s)
            grid = RangeGrid(
                end_ts - (int(target.context / target.step) - 1) * step_s, end_ts, step_s
            )
            query = candidate_query(target.query, target.direction, target.max_series, end_ts)
            matrix = await range_query(self.resources.client, self.config.tsdb, query, grid)
            total = await series_count(
                self.resources.client, self.config.tsdb, target.query, end_ts
            )
            result = await asyncio.to_thread(
                _forecast_target, target, matrix, self.resources.forecaster, end_ts
            )
            audit = self._audits[target.name]
            audit.summaries(end_ts)
            for actual in result.actuals[: target.max_series]:
                audit.score_series(actual)
            for forecast in result.forecasts:
                audit.record(forecast)
            self.resources.store.update(
                target,
                result.forecasts,
                result.skipped,
                result.last_values,
                CycleCompletion(self._clock.wall(), max(0, total - target.max_series)),
            )
            success = bool(result.forecasts)
            record.update(
                status="ok" if success else "no_forecasts",
                forecasts=len(result.forecasts),
                skipped=[
                    {"labels": dict(item.labels), "reason": item.reason} for item in result.skipped
                ],
            )
        except ValidationError:
            success = False
            record.update(status="invalid_data", error="invalid numeric TSDB samples")
        except TsdbError as error:
            success = False
            record.update(
                status="tsdb_error",
                error=str(error),
                http_status=error.status,
                tsdb_url=str(self.config.tsdb.url),
            )
        if success:
            self._succeeded.add(target.name)
            self._failures[target.name] = 0
        else:
            self._failures[target.name] += 1
        record["duration_s"] = self._clock.monotonic() - started
        level = logging.INFO if success else logging.WARNING
        if record.get("skipped"):
            level = logging.WARNING
        LOGGER.log(level, json.dumps(record, sort_keys=True))

    async def cycle(self) -> float:
        """Complete one cycle without overlapping inference, returning elapsed seconds."""
        async with self._lock:
            started = self._clock.monotonic()
            now_ts = self._clock.wall()
            for target in self.config.targets:
                await self._target(target, now_ts)
            self.resources.collector.replace_audit(
                tuple(
                    AuditSample(
                        summary.target,
                        lead,
                        summary.method,
                        summary.mase,
                        summary.samples,
                        summary.coverage_ratio,
                    )
                    for name, audit in self._audits.items()
                    for summary in audit.summaries(now_ts)
                    for lead, seconds in self._audit_leads[name]
                    if seconds == summary.lead_s
                )
            )
            elapsed_s = self._clock.monotonic() - started
            self.resources.collector.observe_cycle(self.config.model.name, elapsed_s)
            if elapsed_s > self.config.cycle.total_seconds():
                LOGGER.warning(
                    json.dumps(
                        {
                            "event": "cycle_overrun",
                            "duration_s": elapsed_s,
                            "next": "promcast bench",
                        }
                    )
                )
            return elapsed_s

    async def run(self, stop: asyncio.Event) -> None:
        """Start immediately, wait only the remaining interval and finish on shutdown."""
        try:
            while not stop.is_set():
                elapsed_s = await self.cycle()
                delay_s = max(0, self.config.cycle.total_seconds() - elapsed_s)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay_s)
                except TimeoutError:
                    continue
        finally:
            self._running = False
