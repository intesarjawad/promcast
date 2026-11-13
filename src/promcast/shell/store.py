"""Atomic cycle snapshots with complete candidate and dropped-series counts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType

from promcast.core.config import TargetConfig
from promcast.core.forecast import Forecast
from promcast.core.series import Labels, Skipped


@dataclass(frozen=True)
class CycleCompletion:
    """Completion time and total cap drops, including candidates not fetched."""

    completed_ts: float
    dropped_count: int


@dataclass(frozen=True)
class TargetSnapshot:
    """One complete cycle, with retained success time and cumulative drops."""

    config: TargetConfig
    forecasts: tuple[Forecast, ...]
    skipped: tuple[Skipped, ...]
    last_values: Mapping[Labels, float]
    last_success_ts: float | None
    dropped_total: int


class Store:
    """Atomically replace each target's latest cycle for concurrent scrapes."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._targets: dict[str, TargetSnapshot] = {}

    def update(
        self,
        config: TargetConfig,
        forecasts: Sequence[Forecast],
        skipped: Sequence[Skipped],
        last_values: Mapping[Labels, float],
        completed_ts: float | CycleCompletion,
    ) -> None:
        """Publish a completed cycle; only forecasts advance last success.

        Last values must be the actuals at each forecast's origin, after trimming
        future and missing samples, rather than the final query grid slot.
        """
        if isinstance(completed_ts, CycleCompletion):
            if completed_ts.dropped_count < 0:
                raise ValueError("dropped_count must be nonnegative")
            cycle_drops = completed_ts.dropped_count
            timestamp = completed_ts.completed_ts
        else:
            cycle_drops = sum(item.reason == "capped" for item in skipped)
            timestamp = completed_ts
        if any(item.target != config.name for item in forecasts):
            raise ValueError("all forecasts must belong to the updated target")
        if any(item.labels not in last_values for item in forecasts):
            raise ValueError("last_values must include the actual at every forecast origin")
        copied = tuple(
            Forecast(
                item.target,
                item.labels,
                item.origin_ts,
                item.step_s,
                item.quantiles.copy(),
                item.missing_fraction,
                item.model,
                item.naive_mae,
            )
            for item in forecasts
        )
        for item in copied:
            item.quantiles.flags.writeable = False
        with self._lock:
            previous = self._targets.get(config.name)
            last_success = previous.last_success_ts if previous else None
            dropped = previous.dropped_total if previous else 0
            self._targets[config.name] = TargetSnapshot(
                config.model_copy(deep=True),
                copied,
                tuple(skipped),
                MappingProxyType(dict(last_values)),
                timestamp if copied else last_success,
                dropped + cycle_drops,
            )

    def snapshot(self) -> tuple[TargetSnapshot, ...]:
        """Read all targets from one store revision without holding a scrape lock."""
        with self._lock:
            return tuple(self._targets.values())
