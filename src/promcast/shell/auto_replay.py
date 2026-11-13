import argparse
import math
import os
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from promcast.core.config import Config, TargetConfig
from promcast.core.forecast import run_cycle
from promcast.core.forecaster import Forecaster
from promcast.core.recipes import candidate_query, scan_target
from promcast.core.replay import ReplaySpec
from promcast.core.series import Series, align, end_at_origin
from promcast.shell.config import load_config
from promcast.shell.errors import (
    ConfigMissingError,
    ReplayHistoryTooShortError,
    ReplayProbeUnavailableError,
)
from promcast.shell.scan_cli import discover_config
from promcast.shell.sources import ReplaySource
from promcast.shell.tsdb import RangeGrid, range_query


@dataclass(frozen=True)
class AutoReplay:
    """One target series ready for a deterministic replay."""

    target: TargetConfig
    source: ReplaySource
    labels: dict[str, str]


async def _auto_config(arguments: argparse.Namespace) -> Config:
    path = Path(arguments.config) if arguments.config else Path("promcast/promcast.yaml")
    if path.exists():
        config = load_config(path, environment=os.environ)
    elif arguments.config:
        raise ConfigMissingError(str(path))
    else:
        async with httpx.AsyncClient() as client:
            config, recipes = await discover_config(arguments, client)
        config = config.model_copy(
            update={"targets": [scan_target(r, 300, arguments.max_series) for r in recipes]}
        )
    return config


async def prepare_auto(
    arguments: argparse.Namespace, forecaster: Forecaster
) -> tuple[Config, list[AutoReplay], int]:
    """Read at most thirty days per selected series and measure inference throughput."""
    config = await _auto_config(arguments)
    ca_file = config.tsdb.tls.ca_file
    trust = ssl.create_default_context(cafile=str(ca_file.expanduser())) if ca_file else True
    jobs: list[AutoReplay] = []
    elapsed_s = 0.0
    needed = 0
    has_history = False
    required_s = min(
        (
            int(2 * (target.context + target.horizon).total_seconds() - target.step.total_seconds())
            for target in config.targets
        ),
        default=0,
    )
    available_s = 0
    async with httpx.AsyncClient(verify=trust) as client:
        for target in config.targets:
            step = int(target.step.total_seconds())
            end = int(time.time() // step) * step
            start = end - 30 * 86400
            matrix = await range_query(
                client,
                config.tsdb,
                candidate_query(target.query, target.direction, target.max_series, end),
                RangeGrid(start, end, step),
            )
            series, _ = align(matrix, step, end, (end - start) // step + 1)
            for item in series:
                observed = np.flatnonzero(~np.isnan(item.values))
                if not observed.size:
                    continue
                trimmed = end_at_origin(
                    Series(item.labels, step, item.end_ts, item.values[int(observed[0]) :])
                )
                if trimmed is None:
                    continue
                context = int(target.context / target.step)
                horizon = int(target.horizon / target.step)
                required = (2 * (context + horizon) - 1) * step
                available = (trimmed.values.size - 1) * step
                if required - available < required_s - available_s:
                    required_s, available_s = required, available
                if trimmed.values.size < 2 * (context + horizon):
                    continue
                has_history = True
                probe = Series(trimmed.labels, step, trimmed.end_ts, trimmed.values[-context:])
                started = time.perf_counter()
                forecasts, _ = run_cycle(target, [probe], forecaster, end)
                elapsed_s += time.perf_counter() - started
                if not forecasts:
                    continue
                cycle = max(1, int(config.cycle / target.step))
                spec = ReplaySpec(
                    target.direction,
                    context,
                    horizon,
                    step,
                    cycle,
                    int(target.season / target.step),
                    threshold=target.threshold,
                )
                needed += max(0, (trimmed.values.size - context - horizon) // cycle)
                source = ReplaySource(
                    trimmed.values, trimmed.end_ts - (trimmed.values.size - 1) * step, spec
                )
                jobs.append(AutoReplay(target, source, dict(item.labels)))
    if not jobs:
        if has_history:
            raise ReplayProbeUnavailableError()
        raise ReplayHistoryTooShortError(available_s, required_s)
    eta = math.ceil(needed * elapsed_s / len(jobs) / 60)
    sys.stderr.write(f"Replay ETA: {eta} min ({len(jobs)} series, measured inference throughput)\n")
    return config, jobs, eta
