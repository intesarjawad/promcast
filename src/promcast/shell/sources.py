import csv
import ssl
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

import httpx
import numpy as np
from pydantic import BaseModel, TypeAdapter

from promcast.core.alerting import Direction
from promcast.core.config import Config, TargetConfig
from promcast.core.replay import Incident, ReplaySpec
from promcast.core.scenarios import SCENARIOS
from promcast.core.series import FloatArray, align, is_monotone_counter
from promcast.shell.errors import ReplayArgumentsError, ReplayCounterRefusedError
from promcast.shell.tsdb import RangeGrid, range_query

_SYNTHETIC_STEPS = 14 * 24 * 60
_BOOM_STEPS = 4096
_CORPUS_STEP_S = 300


@dataclass(frozen=True)
class ReplaySource:
    """A single regular timeline, its protocol and source caveats."""

    values: FloatArray
    start_ts: float
    spec: ReplaySpec
    caveats: tuple[str, ...] = ()


def timestamp(text: str) -> float:
    """Parse ISO timestamps, treating timezone-free corpus dates as UTC."""
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return value.replace(tzinfo=UTC).timestamp() if value.tzinfo is None else value.timestamp()


def synthetic(scenario: str) -> ReplaySource:
    """Build seeded E1 signals; the leak crosses zero two hours after day eleven."""
    rng = np.random.default_rng(0)
    minutes = np.arange(_SYNTHETIC_STEPS, dtype=np.float64)
    elapsed = np.maximum(minutes - 11 * 1440, 0) / 120
    values = 100 - 100 * elapsed**2
    if scenario == "linear_leak":
        values = 100 - 100 * elapsed
    elif scenario == "daily_seasonal_peak":
        values = 50 + 40 * np.sin(minutes * 2 * np.pi / 1440)
    elif scenario == "step_change":
        values = np.where(minutes < 11 * 1440, 100.0, -10.0)
    elif scenario == "saturation_curve":
        values = 100 - 120 * (1 - np.exp(-elapsed))
    elif scenario == "noisy_flat":
        values = 100 + rng.normal(0, 1, minutes.size)
    elif scenario == "gappy_accelerating_leak":
        values[rng.random(minutes.size) < 0.2] = np.nan
    elif scenario == "counter":
        values = np.cumsum(rng.poisson(5, minutes.size)).astype(np.float64)
    elif scenario not in SCENARIOS:
        raise ValueError(f"unknown synthetic scenario: {scenario}")
    if is_monotone_counter(values):
        raise ReplayCounterRefusedError(scenario)
    direction, threshold = SCENARIOS[scenario]
    return ReplaySource(
        values,
        0.0,
        ReplaySpec(direction, 1440, 120, 60, 5, 1440, threshold=threshold),
        (f"Synthetic scenario: {scenario}. Seed: 0.",),
    )


def _five_minute_csv(path: Path, corpus: str) -> tuple[FloatArray, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"timestamp", "value"}.issubset(reader.fieldnames):
            raise ValueError(f"{corpus} CSV requires timestamp and value columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{corpus} CSV {path} has no rows")
    samples = [[timestamp(row["timestamp"]), float(row["value"])] for row in rows]
    first = min(time for time, _ in samples)
    end_ts = max(time for time, _ in samples)
    steps = int((end_ts - first) // _CORPUS_STEP_S) + 1
    # The same alignment a live query_range result gets, so missed samples become gaps.
    series, skipped = align(
        {"resultType": "matrix", "result": [{"metric": {}, "values": samples}]},
        _CORPUS_STEP_S,
        end_ts,
        steps,
    )
    if skipped:
        raise ValueError(f"{corpus} CSV {path} holds an infinite value")
    return series[0].values, end_ts - (steps - 1) * _CORPUS_STEP_S


def nab(path: Path) -> ReplaySource:
    """Load NAB's five-minute CSV onto its grid; missed samples stay missing, as in a live query."""
    values, start_ts = _five_minute_csv(path, "NAB")
    return ReplaySource(
        values,
        start_ts,
        ReplaySpec("above", 576, 24, 300, 1, 288),
        ("NAB labels have documented flaws; many labelled anomalies are not forecastable.",),
    )


def homelab(path: Path) -> ReplaySource:
    """Load a recorded five-minute node_exporter series; free space and memory alert low."""
    values, start_ts = _five_minute_csv(path, "Homelab")
    name = path.name.lower()
    direction: Direction = "below" if "filesystem" in name or "memory" in name else "above"
    return ReplaySource(values, start_ts, ReplaySpec(direction, 1152, 72, 300, 1, 288))


class _BoomRow(BaseModel):
    start: datetime
    item_id: str
    freq: str
    target: list[list[float]] | list[float]


class _Taxonomy(BaseModel):
    type: list[str]


class _ArrowTable(Protocol):
    def to_pylist(self) -> list[dict[str, object]]: ...


class _ArrowReader(Protocol):
    def read_all(self) -> _ArrowTable: ...


@runtime_checkable
class _ArrowIPC(Protocol):
    def open_stream(self, source: BinaryIO) -> _ArrowReader: ...
    def open_file(self, source: BinaryIO) -> _ArrowReader: ...


def boom(folder: Path, target: str) -> ReplaySource:
    """Read one selected BOOM gauge variate, bounded to the last 4,096 samples."""
    taxonomy = TypeAdapter(dict[str, _Taxonomy]).validate_json(
        (folder / "dataset_taxonomy.json").read_text()
    )
    entry = taxonomy.get(target)
    if entry is None or entry.type != ["gauge"]:
        raise ValueError("BOOM target must have type [gauge] in dataset_taxonomy.json")
    ipc: object = import_module("pyarrow.ipc")
    if not isinstance(ipc, _ArrowIPC):
        raise TypeError("pyarrow.ipc must provide stream and file readers")
    own = sorted((folder / target).glob("*.arrow"))
    for path in own or sorted(folder.glob("*.arrow")):
        with path.open("rb") as stream:
            # Arrow files have a magic prefix; stream containers do not.
            is_file = stream.read(6) == b"ARROW1"
            stream.seek(0)
            reader = ipc.open_file(stream) if is_file else ipc.open_stream(stream)
            for raw in reader.read_all().to_pylist():
                row = _BoomRow.model_validate(raw)
                if own or row.item_id == target:
                    return _boom_source(row)
    raise ValueError(f"BOOM target not found: {target}")


def _boom_source(row: _BoomRow) -> ReplaySource:
    aliases = {
        "T": 60,
        "1T": 60,
        "5T": 300,
        "10T": 600,
        "30T": 1800,
        "10S": 10,
        "S": 1,
        "H": 3600,
        "1H": 3600,
        "D": 86400,
    }
    if row.freq not in aliases:
        raise ValueError(f"unknown BOOM frequency alias: {row.freq}")
    if not row.target:
        raise ValueError("BOOM target has no variates")
    # A single-variate series stores its target as one flat list.
    variates = np.atleast_2d(np.asarray(row.target, dtype=np.float64))
    variate = int(np.random.default_rng(0).integers(len(variates)))
    full = variates[variate]
    values = np.array(full[-_BOOM_STEPS:], dtype=np.float64)
    step = aliases[row.freq]
    season = 86400 // step
    return ReplaySource(
        values,
        timestamp(row.start.isoformat()) + max(0, len(full) - _BOOM_STEPS) * step,
        ReplaySpec("above", 1024, 64, step, 16, season if season <= 2048 else 0),
        ("BOOM is in-domain and possibly contaminated by model pretraining.",),
    )


def labelled_incidents(path: Path, key: str, start_ts: float) -> list[Incident]:
    """Use combined_labels timestamps as point incidents, never anomaly windows."""
    document = TypeAdapter(list[str] | dict[str, list[str]]).validate_json(path.read_text())
    if isinstance(document, dict):
        matches = [
            value
            for name, value in document.items()
            if name == key or key.endswith(name) or name.endswith(key)
        ]
        if len(matches) != 1:
            raise ValueError("incident file must identify exactly one selected series")
        document = matches[0]
    labels = [timestamp(item) - start_ts for item in document]
    return [Incident(value, value) for value in sorted(set(labels))]


async def tsdb(config: Config, target: TargetConfig, start: str, end: str) -> ReplaySource:
    """Fetch exactly one target series with configured authentication and TLS trust."""
    start_ts, end_ts = timestamp(start), timestamp(end)
    step = int(target.step.total_seconds())
    if end_ts <= start_ts or (end_ts - start_ts) % step:
        raise ValueError("TSDB range must increase and contain a whole number of steps")
    ca_file = config.tsdb.tls.ca_file
    trust = ssl.create_default_context(cafile=str(ca_file.expanduser())) if ca_file else True
    async with httpx.AsyncClient(verify=trust) as client:
        matrix = await range_query(
            client, config.tsdb, target.query, RangeGrid(start_ts, end_ts, step)
        )
    series, skipped = align(matrix, step, end_ts, int((end_ts - start_ts) / step) + 1)
    if len(series) != 1 or skipped:
        raise ValueError("replay requires exactly one gauge series; narrow the target query")
    if is_monotone_counter(series[0].values, known_gauge=target.known_gauge):
        raise ReplayArgumentsError(
            f"TSDB target {target.name} looks like a counter; "
            "set known_gauge: true in its configuration only for a verified gauge"
        )
    return ReplaySource(
        series[0].values,
        start_ts,
        ReplaySpec(
            target.direction,
            int(target.context.total_seconds()) // step,
            int(target.horizon.total_seconds()) // step,
            step,
            max(1, int(config.cycle.total_seconds()) // step),
            int(target.season.total_seconds()) // step,
            threshold=target.threshold,
        ),
    )
