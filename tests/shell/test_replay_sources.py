import asyncio
import json
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

import numpy as np
import pytest
import respx
from httpx import Request, Response
from numpy.testing import assert_array_equal

from promcast.core.replay import Incident
from promcast.core.scenarios import REFUSED_SCENARIOS
from promcast.shell.config import load_config
from promcast.shell.errors import ReplayArgumentsError, ReplayCounterRefusedError
from promcast.shell.sources import (
    boom,
    homelab,
    labelled_incidents,
    nab,
    synthetic,
    timestamp,
    tsdb,
)


@pytest.mark.parametrize(
    "scenario",
    [
        "linear_leak",
        "accelerating_leak",
        "daily_seasonal_peak",
        "step_change",
        "saturation_curve",
        "noisy_flat",
        "gappy_accelerating_leak",
    ],
)
def test_synthetic_is_seeded_and_uses_e1_protocol(scenario: str):
    first, second = synthetic(scenario), synthetic(scenario)
    assert_array_equal(first.values, second.values)
    assert first.values.size == 14 * 1440
    assert first.spec.context_steps == first.spec.season_steps == 1440
    assert first.spec.horizon_steps == 120
    assert first.spec.step_s == 60
    assert first.spec.cycle_steps == 5
    if scenario == "accelerating_leak":
        assert np.flatnonzero(first.values <= 0)[0] == 11 * 1440 + 120
    if scenario in ("daily_seasonal_peak", "noisy_flat"):
        assert first.spec.threshold is not None
        sign = 1 if first.spec.direction == "above" else -1
        assert np.all(sign * first.values < sign * first.spec.threshold)
    if scenario == "gappy_accelerating_leak":
        assert 0.19 < np.mean(np.isnan(first.values)) < 0.21


def test_seasonal_peak_approaches_its_threshold_without_reaching_it():
    source = synthetic("daily_seasonal_peak")
    assert source.spec.direction == "above"
    assert source.spec.threshold is not None
    # The generator peaks at 50 + 40 = 90: 10 below the threshold, an eighth of its daily swing.
    assert source.values.max() == pytest.approx(90)
    assert 0 < source.spec.threshold - source.values.max() <= 10


@pytest.mark.parametrize("scenario", REFUSED_SCENARIOS)
def test_refused_scenarios_are_counters(scenario: str):
    with pytest.raises(ReplayCounterRefusedError) as refused:
        synthetic(scenario)
    assert refused.value.what == (
        f"synthetic scenario {scenario} is a counter, not a gauge; replay forecasts gauges only"
    )
    assert refused.value.next_step == (
        "promcast replay --source synthetic --scenario accelerating_leak"
    )


def test_unknown_synthetic_name_is_refused():
    with pytest.raises(ValueError, match="unknown synthetic"):
        synthetic("typo")


def test_nab_preserves_native_timeline_and_utc(tmp_path: Path):
    path = tmp_path / "metric.csv"
    path.write_text("timestamp,value\n2014-01-01 00:00:00,5\n2014-01-01 00:05:00,4\n")
    source = nab(path)
    assert_array_equal(source.values, [5, 4])
    assert source.start_ts == timestamp("2014-01-01T00:00:00Z")
    assert source.spec.step_s == 300
    assert source.spec.context_steps == 576
    assert "documented flaws" in source.caveats[0]
    assert homelab(path).caveats == ()


def test_corpus_gaps_follow_live_alignment(tmp_path: Path):
    path = tmp_path / "metric.csv"
    path.write_text(
        "timestamp,value\n"
        "2014-01-01 00:00:00,5\n"
        "2014-01-01 00:07:00,9\n"
        "2014-01-01 00:10:00,3\n"
        "2014-01-01 00:10:00,2\n"
        "2014-01-01 00:15:00,1\n"
    )
    for load in (nab, homelab):
        source = load(path)
        assert_array_equal(source.values, [5, np.nan, 2, 1])
        assert source.start_ts == timestamp("2014-01-01T00:00:00Z")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("timestamp,value\n", "has no rows"),
        ("timestamp,value\n2014-01-01 00:00:00,inf\n", "infinite value"),
        ("time,value\n2014-01-01 00:00:00,1\n", "timestamp and value columns"),
    ],
)
def test_corpus_csv_errors_name_the_problem(tmp_path: Path, text: str, message: str):
    path = tmp_path / "metric.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        nab(path)


def test_label_timestamps_are_relative_and_windows_are_refused(tmp_path: Path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"realAWSCloudwatch/example.csv": ["2014-01-01 00:05:00"]}))
    start = timestamp("2014-01-01T00:00:00+00:00")
    assert labelled_incidents(
        path, str(tmp_path / "NAB/data/realAWSCloudwatch/example.csv"), start
    ) == [Incident(300, 300)]
    with pytest.raises(ValueError, match="exactly one"):
        labelled_incidents(path, "different.csv", start)
    path.write_text(json.dumps([["2014-01-01 00:00:00", "2014-01-01 00:10:00"]]))
    with pytest.raises(ValueError, match="validation errors"):
        labelled_incidents(path, "example.csv", start)


class _Table(Protocol):
    @property
    def schema(self) -> object: ...


@runtime_checkable
class _Arrow(Protocol):
    def table(self, data: dict[str, object]) -> _Table: ...


class _Writer(Protocol):
    def write_table(self, table: _Table) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class _IPC(Protocol):
    def new_stream(self, sink: BinaryIO, schema: object) -> _Writer: ...
    def new_file(self, sink: BinaryIO, schema: object) -> _Writer: ...


@pytest.mark.parametrize("container", ["stream", "file"])
def test_boom_reads_both_containers_and_caps_seeded_variate(tmp_path: Path, container: str):
    arrow: object = import_module("pyarrow")
    ipc: object = import_module("pyarrow.ipc")
    assert isinstance(arrow, _Arrow)
    assert isinstance(ipc, _IPC)
    (tmp_path / "dataset_taxonomy.json").write_text(json.dumps({"gauge": {"type": ["gauge"]}}))
    table = arrow.table(
        {
            "start": [datetime(2020, 1, 1, tzinfo=UTC)],
            "item_id": ["gauge"],
            "freq": ["10S"],
            "target": [[list(range(5000)), list(range(10000, 15000))]],
        }
    )
    path = tmp_path / "data.arrow"
    with path.open("wb") as stream:
        writer = (
            ipc.new_stream(stream, table.schema)
            if container == "stream"
            else ipc.new_file(stream, table.schema)
        )
        writer.write_table(table)
        writer.close()
    source = boom(tmp_path, "gauge")
    assert_array_equal(source.values, np.arange(10904, 15000))
    assert source.start_ts == timestamp("2020-01-01T00:00:00Z") + 9040
    assert source.spec.season_steps == 0
    assert source.spec.context_steps == 1024
    assert "possibly contaminated" in source.caveats[0]
    with pytest.raises(ValueError, match="type"):
        boom(tmp_path, "not-a-gauge")


@respx.mock
def test_tsdb_uses_target_protocol_and_refuses_multiple_series():
    config = load_config(Path("demo/promcast.yaml"), environment={})
    target = config.targets[0]
    start = datetime(2020, 1, 1, tzinfo=UTC)
    samples = [[start.timestamp() + i * 60, str(10 + i % 2)] for i in range(100)]

    def respond(request: Request) -> Response:
        assert request.url.params["query"] == target.query
        assert request.url.params["step"] == "60"
        return Response(
            200,
            json={
                "status": "success",
                "data": {"resultType": "matrix", "result": [{"metric": {}, "values": samples}]},
            },
        )

    route = respx.get("http://prometheus:9090/api/v1/query_range").mock(side_effect=respond)
    end = start + timedelta(minutes=99)
    source = asyncio.run(tsdb(config, target, start.isoformat(), end.isoformat()))
    assert source.values.size == 100
    assert source.spec.threshold == target.threshold
    assert source.spec.direction == "below"
    assert source.start_ts == start.timestamp()
    assert route.call_count == 1
    route.mock(
        return_value=Response(
            200,
            json={
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [
                        {"metric": {"instance": name}, "values": samples} for name in ("a", "b")
                    ],
                },
            },
        )
    )
    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(tsdb(config, target, start.isoformat(), end.isoformat()))


@respx.mock
@pytest.mark.parametrize("known_gauge", [True, False])
def test_tsdb_constant_series_uses_the_shared_gauge_policy(known_gauge: bool):
    config = load_config(Path("promcast.example.yaml"), environment={})
    target = config.targets[0].model_copy(update={"known_gauge": known_gauge})
    start = timestamp("2026-01-01T00:00:00Z")
    step = int(target.step.total_seconds())
    values = [[start + i * step, ".4"] for i in range(288)]
    respx.get(str(config.tsdb.url).rstrip("/") + "/api/v1/query_range").respond(
        200,
        json={
            "status": "success",
            "data": {"resultType": "matrix", "result": [{"metric": {}, "values": values}]},
        },
    )
    call = tsdb(
        config,
        target,
        "2026-01-01T00:00:00Z",
        datetime.fromtimestamp(start + 287 * step, UTC).isoformat(),
    )
    if known_gauge:
        source = asyncio.run(call)
        assert source.values.size == 288
    else:
        with pytest.raises(ReplayArgumentsError, match="looks like a counter"):
            asyncio.run(call)
