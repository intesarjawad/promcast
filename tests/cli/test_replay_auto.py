import argparse
import asyncio
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
import yaml

from promcast.core.forecaster import FakeForecaster
from promcast.shell import cli
from promcast.shell.auto_replay import prepare_auto


@respx.mock
def test_auto_prepares_every_target_and_series_with_thirty_day_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    config = yaml.safe_load(Path("demo/promcast.yaml").read_text())
    config["tsdb"] = {"url": "http://auto:9090"}
    first = config["targets"][0]
    first.update(
        step="5m", context="320m", horizon="50m", warn="10m", season="0s", linear_window="1h"
    )
    config["targets"] = [first, first | {"name": "second", "query": "second_metric"}]
    path = tmp_path / "promcast.yaml"
    path.write_text(yaml.safe_dump(config))
    requests: list[tuple[str, float, float]] = []
    end = 1700000100
    monkeypatch.setattr("promcast.shell.auto_replay.time.time", lambda: end)

    def respond(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        start, stop = float(params["start"]), float(params["end"])
        requests.append((params["query"], start, stop))
        values = [[end - (399 - i) * 300, str(100 - i / 10 + np.sin(i))] for i in range(400)]
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [
                        {"metric": {"instance": host}, "values": values} for host in ("a", "b")
                    ],
                },
            },
        )

    respx.get("http://auto:9090/api/v1/query_range").mock(side_effect=respond)
    arguments = argparse.Namespace(config=str(path))
    parsed, jobs, eta = asyncio.run(prepare_auto(arguments, FakeForecaster()))
    assert len(parsed.targets) == 2
    assert [(job.target.name, job.labels["instance"]) for job in jobs] == [
        (first["name"], "a"),
        (first["name"], "b"),
        ("second", "a"),
        ("second", "b"),
    ]
    assert all(job.source.values.size == 400 for job in jobs)
    assert all(stop - start <= 30 * 86400 for _, start, stop in requests)
    assert all("@ 1700000100" in query for query, _, _ in requests)
    assert eta >= 0
    assert "Replay ETA:" in capsys.readouterr().err

    def load(_path: Path, _name: str, _batch_size: int = 16) -> FakeForecaster:
        return FakeForecaster()

    monkeypatch.setattr(cli, "load_forecaster", load)
    output = tmp_path / "replays"
    monkeypatch.setattr(
        sys, "argv", ["promcast", "replay", "--auto", "--config", str(path), "--out", str(output)]
    )
    assert cli.main() == 0
    reports = list(output.rglob("replay.json"))
    assert len(reports) == 4
    assert all((report.parent / "recommended.yaml").exists() for report in reports)


@respx.mock
def test_auto_replay_omits_counter_probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    document = yaml.safe_load(Path("demo/promcast.yaml").read_text())
    document["tsdb"] = {"url": "http://auto:9090"}
    first = document["targets"][0]
    first.update(step="5m", context="320m", horizon="50m", warn="10m", season="0s")
    document["targets"] = [
        first | {"name": "counter"},
        first | {"name": "gauge", "known_gauge": True},
    ]
    path = tmp_path / "promcast.yaml"
    path.write_text(yaml.safe_dump(document))
    end = 1700000100
    monkeypatch.setattr("promcast.shell.auto_replay.time.time", lambda: end)
    values = [[end - i * 300, ".4"] for i in range(287, -1, -1)]
    respx.get("http://auto:9090/api/v1/query_range").respond(
        200,
        json={
            "status": "success",
            "data": {"resultType": "matrix", "result": [{"metric": {}, "values": values}]},
        },
    )
    _, jobs, _ = asyncio.run(prepare_auto(argparse.Namespace(config=str(path)), FakeForecaster()))
    assert [job.target.name for job in jobs] == ["gauge"]


@respx.mock
def test_insufficient_history_reports_required_coverage_and_collection_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    document = yaml.safe_load(Path("demo/promcast.yaml").read_text())
    document["tsdb"] = {"url": "http://auto:9090"}
    target = document["targets"][0]
    target.update(step="5m", context="4d", horizon="6h", warn="1h", season="1d")
    path = tmp_path / "promcast.yaml"
    path.write_text(yaml.safe_dump(document))
    end = 1700000100
    monkeypatch.setattr("promcast.shell.auto_replay.time.time", lambda: end)
    values = [[end - i * 300, ".4"] for i in range(863, -1, -1)]
    respx.get("http://auto:9090/api/v1/query_range").respond(
        200,
        json={
            "status": "success",
            "data": {"resultType": "matrix", "result": [{"metric": {}, "values": values}]},
        },
    )

    def load(_path: Path, _name: str, _batch_size: int = 16) -> FakeForecaster:
        return FakeForecaster()

    monkeypatch.setattr(cli, "load_forecaster", load)
    monkeypatch.setattr(sys, "argv", ["promcast", "replay", "--auto", "--config", str(path)])
    assert cli.main() == 2
    message = capsys.readouterr().err
    assert "71h 55m available" in message
    assert "203h 55m required" in message
    assert "collect at least 132h 00m more" in message
    assert "promcast replay --auto" in message
    assert "promcast scan" not in message
