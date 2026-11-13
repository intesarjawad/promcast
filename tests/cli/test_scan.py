import argparse
import asyncio
import base64
import json
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
import yaml
from pydantic import TypeAdapter

from promcast.core.config import Config, TSDBConfig, parse_config
from promcast.core.forecaster import FakeForecaster
from promcast.core.recipes import scan_target
from promcast.core.replay import forecast_at
from promcast.core.series import FloatArray, Series
from promcast.shell import cli
from promcast.shell.adoption import write_init
from promcast.shell.auto_replay import prepare_auto
from promcast.shell.discover import load_recipes
from promcast.shell.exposition import Collector
from promcast.shell.plotting import scan_board
from promcast.shell.rules import generate_rules
from promcast.shell.scan import ScanDocument, ScanOptions, redact, scan
from promcast.shell.scan_cli import discover_config
from promcast.shell.scheduler import Clock, CycleResources, Scheduler
from promcast.shell.store import Store

END = 1700000100
URL = "http://localhost:9090"
DISK = next(r for r in load_recipes() if r.name == "disk")


class FlatForecast(FakeForecaster):
    def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
        return [
            np.tile(np.linspace(0.06, 0.01, horizon_steps), (9, 1))
            if dict(item.labels).get("instance") == "host-119"
            else np.full((9, horizon_steps), 0.5)
            for item in batch
        ]


def install_routes(
    *, samples: int = 80, scrape: int = 15, fleet: int = 1, gaps: float | None = None
) -> list[int]:
    labels = [
        {"instance": f"host-{i:03}", "mountpoint": "/data", "fstype": "ext4", "device": "disk"}
        for i in range(fleet)
    ]
    selected = labels[-1:] + labels[:49] if fleet > 50 else labels
    requests: list[int] = []

    def instant(request: httpx.Request) -> httpx.Response:
        data: dict[str, object]
        query = request.url.params["query"]
        if query == "1":
            data = {"resultType": "scalar", "result": [END, "1"]}
        elif query.startswith("count("):
            data = {"resultType": "vector", "result": [{"metric": {}, "value": [END, str(fleet)]}]}
        elif query.startswith("count_over_time"):
            data = {
                "resultType": "vector",
                "result": [{"metric": label, "value": [END, str(samples)]} for label in selected],
            }
        elif query.startswith("node_filesystem_avail_bytes[10m]"):
            spacing = gaps or scrape
            raw = [
                [END - j * spacing, "10"]
                for j in range(min(samples, int(600 / spacing) + 1) - 1, -1, -1)
            ]
            data = {
                "resultType": "matrix",
                "result": [{"metric": label, "values": raw} for label in selected],
            }
        else:
            assert "last_over_time" in query
            data = {
                "resultType": "vector",
                "result": [{"metric": label, "value": [END, ".5"]} for label in selected],
            }
        return httpx.Response(200, json={"status": "success", "data": data})

    def history(request: httpx.Request) -> httpx.Response:
        step = int(request.url.params["step"])
        requests.append(step)
        assert int(float(request.url.params["end"])) == END
        start = END - (samples - 1) * scrape
        count = max(1, (END - start) // step + 1)
        values = [
            [END - (count - 1 - i) * step, str(0.05 + (720 + (count - 1 - i) * step) / 100000)]
            for i in range(count)
        ]
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [{"metric": label, "values": values} for label in selected],
                },
            },
        )

    respx.get(URL + "/api/v1/query").mock(side_effect=instant)
    respx.get(URL + "/api/v1/query_range").mock(side_effect=history)
    respx.get(URL + "/api/v1/label/__name__/values").respond(
        200, json={"status": "success", "data": DISK.requires}
    )
    return requests


def run_scan():
    async def run():
        async with httpx.AsyncClient() as client:
            return await scan(
                client,
                Config(tsdb=TSDBConfig.model_validate({"url": URL}), targets=[]),
                [DISK],
                FlatForecast(),
                ScanOptions(END),
            )

    return asyncio.run(run())


@pytest.mark.parametrize(
    ("samples", "scrape", "expected_step", "status"),
    [
        (40, 15, 15, "short history: 10 min"),
        (20, 15, 15, "warming up, 3m to first forecast"),
        (31, 60, 60, "warming up"),
        (32, 60, 60, "short history"),
        (1, 15, 15, "warming up"),
    ],
)
@respx.mock
def test_history_ladder_and_real_sample_gate(
    samples: int, scrape: int, expected_step: int, status: str
):
    requests = install_routes(samples=samples, scrape=scrape)
    result = run_scan()
    assert requests[-1] == expected_step
    board = scan_board(result)
    assert status in board
    if samples == 40:
        assert "clear" in board
        assert result.document.rows[0].linear_breach_ts is None
        assert result.document.rows[0].horizon_s == 600
    if samples < 32:
        assert not result.document.rows
        assert result.document.skipped[0].reason == "too_short"


@respx.mock
def test_scrape_jitter_rounds_to_fifteen_seconds():
    requests = install_routes(samples=40, gaps=15.002)
    assert run_scan().document.rows
    assert requests == [300, 60, 15]


@respx.mock
def test_board_reports_effective_fifteen_second_horizon():
    install_routes(samples=80)
    result = run_scan()
    assert "2h" in scan_board(result).splitlines()[4]
    assert result.document.rows[0].horizon_s == 7680


@respx.mock
def test_nearest_candidate_that_sorts_last_is_on_board_and_footer():
    install_routes(fleet=120)
    result = run_scan()
    assert any(row.labels["instance"] == "host-119" for row in result.document.rows)
    assert result.document.rows[0].labels["instance"] == "host-119"
    board = scan_board(result)
    assert "host-119" in board
    assert "70 more series not forecast; rerun with --max-series 120" in board
    assert max(map(len, board.splitlines())) <= 80


@respx.mock
def test_redaction_stable_and_json_schema_shape():
    install_routes(fleet=2)
    document = run_scan().document
    first = redact(document).model_dump_json()
    assert first == redact(document).model_dump_json()
    assert "host-000" not in first
    assert "mount-" in first
    assert "host-" in first
    assert ScanDocument.model_validate_json(first).rows
    schema = json.loads(Path("docs/schema/scan.schema.json").read_text())
    assert set(json.loads(first)) == set(schema["properties"])
    assert set(json.loads(first)["rows"][0]) == set(schema["$defs"]["ScanRow"]["properties"])


@respx.mock
def test_empty_cli_offers_oci_demo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    respx.get(URL + "/api/v1/query").respond(
        200, json={"status": "success", "data": {"resultType": "scalar", "result": [END, "1"]}}
    )
    respx.get(URL + "/api/v1/label/__name__/values").respond(
        200, json={"status": "success", "data": []}
    )
    monkeypatch.setattr(sys, "argv", ["promcast", "scan", URL])
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "docker compose -f oci://ghcr.io/intesarjawad/promcast-demo:0.1.0 up" in output
    assert "promcast scan --query '<PromQL>' --below <threshold>" in output
    assert output.count("next ") == 2


def test_max_series_501_is_two_line_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(sys, "argv", ["promcast", "scan", "--max-series", "501"])
    assert cli.main() == 2
    assert len(capsys.readouterr().err.splitlines()) == 2


@respx.mock
def test_init_validates_and_preserves_existing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    install_routes(samples=40)
    monkeypatch.setattr("promcast.shell.scan_cli.time.time", lambda: END)

    def load(_path: Path, _name: str, _batch_size: int = 16) -> FlatForecast:
        return FlatForecast()

    monkeypatch.setattr(cli, "load_forecaster", load)
    monkeypatch.setattr(sys, "argv", ["promcast", "init", URL, "--out", str(tmp_path)])
    assert cli.main() == 0
    config = parse_config((tmp_path / "promcast.yaml").read_text(), environment={})
    assert config.targets[0].query == DISK.query
    assert config.tsdb.url.host == "host.docker.internal"
    original = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert cli.main() == 2
    assert all((tmp_path / name).read_bytes() == data for name, data in original.items())
    assert "--force" in capsys.readouterr().err
    compose = yaml.safe_load((tmp_path / "compose.yml").read_text())
    assert compose["services"]["promcast"]["extra_hosts"] == ["host.docker.internal:host-gateway"]
    assert (tmp_path / "grafana/dashboards/promcast.json").exists()
    write_init(config, tmp_path, force=True)
    assert TypeAdapter(dict[str, object]).validate_json(
        (tmp_path / "grafana/dashboards/promcast.json").read_text()
    )


@respx.mock
def test_json_cli_is_clean_and_redaction_repeats(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    install_routes(samples=40, fleet=2)
    monkeypatch.setattr("promcast.shell.scan_cli.time.time", lambda: END)

    def load(_path: Path, _name: str, _batch_size: int = 16) -> FlatForecast:
        return FlatForecast()

    monkeypatch.setattr(cli, "load_forecaster", load)
    monkeypatch.setattr(sys, "argv", ["promcast", "scan", URL, "--json", "--redact"])
    assert cli.main() == 0
    first = capsys.readouterr().out
    assert ScanDocument.model_validate_json(first).rows
    assert cli.main() == 0
    assert capsys.readouterr().out == first
    assert "host-000" not in first


@respx.mock
def test_svg_flag_writes_the_redacted_board(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    install_routes(samples=40, fleet=2)
    monkeypatch.setattr("promcast.shell.scan_cli.time.time", lambda: END)

    def load(_path: Path, _name: str, _batch_size: int = 16) -> FlatForecast:
        return FlatForecast()

    monkeypatch.setattr(cli, "load_forecaster", load)
    target = tmp_path / "board.svg"
    monkeypatch.setattr(sys, "argv", ["promcast", "scan", URL, "--redact", "--svg", str(target)])
    assert cli.main() == 0
    svg = target.read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "host-000" not in svg
    assert "host-" in capsys.readouterr().out


@respx.mock
def test_recipe_gauge_does_not_become_a_counter_when_constant():
    install_routes(samples=40)

    def constant(request: httpx.Request) -> httpx.Response:
        step = int(request.url.params["step"])
        count = max(1, 39 * 15 // step + 1)
        label = {"instance": "host-000", "mountpoint": "/data", "fstype": "ext4", "device": "disk"}
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [
                        {
                            "metric": label,
                            "values": [[END - i * step, ".5"] for i in range(count - 1, -1, -1)],
                        }
                    ],
                },
            },
        )

    respx.get(URL + "/api/v1/query_range").mock(side_effect=constant)
    result = run_scan()
    assert len(result.document.rows) == 1
    assert not result.document.skipped


@respx.mock
def test_short_history_plain_snapshot():
    install_routes(samples=40)
    result = replace(run_scan(), forecasts_per_s=1000)
    for width in (80, 160):
        expected = Path(f"tests/cli/__snapshots__/scan-short-{width}.txt").read_text()
        assert scan_board(result, width=width) == expected


def test_multi_target_rules_have_distinct_static_identities():
    config = Config(
        tsdb=TSDBConfig.model_validate({"url": URL}),
        targets=[scan_target(r, 15, 50) for r in load_recipes()],
    )
    rules = yaml.safe_load(generate_rules(config))["groups"][0]["rules"]
    assert len({(rule["alert"], rule["labels"]["target"]) for rule in rules}) == 12


@respx.mock
def test_ad_hoc_compound_query_uses_a_real_metric_for_scrape_detection():
    install_routes()
    arguments = argparse.Namespace(
        url=URL,
        recipes=None,
        max_series=50,
        query="(node_filesystem_avail_bytes / node_filesystem_size_bytes)",
        below=0.05,
        above=None,
    )

    async def run():
        async with httpx.AsyncClient() as client:
            return await discover_config(arguments, client)

    _, recipes = asyncio.run(run())
    assert recipes[0].requires == ["node_filesystem_avail_bytes"]
    assert recipes[0].threshold == 0.05


@respx.mock
def test_discovery_uses_the_cache_directory_override(monkeypatch: pytest.MonkeyPatch):
    install_routes()
    monkeypatch.setenv("PROMCAST_CACHE_DIR", "/cache/models")
    arguments = argparse.Namespace(
        url=URL, recipes=None, max_series=50, query=None, below=None, above=None
    )

    async def run():
        async with httpx.AsyncClient() as client:
            return await discover_config(arguments, client)

    config, _ = asyncio.run(run())
    assert config.model.cache_dir == Path("/cache/models")


@respx.mock
def test_constant_gauge_survives_scan_adoption_serve_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):

    install_routes(samples=288, scrape=300)
    values = [[END - i * 300, ".5"] for i in range(287, -1, -1)]
    matrix = {
        "resultType": "matrix",
        "result": [
            {
                "metric": {
                    "instance": "host-000",
                    "mountpoint": "/data",
                    "fstype": "ext4",
                    "device": "disk",
                },
                "values": values,
            }
        ],
    }
    respx.get(URL + "/api/v1/query_range").respond(200, json={"status": "success", "data": matrix})
    result = run_scan()
    assert len(result.document.rows) == 1
    targets = [
        target.model_copy(update={"context": target.step * 288}) for target in result.targets
    ]
    config = Config(tsdb=TSDBConfig.model_validate({"url": URL}), targets=targets)
    write_init(config, tmp_path)
    adopted = parse_config((tmp_path / "promcast.yaml").read_text(), environment={})
    target = adopted.targets[0]

    async def serve_cycle() -> None:
        store = Store()
        collector = Collector(store, "0.1.0", "fake", "fixture")
        async with httpx.AsyncClient() as client:
            config = adopted.model_copy(update={"tsdb": TSDBConfig.model_validate({"url": URL})})
            scheduler = Scheduler(
                config,
                CycleResources(client, FakeForecaster(), store, collector),
                Clock(wall=lambda: END),
            )
            await scheduler.cycle()
        assert len(store.snapshot()[0].forecasts) == 2

    asyncio.run(serve_cycle())
    # A short context permits a replay probe with the same 288 observations.
    target = target.model_copy(update={"context": target.step * 64, "horizon": target.step * 10})
    adopted = adopted.model_copy(update={"targets": [target]})

    def load(_path: Path, **_kwargs: object) -> Config:
        return adopted

    monkeypatch.setattr("promcast.shell.auto_replay.load_config", load)
    monkeypatch.setattr("promcast.shell.auto_replay.time.time", lambda: END)
    adopted = adopted.model_copy(update={"tsdb": TSDBConfig.model_validate({"url": URL})})
    arguments = argparse.Namespace(config=str(tmp_path / "promcast.yaml"))
    _, jobs, _ = asyncio.run(prepare_auto(arguments, FakeForecaster()))
    assert len(jobs) == 1
    assert (
        forecast_at(jobs[0].source.values, 100, jobs[0].source.spec, FakeForecaster()) is not None
    )


@respx.mock
def test_redacted_url_and_aliases_survive_population_changes():
    install_routes(fleet=2)
    document = run_scan().document.model_copy(update={"url": "https://private.example/tenant/path"})
    original = redact(document)
    smaller = redact(document.model_copy(update={"rows": document.rows[1:]}))
    assert original.rows[1].labels == smaller.rows[0].labels
    assert original.url == "<redacted>"
    assert "private.example" not in original.model_dump_json()
    assert "/tenant/path" not in scan_board(replace(run_scan(), document=original), width=80)


@pytest.mark.parametrize("command", ["scan", "init", "replay"])
def test_discovery_help_explains_defaults_and_thresholds(
    command: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(sys, "argv", ["promcast", command, "--help"])
    with pytest.raises(SystemExit) as exit_code:
        cli.main()
    assert exit_code.value.code == 0
    text = capsys.readouterr().out
    assert "PROMETHEUS_URL" in text
    assert "default: 50" in text
    assert "threshold in query units" in " ".join(text.split())


@pytest.mark.parametrize("command", ["scan", "init"])
@pytest.mark.parametrize("authentication", ["bearer", "basic", "headers"])
@respx.mock
def test_authenticated_discovery_reads_configuration(
    command: str,
    authentication: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    install_routes(samples=40)
    secret = tmp_path / "credential"
    secret.write_text("private-token")
    auth = (
        {"bearer_token_file": str(secret)}
        if authentication == "bearer"
        else {"basic_auth": {"username": "reader", "password_file": str(secret)}}
        if authentication == "basic"
        else {"headers": {"Authorization": "Bearer ${SCAN_TOKEN}", "X-Scope-OrgID": "tenant"}}
    )
    expected = (
        "Basic " + base64.b64encode(b"reader:private-token").decode()
        if authentication == "basic"
        else "Bearer private-token"
    )
    path = tmp_path / "auth.yaml"
    path.write_text(yaml.safe_dump({"tsdb": {"url": URL, **auth}, "targets": []}))
    monkeypatch.setenv("SCAN_TOKEN", "private-token")
    monkeypatch.setattr("promcast.shell.scan_cli.time.time", lambda: END)

    def load(_path: Path, _name: str, _batch_size: int = 16) -> FakeForecaster:
        return FakeForecaster()

    monkeypatch.setattr(cli, "load_forecaster", load)
    flags = ["--json"] if command == "scan" else ["--out", str(tmp_path / "adopted")]
    monkeypatch.setattr(sys, "argv", ["promcast", command, "--config", str(path), *flags])
    assert cli.main() == 0
    assert respx.calls
    assert all(
        route.calls.last.request.headers.get("Authorization") == expected
        for route in respx.routes
        if route.calls.called
    )
    if authentication == "headers":
        assert all(
            route.calls.last.request.headers["X-Scope-OrgID"] == "tenant"
            for route in respx.routes
            if route.calls.called
        )
    output = capsys.readouterr()
    assert "private-token" not in output.out + output.err


@pytest.mark.parametrize("flag", ["--below", "--above"])
@pytest.mark.parametrize("threshold", ["nan", "inf", "-inf"])
def test_nonfinite_threshold_is_an_argument_error(
    flag: str, threshold: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(sys, "argv", ["promcast", "scan", "--query", "up", f"{flag}={threshold}"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert "finite" in output
    assert "unexpected" not in output
    assert "issues/new" not in output
