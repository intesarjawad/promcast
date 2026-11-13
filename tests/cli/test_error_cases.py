import argparse
import asyncio
import sys
from pathlib import Path

import httpx
import pytest
import respx
import yaml

from promcast.core.config import ConfigError
from promcast.core.forecaster import FakeForecaster
from promcast.core.recipes import scan_target
from promcast.shell import cli, errors
from promcast.shell.auto_replay import prepare_auto
from promcast.shell.discover import load_recipes
from promcast.shell.scan import ScanDocument, ScanResult

URL = "http://localhost:9090"
DISK = next(recipe for recipe in load_recipes() if recipe.name == "disk")
BUG_URL = "https://github.com/intesarjawad/promcast/issues/new?template=bug.yml"


def fail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *argv: str,
    what: str,
    next_step: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["promcast", *argv])
    assert cli.main() == 2
    output = capsys.readouterr()
    assert output.out == ""
    lines = [line for line in output.err.splitlines() if not line.startswith("Downloading ")]
    assert len(lines) == 2
    assert lines[0].startswith("error: ")
    assert what in lines[0]
    assert lines[1].startswith("next:  ")
    assert next_step in lines[1]


def prometheus(*names: str, labels: httpx.Response | None = None) -> None:
    respx.get(URL + "/api/v1/query").respond(
        200, json={"status": "success", "data": {"resultType": "scalar", "result": [1, "1"]}}
    )
    respx.get(URL + "/api/v1/label/__name__/values").mock(
        return_value=labels or httpx.Response(200, json={"status": "success", "data": list(names)})
    )


def local_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    text = Path("promcast.example.yaml").read_text()
    path.write_text(text.replace("~/.cache/promcast/models", str(tmp_path / "models")))
    return path


@respx.mock
def test_explicit_tsdb_status(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    respx.get(URL + "/api/v1/query").respond(503)
    fail(
        monkeypatch,
        capsys,
        "scan",
        URL,
        what="TSDB returned HTTP 503",
        next_step="promcast scan http://localhost:9090",
    )


@respx.mock
def test_tsdb_unavailable(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    respx.get(URL + "/api/v1/query").respond(
        200, json={"status": "success", "data": {"resultType": "scalar", "result": [1, "1"]}}
    )
    respx.get(URL + "/api/v1/label/__name__/values").mock(side_effect=httpx.ConnectError("down"))
    fail(
        monkeypatch,
        capsys,
        "scan",
        URL,
        what="connection failed",
        next_step="edit tsdb.url",
    )


@respx.mock
@pytest.mark.parametrize("status", [401, 403])
def test_tsdb_auth_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], status: int
):
    prometheus(labels=httpx.Response(status, text="denied"))
    fail(
        monkeypatch,
        capsys,
        "scan",
        URL,
        what=f"TSDB rejected the credentials (HTTP {status})",
        next_step="edit tsdb.basic_auth",
    )


@respx.mock
def test_tsdb_rejected(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    prometheus(labels=httpx.Response(500, text="internal error"))
    fail(
        monkeypatch,
        capsys,
        "scan",
        URL,
        what="TSDB returned HTTP 500: internal error",
        next_step="promcast scan http://localhost:9090",
    )


def test_config_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    text = Path("promcast.example.yaml").read_text().replace("quantile: 0.9", "quantile: 0.95")
    config = tmp_path / "invalid.yaml"
    config.write_text(text)
    fail(
        monkeypatch,
        capsys,
        "validate",
        str(config),
        what="targets.0.quantile",
        next_step=f"edit {config}, then run promcast validate {config}",
    )


def test_config_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    # replay --auto reads the file earlier, so only prepare_auto itself reports it missing.
    missing = tmp_path / "missing.yaml"

    def prepare(arguments: argparse.Namespace) -> None:
        asyncio.run(prepare_auto(arguments, FakeForecaster()))

    monkeypatch.setattr(cli, "_replay_auto", prepare)
    fail(
        monkeypatch,
        capsys,
        *("replay", "--auto", "--config", str(missing), "--out", str(tmp_path)),
        what=f"config does not exist: {missing}",
        next_step="promcast init",
    )


def test_compare_extra_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def unavailable(name: str) -> object:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr("promcast.core.forecaster_chronos.import_module", unavailable)
    fail(
        monkeypatch,
        capsys,
        *("replay", "--source", "synthetic", "--models", "chronos-2-small"),
        *("--out", str(tmp_path)),
        what="install the compare extra to use chronos-2-small",
        next_step="uvx --from 'promcast[compare]' promcast replay",
    )


def test_unknown_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fail(
        monkeypatch,
        capsys,
        *("replay", "--source", "synthetic", "--models", "nope", "--out", str(tmp_path)),
        what="unknown model: nope",
        next_step="promcast replay --models toto-2.0-22m",
    )


def test_model_download_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def offline(*_args: object, **_kwargs: object) -> str:
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", offline)
    fail(
        monkeypatch,
        capsys,
        "serve",
        "--config",
        str(local_config(tmp_path)),
        what="download or checksum verification failed",
        next_step="promcast scan after restoring network access",
    )


def test_max_series_out_of_range(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fail(
        monkeypatch,
        capsys,
        *("scan", "--max-series", "501"),
        what="--max-series 501 is outside 1..500",
        next_step="promcast scan --max-series 50",
    )


@respx.mock
def test_query_threshold_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    prometheus("up")
    fail(
        monkeypatch,
        capsys,
        *("scan", URL, "--query", "up"),
        what="--query requires --below or --above",
        next_step="promcast scan --query '<PromQL>' --below 0",
    )


@respx.mock
def test_query_metric_unknown(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    prometheus("up")
    fail(
        monkeypatch,
        capsys,
        *("scan", URL, "--query", "nothing_here", "--below", "0"),
        what="cannot identify a metric in nothing_here",
        next_step="edit a recipe YAML",
    )


@respx.mock
def test_no_recipe_matched_for_init(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    prometheus()
    fail(
        monkeypatch,
        capsys,
        *("init", URL, "--out", str(tmp_path)),
        what="no recipe metrics matched",
        next_step="promcast scan --query '<PromQL>' --below 0",
    )


@respx.mock
def test_no_recipe_matched_in_recipes_folder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    prometheus()
    (tmp_path / "disk.yaml").write_text(
        (Path("src/promcast/recipes") / "disk.yaml").read_text(encoding="utf-8")
    )
    fail(
        monkeypatch,
        capsys,
        *("scan", URL, "--recipes", str(tmp_path)),
        what=f"no recipe metric matched --recipes {tmp_path}",
        next_step="promcast scan --query '<PromQL>' --below 0",
    )


@respx.mock
def test_recipes_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    prometheus(*DISK.requires)
    fail(
        monkeypatch,
        capsys,
        *("scan", URL, "--recipes", str(tmp_path)),
        what=f"no recipes in {tmp_path}",
        next_step="promcast scan, without --recipes",
    )


@respx.mock
def test_recipe_invalid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    prometheus(*DISK.requires)
    (tmp_path / "broken.yaml").write_text("name: broken\n")
    fail(
        monkeypatch,
        capsys,
        *("scan", URL, "--recipes", str(tmp_path)),
        what="invalid recipe: ",
        next_step="edit the recipe YAML",
    )


@respx.mock
def test_init_target_exists(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    prometheus(*DISK.requires)
    (tmp_path / "promcast.yaml").write_text("keep")

    async def scan(*_arguments: object) -> ScanResult:
        document = ScanDocument(url=URL, model="fake", origin_ts=0, rows=[], skipped=[])
        return ScanResult(document, [scan_target(DISK, 15, 50)], 0, 0, 0.0)

    def load(_path: Path, _name: str, _batch_size: int = 16) -> FakeForecaster:
        return FakeForecaster()

    monkeypatch.setattr(cli, "load_forecaster", load)
    monkeypatch.setattr("promcast.shell.scan_cli.scan", scan)
    fail(
        monkeypatch,
        capsys,
        *("init", URL, "--out", str(tmp_path)),
        what=f"init target exists: {tmp_path / 'promcast.yaml'}",
        next_step="promcast init --out DIR, or add --force",
    )
    assert (tmp_path / "promcast.yaml").read_text() == "keep"


def test_replay_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fail(
        monkeypatch,
        capsys,
        *("replay", "--source", "nab", "--out", str(tmp_path)),
        what="--input is required for a NAB CSV or BOOM folder",
        next_step="promcast replay --auto",
    )


@respx.mock
def test_replay_history_too_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    config = yaml.safe_load(Path("demo/promcast.yaml").read_text())
    config["tsdb"] = {"url": "http://auto:9090"}
    path = tmp_path / "promcast.yaml"
    path.write_text(yaml.safe_dump(config))
    values = [[1700000000 + i * 15, "1"] for i in range(10)]
    respx.get("http://auto:9090/api/v1/query_range").respond(
        200,
        json={
            "status": "success",
            "data": {"resultType": "matrix", "result": [{"metric": {"a": "b"}, "values": values}]},
        },
    )

    def load(_path: Path, _name: str, _batch_size: int = 16) -> FakeForecaster:
        return FakeForecaster()

    monkeypatch.setattr(cli, "load_forecaster", load)
    fail(
        monkeypatch,
        capsys,
        *("replay", "--auto", "--config", str(path), "--out", str(tmp_path / "out")),
        what="no series has enough history for replay --auto",
        next_step="collect at least 51h 59m more, "
        "then rerun the same promcast replay --auto command",
    )


def test_replay_probe_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    # The fake forecaster never fails a probe, so this checks only how main reports it.
    def unavailable(_arguments: object) -> None:
        raise errors.ReplayProbeUnavailableError()

    monkeypatch.setattr(cli, "_replay_auto", unavailable)
    fail(
        monkeypatch,
        capsys,
        *("replay", "--auto", "--out", str(tmp_path)),
        what="no eligible inference probe for replay ETA",
        next_step="promcast scan",
    )


def test_unexpected_failure_exits_one_with_the_bug_report_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def broken(_arguments: object) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_dispatch", broken)
    monkeypatch.setattr(sys, "argv", ["promcast", "rules", "--config", "promcast.example.yaml"])
    assert cli.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.splitlines() == [
        "error: unexpected RuntimeError: boom",
        f"next:  report it at {BUG_URL}",
    ]


def test_configuration_errors_from_any_command_use_the_two_line_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def broken(_arguments: object) -> int:
        raise ConfigError("bad value")

    monkeypatch.setattr(cli, "_dispatch", broken)
    fail(
        monkeypatch,
        capsys,
        *("doctor", "--config", str(tmp_path / "c.yaml")),
        what="bad value",
        next_step=f"edit {tmp_path / 'c.yaml'}",
    )


@pytest.mark.parametrize("flag", ["--max-series", "--below"])
def test_argument_errors_sanitize_url_values(
    flag: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    url = "https://alice:password@metrics.example/?token=secret"
    monkeypatch.setattr(sys, "argv", ["promcast", "scan", flag, url])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    output = capsys.readouterr().err
    for secret in ("alice", "password", "secret"):
        assert secret not in output
    assert "metrics.example" in output
