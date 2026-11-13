import importlib.util
import shlex
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from promcast.shell.config import load_config

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import linear_parity
else:
    spec = importlib.util.spec_from_file_location(
        "linear_parity", ROOT / "scripts/linear_parity.py"
    )
    assert spec is not None
    assert spec.loader is not None
    linear_parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(linear_parity)

ORIGIN = 1_700_000_000.0
ARGS = [
    "--target",
    "demo_disk_free",
    "--query",
    "demo_disk_free_bytes",
    "--threshold",
    "0",
    "--window-s",
    "3600",
    "--timeout-s",
    "0",
]


def _vector(value: float | None) -> dict[str, Any]:
    result: list[dict[str, Any]] = (
        [] if value is None else [{"metric": {}, "value": [ORIGIN, repr(value)]}]
    )
    return {"status": "success", "data": {"resultType": "vector", "result": result}}


def _prometheus(
    breach_s: float | None, predicted_s: float | None
) -> tuple[linear_parity.Fetch, list[dict[str, str]]]:
    calls: list[dict[str, str]] = []

    def fetch(url: str) -> dict[str, Any]:
        parsed = urllib.parse.urlsplit(url)
        assert parsed.path == "/api/v1/query"
        query = dict(urllib.parse.parse_qsl(parsed.query))
        calls.append(query)
        if query["query"].startswith("promcast_forecast_origin"):
            return _vector(ORIGIN)
        if query["query"].startswith("promcast_breach_timestamp_seconds"):
            return _vector(None if breach_s is None else ORIGIN + breach_s)
        return _vector(None if predicted_s is None else ORIGIN + predicted_s)

    return fetch, calls


def test_predict_linear_query_solves_for_the_threshold_crossing():
    target = linear_parity.Target("demo_disk_free", "demo_disk_free_bytes", 0.0, 3600)
    assert linear_parity.predict_linear_breach(target) == (
        "time() + (0.0 - predict_linear(demo_disk_free_bytes[3600s:60s], 0))"
        " / deriv(demo_disk_free_bytes[3600s:60s])"
    )


@pytest.mark.parametrize(
    ("promcast_s", "prometheus_s", "expected"),
    [(3420.0, 3420.0, True), (3454.0, 3420.0, True), (3455.0, 3420.0, False), (10.0, -5.0, False)],
)
def test_agreement_is_one_percent_of_predict_linear(
    promcast_s: float, prometheus_s: float, *, expected: bool
):
    assert linear_parity.agrees(promcast_s, prometheus_s) is expected


def test_main_evaluates_predict_linear_at_the_forecast_origin(capsys: pytest.CaptureFixture[str]):
    fetch, calls = _prometheus(3420.0, 3430.0)
    assert linear_parity.main(ARGS, fetch) == 0
    assert calls[-1]["time"] == f"{ORIGIN:.3f}"
    assert 'method="linear"' in calls[1]["query"]
    assert capsys.readouterr().out == (
        "time to breach: promcast 3420.0 s, predict_linear 3430.0 s\n"
    )


@pytest.mark.parametrize(
    ("breach_s", "predicted_s", "message"),
    [
        (3420.0, 3800.0, "differ by more than 1%"),
        (None, 3420.0, "no linear breach forecast for demo_disk_free"),
        (3420.0, None, "predict_linear has no breach"),
    ],
)
def test_main_fails_with_the_reason(
    capsys: pytest.CaptureFixture[str],
    breach_s: float | None,
    predicted_s: float | None,
    message: str,
):
    fetch, _ = _prometheus(breach_s, predicted_s)
    assert linear_parity.main(ARGS, fetch) == 1
    assert message in capsys.readouterr().err


def test_main_reports_an_unexpected_response(capsys: pytest.CaptureFixture[str]):
    assert linear_parity.main(ARGS, lambda _url: {"status": "error", "error": "bad"}) == 1
    assert "Prometheus rejected" in capsys.readouterr().err


def test_ci_compares_the_target_the_demo_configures():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["compose-smoke"]["steps"]
        if "scripts/linear_parity.py" in step.get("run", "")
    )
    words = shlex.split(step["run"])
    flags = dict(zip(words[2::2], words[3::2], strict=True))
    config = load_config(ROOT / "demo/promcast.yaml", environment={})
    target = next(target for target in config.targets if target.name == flags["--target"])
    assert flags["--query"] == target.query
    assert float(flags["--threshold"]) == target.threshold
    assert int(flags["--window-s"]) == int(target.linear_window.total_seconds())
    assert target.direction == "below"
