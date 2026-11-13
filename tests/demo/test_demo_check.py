import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "demo_check.sh"
FORECAST = {
    "status": "success",
    "data": {
        "resultType": "matrix",
        "result": [{"metric": {"target": "demo_disk_free"}, "values": [[1060, "0"], [1120, "-1"]]}],
    },
}
FAKE_COMMAND = """
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["DEMO_TEST_ROOT"])
clock = root / "clock"
now = int(clock.read_text())
command = Path(sys.argv[0]).name
if command == "date":
    print(now)
elif command == "sleep":
    assert sys.argv[1:] == ["5"]
    clock.write_text(str(now + int(os.environ.get("DEMO_TEST_ADVANCE", "5"))))
else:
    frames = json.loads((root / "frames.json").read_text())
    frame = frames[min(
        (now - int(os.environ["DEMO_TEST_INITIAL_TIME"])) // int(os.environ["DEMO_TEST_ADVANCE"]),
        len(frames) - 1,
    )]
    with (root / "requests").open("a") as stream:
        stream.write(json.dumps(sys.argv[1:]) + "\\n")
    if any("/api/v1/status/runtimeinfo" in arg for arg in sys.argv):
        payload = frame.get("runtimeinfo", {
            "status": "success", "data": {"startTime": "1970-01-01T00:16:40.123Z"}
        })
    elif any("http://metrics.test:9090/api/v1/query_range" == arg for arg in sys.argv):
        assert (
            'query=ALERTS{alertstate="firing",'
            'alertname=~"PromcastBreachSoon|DemoLinearBreachSoon|DemoDiskFull"}'
        ) in sys.argv
        assert "start=1000" in sys.argv
        assert f"end={now}" in sys.argv
        assert "step=1" in sys.argv
        payload = frame.get("history", {"status": "success", "data": {
            "resultType": "matrix", "result": [
                {"metric": alert["labels"], "values": [[1000 + index * 5, "1"]]}
                for index, previous in enumerate(frames[:(now - 1000) // 5 + 1])
                if isinstance(previous.get("alerts"), dict)
                for alert in previous["alerts"]["data"]["alerts"]
                if alert["state"] == "firing"
            ]
        }})
    elif any("/api/v1/query_range" in arg for arg in sys.argv):
        assert 'query=promcast_forecast_q90{target="demo_disk_free"}' in sys.argv
        assert "step=60" in sys.argv
        assert f"start={now}" in sys.argv
        assert f"end={now + 7200}" in sys.argv
        payload = frame["forecast"]
    elif any("http://forecast.test:8787/metrics" == arg for arg in sys.argv):
        payload = frame.get("metrics", 'promcast_selfaudit_coverage_ratio{'
            'target="demo_disk_free",lead="10m",method="toto"} 0.8')
    else:
        assert any("/api/datasources/uid/promcast/health" in arg for arg in sys.argv)
        payload = frame.get("health", {"status": "OK"})
    if payload == "connection refused":
        sys.exit(22)
    print(payload if isinstance(payload, str) else json.dumps(payload))
"""


class FirstFiring(BaseModel):
    PromcastBreachSoon: int | None
    DemoLinearBreachSoon: int | None
    DemoDiskFull: int | None


class CheckResult(BaseModel):
    seconds_to_firing: int | None
    first_firing: FirstFiring
    q90_data: bool
    grafana_health: bool
    selfaudit_coverage: bool
    timed_out: bool
    promcast_lead_s: int | None = None
    linear_lead_s: int | None = None


def alerts(*names: str, state: str = "firing") -> dict[str, object]:
    return {
        "status": "success",
        "data": {"alerts": [{"labels": {"alertname": name}, "state": state} for name in names]},
    }


def run_check(
    tmp_path: Path,
    frames: list[dict[str, object]],
    *args: str,
    advance: int = 5,
    initial_time: int = 1000,
):
    commands = tmp_path / "bin"
    commands.mkdir()
    for command in ("date", "sleep", "curl"):
        executable = commands / command
        executable.write_text(f"#!{sys.executable}\n" + FAKE_COMMAND)
        executable.chmod(0o755)
    (tmp_path / "clock").write_text(str(initial_time))
    (tmp_path / "frames.json").write_text(json.dumps(frames))
    output = tmp_path / "result.json"
    environment = {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "DEMO_TEST_ROOT": str(tmp_path),
        "DEMO_TEST_ADVANCE": str(advance),
        "DEMO_TEST_INITIAL_TIME": str(initial_time),
        "PROMETHEUS_URL": "http://metrics.test:9090",
        "PROMCAST_URL": "http://forecast.test:8787",
        "GRAFANA_URL": "http://dashboard.test:3000",
    }

    async def invoke():
        process = await asyncio.create_subprocess_exec(
            str(SCRIPT),
            "--json",
            str(output),
            *args,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
        return process.returncode, stdout.decode(), stderr.decode()

    code, stdout, stderr = asyncio.run(invoke())
    report = CheckResult.model_validate_json(output.read_text()) if output.exists() else None
    return code, report, stdout, stderr


@pytest.fixture(autouse=True)
def require_jq():
    assert shutil.which("jq"), "demo checks require jq"


def test_smoke_retries_startup_and_requires_firing(tmp_path: Path):
    code, report, stdout, stderr = run_check(
        tmp_path,
        [
            {
                "alerts": "connection refused",
                "runtimeinfo": "connection refused",
                "forecast": "not json",
                "health": {},
            },
            {
                "alerts": alerts("PromcastBreachSoon", state="pending"),
                "forecast": FORECAST,
                "health": {"status": "OK"},
            },
            {"alerts": alerts("PromcastBreachSoon"), "forecast": FORECAST},
        ],
    )
    assert code == 0, stderr
    assert report is not None
    assert report.seconds_to_firing == 10
    assert report.q90_data is True
    assert report.grafana_health is True
    assert report.selfaudit_coverage is True
    assert report.timed_out is False
    assert "promcast_lead_s" not in report.model_fields_set
    assert "passed" in stdout
    requests = (tmp_path / "requests").read_text()
    assert "http://metrics.test:9090/api/v1/status/runtimeinfo" in requests
    assert "http://metrics.test:9090/api/v1/query_range" in requests
    assert "http://forecast.test:8787/api/v1/query_range" in requests
    assert "http://forecast.test:8787/metrics" in requests
    assert "http://dashboard.test:3000/api/datasources/uid/promcast/health" in requests


def test_full_retains_first_firing_after_alert_disappears(tmp_path: Path):
    code, report, _, stderr = run_check(
        tmp_path,
        [
            {"forecast": FORECAST},
            {"alerts": alerts("PromcastBreachSoon"), "forecast": FORECAST},
            {"alerts": alerts("PromcastBreachSoon", "DemoLinearBreachSoon")},
            {"alerts": alerts("DemoDiskFull")},
        ],
        "--full",
    )
    assert code == 0, stderr
    assert report is not None
    assert report.seconds_to_firing == 5
    assert report.promcast_lead_s == 10
    assert report.linear_lead_s == 5
    assert report.first_firing.model_dump() == {
        "PromcastBreachSoon": 1005,
        "DemoLinearBreachSoon": 1010,
        "DemoDiskFull": 1015,
    }


@pytest.mark.parametrize("late", [False, True])
def test_full_uses_stored_firing_times_from_stack_start(tmp_path: Path, late: bool):
    def history(*names: str) -> dict[str, object]:
        times = {"PromcastBreachSoon": 1120, "DemoLinearBreachSoon": 1995, "DemoDiskFull": 3392}
        return {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {"metric": {"alertname": name}, "values": [[times[name], "1"]]}
                    for name in names
                ],
            },
        }

    warnings = ("PromcastBreachSoon", "DemoLinearBreachSoon")
    frames: list[dict[str, object]] = [
        {"history": history(), "forecast": FORECAST},
        {"history": history(*warnings), "forecast": FORECAST},
        {"history": history(*warnings, "DemoDiskFull"), "forecast": FORECAST},
    ]
    if late:
        frames = frames[2:]
    code, report, _, stderr = run_check(
        tmp_path, frames, "--full", initial_time=3400 if late else 1000, advance=1200
    )
    assert code == 0, stderr
    assert report is not None
    assert report.seconds_to_firing == 120
    assert report.promcast_lead_s == 2272
    assert report.linear_lead_s == 1397
    assert report.first_firing.DemoDiskFull == 3392


@pytest.mark.parametrize("history", ["connection refused", "not json", {"status": "error"}])
def test_full_retries_failed_history_queries(tmp_path: Path, history: object):
    code, report, _, stderr = run_check(
        tmp_path,
        [
            {"history": history, "forecast": FORECAST},
            {"alerts": alerts("PromcastBreachSoon", "DemoDiskFull")},
        ],
        "--full",
    )
    assert code == 0, stderr
    assert report is not None
    assert report.seconds_to_firing == 5
    assert report.first_firing.DemoDiskFull == 1005
    assert report.timed_out is False


@pytest.mark.parametrize("warning", [None, "PromcastBreachSoon", "DemoLinearBreachSoon"])
def test_full_has_null_lead_without_an_earlier_warning(tmp_path: Path, warning: str | None):
    names = ("DemoDiskFull",) if warning is None else ("DemoDiskFull", warning)
    code, report, _, stderr = run_check(
        tmp_path, [{"alerts": alerts(*names), "forecast": FORECAST}], "--full"
    )
    assert code == 0, stderr
    assert report is not None
    assert report.promcast_lead_s is None
    assert report.linear_lead_s is None


@pytest.mark.parametrize(
    "forecast",
    [
        {"status": "success", "data": {"resultType": "matrix", "result": []}},
        {"status": "error", "data": FORECAST["data"]},
        {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{"metric": {"target": "demo_disk_free"}, "values": [[1060, "NaN"]]}],
            },
        },
    ],
)
def test_smoke_rejects_missing_or_nonfinite_forecasts(tmp_path: Path, forecast: object):
    code, report, _, stderr = run_check(
        tmp_path,
        [{"alerts": alerts("PromcastBreachSoon"), "forecast": forecast}],
        advance=900,
    )
    assert code == 1
    assert report is not None
    assert report.timed_out is True
    assert report.q90_data is False
    assert "900s" in stderr


def test_smoke_requires_healthy_grafana(tmp_path: Path):
    code, report, _, stderr = run_check(
        tmp_path,
        [{"alerts": alerts("PromcastBreachSoon"), "forecast": FORECAST, "health": {}}],
        advance=900,
    )
    assert code == 1
    assert report is not None
    assert report.grafana_health is False
    assert "900s" in stderr


def test_full_times_out_without_disk_full(tmp_path: Path):
    code, report, _, stderr = run_check(
        tmp_path,
        [{"alerts": alerts("PromcastBreachSoon"), "forecast": FORECAST}],
        "--full",
        advance=3600,
    )
    assert code == 1
    assert report is not None
    assert report.timed_out is True
    assert report.first_firing.DemoDiskFull is None
    assert report.promcast_lead_s is None
    assert "3600s" in stderr


@pytest.mark.parametrize("args", [("--unknown",), ("--json",), ("--json", "")])
def test_invalid_arguments(tmp_path: Path, args: tuple[str, ...]):
    code, report, _, stderr = run_check(tmp_path, [], *args)
    assert code == 2
    assert report is None
    assert "Usage:" in stderr


@pytest.mark.parametrize(
    "metrics",
    [
        "connection refused",
        "",
        "# HELP promcast_selfaudit_coverage_ratio Coverage.",
        'promcast_selfaudit_coverage_ratio{target="demo_disk_free",method="toto"} NaN',
        'promcast_selfaudit_coverage_ratio{target="demo_disk_free",method="toto"} 1.1',
        'promcast_selfaudit_coverage_ratio{target="other",method="toto"} 0.8',
    ],
)
def test_smoke_requires_populated_coverage(tmp_path: Path, metrics: str):
    code, report, _, stderr = run_check(
        tmp_path,
        [{"alerts": alerts("PromcastBreachSoon"), "forecast": FORECAST, "metrics": metrics}],
        advance=900,
    )
    assert code == 1
    assert report is not None
    assert report.selfaudit_coverage is False
    assert "900s" in stderr


def test_smoke_waits_for_first_mature_forecast(tmp_path: Path):
    code, report, _, stderr = run_check(
        tmp_path,
        [
            {"alerts": alerts("PromcastBreachSoon"), "forecast": FORECAST, "metrics": ""},
            {"alerts": alerts("PromcastBreachSoon"), "forecast": FORECAST},
        ],
        advance=600,
    )
    assert code == 0, stderr
    assert report is not None
    assert report.selfaudit_coverage is True


PLAYWRIGHT_COMMAND = """
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["DEMO_TEST_ROOT"])
with (root / "calls").open("a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
args = sys.argv[1:]
if "screenshot" in args:
    Path(args[args.index("--filename") + 1]).write_bytes(b"png")
    sys.exit(int(os.environ.get("DEMO_TEST_SHOT_EXIT", "0")))
"""


def run_screenshot(tmp_path: Path, shot_exit: int = 0):
    commands = tmp_path / "bin"
    commands.mkdir()
    executable = commands / "playwright-cli"
    executable.write_text(f"#!{sys.executable}\n" + PLAYWRIGHT_COMMAND)
    executable.chmod(0o755)
    target = tmp_path / "panel.png"

    async def invoke():
        process = await asyncio.create_subprocess_exec(
            str(SCRIPT),
            "--screenshot",
            str(target),
            env={
                **os.environ,
                "PATH": str(commands) + os.pathsep + os.environ["PATH"],
                "DEMO_TEST_ROOT": str(tmp_path),
                "DEMO_TEST_SHOT_EXIT": str(shot_exit),
                "GRAFANA_URL": "http://dashboard.test:3000",
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
        return process.returncode, stdout.decode(), stderr.decode()

    code, stdout, stderr = asyncio.run(invoke())
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    return code, stdout, stderr, calls, target


def test_screenshot_opens_the_dashboard_and_closes_only_its_session(tmp_path: Path):
    code, stdout, stderr, calls, target = run_screenshot(tmp_path)
    assert code == 0, stderr
    assert target.read_bytes() == b"png"
    assert f"Wrote {target}" in stdout
    session = calls[0][0]
    assert session.startswith("-s=promcast-demo-")
    assert all(call[0] == session for call in calls)
    assert calls[0][1:] == ["open", "http://dashboard.test:3000/d/promcast-demo?orgId=1&kiosk"]
    assert [call[1] for call in calls] == ["open", "resize", "eval", "screenshot", "close"]


def test_screenshot_closes_its_session_when_capture_fails(tmp_path: Path):
    code, _, _, calls, _ = run_screenshot(tmp_path, shot_exit=3)
    assert code != 0
    assert calls[-1][1] == "close"


@pytest.mark.parametrize(
    "args", [("--screenshot",), ("--screenshot", ""), ("--screenshot", "a", "--full")]
)
def test_screenshot_rejects_bad_arguments(tmp_path: Path, args: tuple[str, ...]):
    code, report, _, stderr = run_check(tmp_path, [], *args)
    assert code == 2
    assert report is None
    assert "Usage:" in stderr
