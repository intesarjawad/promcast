import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from promcast.core.forecaster import FakeForecaster
from promcast.core.series import FloatArray, Series
from promcast.shell import bench, bench_case, cli


@pytest.mark.parametrize("budget", [60, 120])
@pytest.mark.parametrize(
    ("offset", "expected"),
    [(-0.01, bench.MODELS[0]), (0, bench.MODELS[1]), (0.01, bench.MODELS[1])],
)
def test_fallback_uses_full_context_median(budget: float, offset: float, expected: str) -> None:
    measurements = [
        bench.Measurement(bench.MODELS[0], 512, [1, 1, 1], 1, 100, 2),
        bench.Measurement(
            bench.MODELS[0], 1472, [1, budget + offset, 1000], budget + offset, 100, 2
        ),
    ]
    assert bench.recommended(measurements, budget) == expected


def test_measure_times_complete_cycles_and_checks_quantiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[int, int, int]] = []

    class RecordingForecaster(FakeForecaster):
        def __init__(self, path: Path, name: str) -> None:
            assert name == bench.MODELS[0]
            assert path == tmp_path / bench.MODELS[0]

        def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
            calls.append((len(batch), batch[0].values.size, horizon_steps))
            return super().forecast(batch, horizon_steps)

    monkeypatch.setattr(bench_case, "Toto2Forecaster", RecordingForecaster)
    ticks = iter([0, 4, 10, 12, 20, 23])
    monkeypatch.setattr(bench_case, "perf_counter", lambda: next(ticks))
    measurement = bench_case.measure(bench.MODELS[0], 1472, tmp_path)
    assert calls == [(50, 1472, 120)] * 3
    assert measurement.seconds_per_cycle == [4, 2, 3]
    assert measurement.median_seconds_per_cycle == 3
    assert measurement.peak_rss_bytes > 0
    assert measurement.torch_threads > 0


def test_measure_rejects_invalid_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class InvalidForecaster:
        def __init__(self, path: Path, name: str) -> None:
            assert path == tmp_path / name

        def forecast(self, _batch: list[Series], _horizon_steps: int) -> list[FloatArray]:
            return []

    monkeypatch.setattr(bench_case, "Toto2Forecaster", InvalidForecaster)
    with pytest.raises(ValueError, match="invalid benchmark quantiles"):
        bench_case.measure(bench.MODELS[0], 512, tmp_path)


def test_bench_cli_runs_all_cases_in_separate_processes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []

    class Process:
        returncode = 0

        def __init__(self, model: str, context: int) -> None:
            self.measurement = bench.Measurement(model, context, [1, 2, 3], 2, 123456, 4)

        async def communicate(self) -> tuple[bytes, bytes]:
            return json.dumps(asdict(self.measurement)).encode(), b""

    async def spawn(*arguments: str, **options: object) -> Process:
        calls.append(arguments)
        assert arguments[:3] == (sys.executable, "-m", "promcast.shell.bench_case")
        assert arguments[-2:] == ("--cache-dir", str(tmp_path))
        return Process(arguments[4], int(arguments[6]))

    monkeypatch.setattr(bench.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(bench, "cpu_model", lambda: "Test CPU")
    monkeypatch.setattr(sys, "argv", ["promcast", "bench", "--json", "--cache-dir", str(tmp_path)])
    assert cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert len(calls) == 4
    assert {(item["model"], item["context_steps"]) for item in report["measurements"]} == {
        (model, context) for model in bench.MODELS for context in bench.CONTEXTS
    }
    assert report["recommended"] == bench.MODELS[0]
    assert report["cpu_model"] == "Test CPU"
    assert (report["series_count"], report["horizon_steps"], report["repeats"]) == (50, 120, 3)


def test_child_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailedProcess:
        returncode = 1

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b"local Toto model requires config.json"

    async def spawn(*arguments: str, **options: object) -> FailedProcess:
        return FailedProcess()

    monkeypatch.setattr(bench.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(bench, "cpu_model", lambda: "Test CPU")
    with pytest.raises(ValueError, match=r"toto-2.0-22m, context 512:.*config.json"):
        asyncio.run(bench.benchmark(Path("/missing")))
