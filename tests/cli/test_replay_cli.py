import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from promcast.core.forecaster import FakeForecaster
from promcast.core.replay import ReplaySpec
from promcast.shell import cli, sources


@pytest.fixture
def small_synthetic(monkeypatch: pytest.MonkeyPatch) -> sources.ReplaySource:
    source = sources.ReplaySource(
        100 - np.maximum(np.arange(200, dtype=np.float64) - 100, 0) ** 2,
        0,
        ReplaySpec("below", 64, 10, 60, 5, 0, threshold=0),
        ("Synthetic test scenario.",),
    )

    def load_synthetic(_name: str) -> sources.ReplaySource:
        return source

    monkeypatch.setattr(sources, "synthetic", load_synthetic)
    return source


def run_replay(monkeypatch: pytest.MonkeyPatch, out: Path, *flags: str) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promcast",
            "replay",
            "--source",
            "synthetic",
            "--scenario",
            "accelerating_leak",
            "--out",
            str(out),
            *flags,
        ],
    )
    return cli.main()


def test_replay_writes_deterministic_reports_and_headless_png(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    small_synthetic: sources.ReplaySource,
    capsys: pytest.CaptureFixture[str],
):
    first, second = tmp_path / "first", tmp_path / "second"
    assert run_replay(monkeypatch, first) == 0
    assert run_replay(monkeypatch, second) == 0
    for name in ("replay.md", "replay.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first / "replay.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "stars mark calibration-selected points" in (first / "replay.md").read_text()
    assert (first / "replay.md").read_text().startswith("Replay-only calibration winner: ets\n")
    assert not (first / "recommended.yaml").exists()
    document = json.loads((first / "replay.json").read_text())
    assert set(document["methods"]) == {"linear", "ets", "seasonal-naive", "static-margin"}
    for name, method in document["methods"].items():
        selected = next(
            item for item in document["test_sweep"][name] if item["point"] == method["point"]
        )
        assert selected["score"]["leads_s"] == method["test"]["leads_s"]
        assert len(document["test_sweep"][name]) == len(method["sweep"])
    assert "Foundation models not requested" in capsys.readouterr().out


def test_recommendation_is_valid_fragment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    small_synthetic: sources.ReplaySource,
    capsys: pytest.CaptureFixture[str],
):
    # The source horizon must permit the pinned demo target's linear windows.
    def load_synthetic(_name: str) -> sources.ReplaySource:
        return replace(small_synthetic, spec=replace(small_synthetic.spec, horizon_steps=120))

    monkeypatch.setattr(sources, "synthetic", load_synthetic)
    assert (
        run_replay(
            monkeypatch, tmp_path, "--config", "demo/promcast.yaml", "--target", "demo_disk_free"
        )
        == 0
    )
    monkeypatch.setattr(
        sys, "argv", ["promcast", "validate", str(tmp_path / "recommended.yaml"), "--fragment"]
    )
    assert cli.main() == 0
    assert capsys.readouterr().out.endswith("ok\n")


def test_explicit_models_are_loaded_and_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, small_synthetic: sources.ReplaySource
):
    loaded: list[tuple[Path, str]] = []

    def load(path: Path, name: str) -> FakeForecaster:
        loaded.append((path, name))
        return FakeForecaster()

    monkeypatch.setattr(cli, "load_forecaster", load)
    assert run_replay(monkeypatch, tmp_path, "--models", "toto-2.0-4m") == 0
    assert loaded[0][1] == "toto-2.0-4m"
    assert "toto-2.0-4m" in json.loads((tmp_path / "replay.json").read_text())["methods"]


def test_labels_write_secondary_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, small_synthetic: sources.ReplaySource
):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["1970-01-01T01:50:00Z"]))
    assert run_replay(monkeypatch, tmp_path, "--incidents", str(labels)) == 0
    text = (tmp_path / "replay.md").read_text()
    assert "Secondary labelled-anomaly scores" in text
    document = json.loads((tmp_path / "replay.json").read_text())
    assert document["labelled_incidents"]["methods"]["linear"]["test"]["n_incidents"] == 1


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (("--budget", "-1"), ("budget must be finite", 1)),
        (("--models", "unknown"), ("unknown model", 2)),
        (("--config", "demo/promcast.yaml", "--target", "unknown"), ("--target must name", 2)),
    ],
)
@pytest.mark.usefixtures("small_synthetic")
def test_bad_replay_inputs_fail_without_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flags: tuple[str, ...],
    expected: tuple[str, int],
):
    message, status = expected
    assert run_replay(monkeypatch, tmp_path, *flags) == status
    assert message in capsys.readouterr().err
    assert not (tmp_path / "replay.json").exists()


def test_counter_scenario_is_refused_as_a_user_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert run_replay(monkeypatch, tmp_path, "--scenario", "counter") == 2
    err = capsys.readouterr().err
    assert "counter is a counter, not a gauge" in err
    assert "promcast replay --source synthetic --scenario accelerating_leak" in err
    assert not (tmp_path / "replay.json").exists()
