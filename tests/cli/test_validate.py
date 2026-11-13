import sys
from pathlib import Path

import pytest

from promcast.shell.cli import main

EXAMPLE_PATH = Path(__file__).parents[2] / "promcast.example.yaml"


def test_validate_example(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["promcast", "validate", str(EXAMPLE_PATH)])
    assert main() == 0
    assert capsys.readouterr() == ("ok\n", "")


def test_validate_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fragment = tmp_path / "recommended.yaml"
    example = EXAMPLE_PATH.read_text(encoding="utf-8")
    fragment.write_text("targets:" + example.split("targets:", 1)[1], encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["promcast", "validate", str(fragment), "--fragment"])
    assert main() == 0
    assert capsys.readouterr() == ("ok\n", "")
    monkeypatch.setattr(sys, "argv", ["promcast", "validate", str(fragment)])
    assert main() == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "tsdb" in output.err


@pytest.mark.parametrize("fragment", [False, True])
def test_validate_reports_field_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fragment: bool,
) -> None:
    text = EXAMPLE_PATH.read_text(encoding="utf-8").replace("quantile: 0.9", "quantile: 0.95")
    if fragment:
        text = "targets:" + text.split("targets:", 1)[1]
    config = tmp_path / "invalid.yaml"
    config.write_text(text, encoding="utf-8")
    arguments = ["promcast", "validate", str(config)]
    if fragment:
        arguments.append("--fragment")
    monkeypatch.setattr(sys, "argv", arguments)
    assert main() == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "targets.0.quantile" in output.err


def test_validate_uses_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "environment.yaml"
    config.write_text(
        EXAMPLE_PATH.read_text(encoding="utf-8").replace(
            "http://prometheus:9090", "${PROMCAST_TEST_URL}"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMCAST_TEST_URL", "http://localhost:9090")
    monkeypatch.setattr(sys, "argv", ["promcast", "validate", str(config)])
    assert main() == 0
    assert capsys.readouterr() == ("ok\n", "")
    monkeypatch.delenv("PROMCAST_TEST_URL")
    assert main() == 2
    assert "PROMCAST_TEST_URL" in capsys.readouterr().err


@pytest.mark.parametrize("contents", [None, b"\xff", b"headers: {secret: private-value"])
def test_validate_file_errors_are_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contents: bytes | None,
) -> None:
    config = tmp_path / "broken.yaml"
    if contents is not None:
        config.write_bytes(contents)
    monkeypatch.setattr(sys, "argv", ["promcast", "validate", str(config)])
    assert main() == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err
    assert "private-value" not in output.err
    assert "Traceback" not in output.err
