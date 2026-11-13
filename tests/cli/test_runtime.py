import asyncio
import hashlib
import json
import sys
import threading
from pathlib import Path

import pytest

from promcast.core.config import Config
from promcast.core.forecaster import FakeForecaster, Forecaster
from promcast.shell.cli import main


def local_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    text = Path("promcast.example.yaml").read_text()
    path.write_text(text.replace("~/.cache/promcast/models", str(tmp_path)))
    return path


def test_doctor_redacts_secrets_and_reports_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = local_config(tmp_path)
    path.write_text(
        path.read_text().replace("headers: {}", "headers: {Authorization: secret-value}")
    )
    monkeypatch.setattr(sys, "argv", ["promcast", "doctor", "--config", str(path)])
    assert main() == 1
    output = capsys.readouterr()
    assert "secret-value" not in output.out + output.err
    assert "<redacted>" in output.out
    report = json.loads(output.out)
    assert report["model_files"] == {"config.json": False, "model.safetensors": False}


def test_serve_loads_selected_local_model_off_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = local_config(tmp_path)
    model_dir = tmp_path / "toto-2.0-22m"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"fixture")
    calls: list[str] = []
    main_thread = threading.get_ident()

    def load(directory: Path, name: str, batch_size: int) -> Forecaster:
        assert directory == model_dir
        assert name == "toto-2.0-22m"
        assert batch_size == 16
        assert threading.get_ident() != main_thread
        calls.append("load")
        return FakeForecaster()

    async def serve(config: Config, forecaster: Forecaster, model_sha: str) -> None:
        assert asyncio.get_running_loop().is_running()
        assert config.targets[0].name == "root_disk_free"
        assert forecaster.name == "fake"
        assert model_sha == hashlib.sha256(b"fixture").hexdigest()
        calls.append("serve")

    monkeypatch.setattr("promcast.shell.cli.load_forecaster", load)
    monkeypatch.setattr("promcast.shell.server.serve", serve)
    monkeypatch.setattr(sys, "argv", ["promcast", "serve", "--config", str(path)])
    assert main() == 0
    assert calls == ["load", "serve"]


@pytest.mark.parametrize("command", ["bench", "replay"])
def test_command_prerequisites_are_reported(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "argv", ["promcast", command])
    if command == "bench":
        monkeypatch.setattr(sys, "argv", ["promcast", "bench", "--cache-dir", str(tmp_path)])
        assert main() == 1
        output = capsys.readouterr()
        assert output.out == ""
        assert "benchmark failed for toto-2.0-22m, context 512" in output.err
        assert f"local Toto model requires {tmp_path / 'toto-2.0-22m/config.json'}" in output.err
    else:
        with pytest.raises(SystemExit, match="2"):
            main()
        output = capsys.readouterr()
        assert output.out == ""
        assert "the following arguments are required: --source, --out" in output.err
