import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from promcast.core.forecaster import FakeForecaster
from promcast.shell import cli, model_download
from promcast.shell.errors import ModelFilesMissingError


@pytest.fixture
def model_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    source = tmp_path / "source"
    source.mkdir()
    weights, config = b"fixture weights", b"{}"
    for filename, content in (("model.safetensors", weights), ("config.json", config)):
        (source / filename).write_bytes(content)
    artifact = replace(
        model_download.TOTO_MODELS[0],
        sha256=hashlib.sha256(weights).hexdigest(),
        config_sha256=hashlib.sha256(config).hexdigest(),
    )
    monkeypatch.setattr(model_download, "TOTO_MODELS", (artifact,))
    calls: list[str] = []

    def download(
        repo_id: str, filename: str, *, revision: str, local_dir: Path, token: bool
    ) -> str:
        assert repo_id == artifact.repository
        assert revision == artifact.revision
        assert token is False
        calls.append(filename)
        path = local_dir / filename
        path.write_bytes((source / filename).read_bytes())
        return str(path)

    def select(model_dir: Path, name: str, batch_size: int, threads: int | None) -> FakeForecaster:
        for filename in ("config.json", "model.safetensors"):
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(str(model_dir / filename))
        return FakeForecaster()

    monkeypatch.setattr(model_download.huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(cli, "_select_forecaster", select)
    return source, calls


def test_empty_cache_downloads_once_from_pinned_source(
    tmp_path: Path, model_source: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str]
):
    _, calls = model_source
    directory = tmp_path / "cache/toto-2.0-22m"
    assert cli.load_forecaster(directory, "toto-2.0-22m").name == "fake"
    assert calls == ["model.safetensors", "config.json"]
    assert (directory / "config.json").read_bytes() == b"{}"
    message = capsys.readouterr().err
    assert message.count("Downloading") == 1
    assert "Datadog/Toto-2.0-22m" in message
    assert str(directory) in message
    cli.load_forecaster(directory, "toto-2.0-22m")
    assert len(calls) == 2
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("filename", ["model.safetensors", "config.json"])
def test_first_use_rejects_bad_checksum(
    tmp_path: Path, model_source: tuple[Path, list[str]], filename: str
):
    source, _ = model_source
    (source / filename).write_bytes(b"corrupt")
    with pytest.raises(ModelFilesMissingError, match="sha256"):
        cli.load_forecaster(tmp_path / "cache/toto-2.0-22m", "toto-2.0-22m")


def test_first_use_offline_has_actionable_error(
    tmp_path: Path, model_source: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
):
    def offline(*_args: object, **_kwargs: object) -> str:
        raise OSError("offline")

    monkeypatch.setattr(model_download.huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(ModelFilesMissingError) as error:
        cli.load_forecaster(tmp_path / "cache/toto-2.0-22m", "toto-2.0-22m")
    assert "offline" in error.value.what
    assert "network" in error.value.next_step
    assert "PROMCAST_CACHE_DIR" in error.value.next_step
