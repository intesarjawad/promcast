import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
DOWNLOAD_HARNESS = """
import hashlib
import json
import os
import runpy
import sys
from dataclasses import asdict, replace
from pathlib import Path

import huggingface_hub
from promcast.shell import model_download

weights = b"test model weights"
config = b'{"patch_size": 32}'
contents = {"model.safetensors": weights, "config.json": config}
model_download.TOTO_MODELS = tuple(
    replace(
        model,
        sha256=hashlib.sha256(weights).hexdigest(),
        config_sha256=hashlib.sha256(config).hexdigest(),
    )
    for model in model_download.TOTO_MODELS
)
Path("expected.json").write_text(
    json.dumps([asdict(model) for model in model_download.TOTO_MODELS])
)
model_download.MODELS_DIRECTORY = Path.cwd() / "models"
calls = []

def download(repo_id, filename, *, revision, local_dir, token):
    calls.append(dict(repo_id=repo_id, filename=filename, revision=revision, token=token))
    Path("downloads.json").write_text(json.dumps(calls))
    local_dir.mkdir(parents=True, exist_ok=True)
    content = contents[filename]
    if os.environ.get("CORRUPT_FILE") == filename:
        content = b"corrupt"
    path = local_dir / filename
    path.write_bytes(content)
    return str(path)

huggingface_hub.hf_hub_download = download
script = sys.argv[1]
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
"""


def test_image_installs_locked_runtime_and_runs_without_uv():
    logical = DOCKERFILE.replace("\\\n", "")
    assert "FROM python:3.12-slim AS base" in logical
    assert "ghcr.io/astral-sh/uv:0.13.7" in logical
    sync = next(line for line in logical.splitlines() if line.startswith("RUN uv sync "))
    assert {"--frozen", "--no-dev", "--no-editable"} <= set(shlex.split(sync))
    assert "--extra" not in sync
    assert "--all-extras" not in sync
    runtime = logical.split("FROM base AS runtime\n")[1]
    assert "HF_HUB_OFFLINE=1" in runtime
    assert "HF_HUB_DISABLE_TELEMETRY=1" in runtime
    assert "PYTHONDONTWRITEBYTECODE=1" in runtime
    assert "MPLCONFIGDIR=/tmp/matplotlib" in runtime
    assert 'VOLUME ["/tmp"]' in runtime
    assert "COPY --from=build /opt/promcast/.venv /opt/promcast/.venv" in runtime
    assert "COPY --from=build /opt/promcast/models /opt/promcast/models" in runtime
    assert "COPY licences/ /opt/promcast/licences/" in runtime
    assert 'for package in ("numpy", "torch", "augurs"):' in runtime
    assert runtime.index("USER 65532") < runtime.index("RUN --network=none python")
    assert "EXPOSE 8787" in runtime
    assert 'ENTRYPOINT ["promcast"]' in runtime
    assert 'CMD ["serve", "--config", "/etc/promcast.yaml"]' in runtime


def test_linux_torch_is_cpu_only_for_both_image_architectures():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    torch = next(
        package
        for package in lock["package"]
        if package["name"] == "torch" and package["version"].endswith("+cpu")
    )
    assert torch["source"]["registry"] == "https://download.pytorch.org/whl/cpu"
    wheels = [wheel["url"] for wheel in torch["wheels"] if "cp312-cp312" in wheel["url"]]
    assert any("aarch64" in wheel for wheel in wheels)
    assert any("x86_64" in wheel for wheel in wheels)


def test_context_only_includes_build_inputs():
    patterns = (ROOT / ".dockerignore").read_text().splitlines()
    assert patterns[0] == "**"
    included = {pattern.removeprefix("!") for pattern in patterns if pattern.startswith("!")}
    assert included == {
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "src/",
        "src/**",
        "licences/",
        "licences/**",
    }
    assert "**/__pycache__/" in patterns
    assert "**/*.py[cod]" in patterns
    assert "COPY scripts/" not in DOCKERFILE
    assert not (ROOT / "src/promcast/shell/model_download.py").is_symlink()


@pytest.mark.parametrize("corrupt", [None, "model.safetensors", "config.json"])
def test_image_fetches_and_verifies_weights_and_configs(tmp_path: Path, corrupt: str | None):
    match = re.search(r"RUN \.venv/bin/python - <<'PY'\n(.*?)\nPY", DOCKERFILE, re.DOTALL)
    assert match is not None
    script = tmp_path / "fetch.py"
    script.write_text(match[1])
    harness = tmp_path / "harness.py"
    harness.write_text(DOWNLOAD_HARNESS)

    async def invoke():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(harness),
            str(script),
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CORRUPT_FILE": corrupt or ""},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return process.returncode, (stdout + stderr).decode()

    code, output = asyncio.run(invoke())
    calls = json.loads((tmp_path / "downloads.json").read_text())
    if corrupt:
        assert code != 0
        assert f"file {tmp_path / 'models'}" in output
        assert f"/{corrupt} has sha256" in output
        fetched = [call["filename"] for call in calls]
        assert fetched == ["model.safetensors", "config.json"][: fetched.index(corrupt) + 1]
    else:
        assert code == 0, output
        models = json.loads((tmp_path / "expected.json").read_text())
        for model in models:
            assert {
                call["filename"] for call in calls if call["repo_id"] == model["repository"]
            } == {
                "config.json",
                "model.safetensors",
            }
            assert all(
                call["revision"] == model["revision"] and call["token"] is False
                for call in calls
                if call["repo_id"] == model["repository"]
            )
            for filename in ("config.json", "model.safetensors"):
                assert (tmp_path / "models" / model["name"] / filename).is_file()


def test_model_cache_keys_hash_the_packaged_digests():
    for name in ("ci.yml", "evidence.yml"):
        workflow = (ROOT / ".github/workflows" / name).read_text()
        keys = [line for line in workflow.splitlines() if "key: models" in line]
        assert keys
        assert all("hashFiles('src/promcast/shell/model_download.py')" in line for line in keys)
        assert "scripts/fetch_models.py" not in workflow


def test_repository_setup_uses_the_packaged_downloader(tmp_path: Path):
    harness = tmp_path / "harness.py"
    harness.write_text(DOWNLOAD_HARNESS)
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(harness), str(ROOT / "scripts/fetch_models.py")],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = json.loads((tmp_path / "downloads.json").read_text())
    assert len(calls) == 4
