import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import check_version
else:
    spec = importlib.util.spec_from_file_location(
        "check_version", ROOT / "scripts/check_version.py"
    )
    assert spec is not None
    assert spec.loader is not None
    check_version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_version)


def _pyproject(tmp_path: Path, version: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(f'[project]\nname = "promcast"\nversion = "{version}"\n')
    return path


@pytest.mark.parametrize(
    ("tag", "version"),
    [("v0.1.0-rc.1", "0.1.0rc1"), ("v0.1.0rc1", "0.1.0rc1"), ("v0.1.0", "0.1.0")],
)
def test_matching_tag_passes(tmp_path: Path, tag: str, version: str):
    assert check_version.main([tag, "--pyproject", str(_pyproject(tmp_path, version))]) == 0


@pytest.mark.parametrize(
    ("tag", "message"),
    [
        ("v0.1.0-rc.2", "set the pyproject version to 0.1.0rc2 or tag v0.1.0rc1"),
        ("v0.1.0", "set the pyproject version to 0.1.0 or tag v0.1.0rc1"),
        ("0.1.0-rc.1", "must start with 'v'"),
        ("vnext", "is not a PEP 440 version; tag v0.1.0rc1 instead"),
    ],
)
def test_mismatched_tag_fails_with_next_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], tag: str, message: str
):
    assert check_version.main([tag, "--pyproject", str(_pyproject(tmp_path, "0.1.0rc1"))]) == 1
    assert message in capsys.readouterr().err


def test_repository_version_has_a_tag():
    with (ROOT / "pyproject.toml").open() as stream:
        version = next(line for line in stream if line.startswith("version = ")).split('"')[1]
    assert check_version.mismatch(f"v{version}", ROOT / "pyproject.toml") is None
