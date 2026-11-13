import importlib.util
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import image_tag
else:
    spec = importlib.util.spec_from_file_location("image_tag", ROOT / "scripts/image_tag.py")
    assert spec is not None
    assert spec.loader is not None
    image_tag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_tag)

TAG = image_tag.image_tag(image_tag.package_version(ROOT / "pyproject.toml"))


@pytest.mark.parametrize(
    ("version", "tag"),
    [
        ("0.1.0", "0.1.0"),
        ("0.1.0rc1", "0.1.0-rc.1"),
        ("1.2.3a4", "1.2.3-a.4"),
        ("2.0.0b10", "2.0.0-b.10"),
    ],
)
def test_package_version_maps_to_semver_tag(version: str, tag: str):
    assert image_tag.image_tag(version) == tag


@pytest.mark.parametrize(
    "version", ["0.1.0.dev1", "0.1.0.post1", "0.1.0+local", "1.0", "v0.1.0", "0.1.0-rc.1", ""]
)
def test_versions_without_a_tag_are_refused(version: str):
    with pytest.raises(ValueError, match=r"use X\.Y\.Z or X\.Y\.Z"):
        image_tag.image_tag(version)


def test_main_prints_the_tag_of_the_declared_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.2.0rc3"\n')
    assert image_tag.main(["--pyproject", str(pyproject)]) == 0
    assert capsys.readouterr().out == "0.2.0-rc.3\n"


def test_main_explains_a_version_without_a_tag(capsys: pytest.CaptureFixture[str]):
    assert image_tag.main(["0.1.0.dev1"]) == 1
    assert "use X.Y.Z or X.Y.Z{a,b,rc}N in pyproject.toml" in capsys.readouterr().err


def test_chart_versions_match_the_image_tag():
    chart = yaml.safe_load((ROOT / "deploy/helm/promcast/Chart.yaml").read_text())
    assert (str(chart["version"]), str(chart["appVersion"])) == (TAG, TAG)


@pytest.mark.parametrize(
    "compose",
    ["docker-compose.yml", "deploy/compose/victoriametrics.yml", "deploy/compose/mimir.yml"],
)
def test_compose_defaults_name_the_image_tag(compose: str):
    defaults = re.findall(
        r"ghcr\.io/intesarjawad/promcast:\$\{PROMCAST_VERSION:-([^}]*)\}",
        (ROOT / compose).read_text(),
    )
    assert defaults == [TAG]


def test_release_pushes_only_the_derived_tag_and_latest():
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    names = next(step for step in workflow["jobs"]["version"]["steps"] if step.get("id") == "names")
    assert "tag=$(python3 scripts/image_tag.py)" in names["run"]
    tags = next(step for step in workflow["jobs"]["image"]["steps"] if step.get("id") == "tags")
    assert re.findall(r'echo "(\$IMAGE:[^"]*)"', tags["run"]) == ["$IMAGE:$TAG", "$IMAGE:latest"]
