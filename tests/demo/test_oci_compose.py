import asyncio
import importlib.util
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import oci_compose
else:
    spec = importlib.util.spec_from_file_location("oci_compose", ROOT / "scripts/oci_compose.py")
    assert spec is not None
    assert spec.loader is not None
    oci_compose = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oci_compose)

generate = oci_compose.generate


def test_generated_artifact_is_current_and_has_no_interpolation():
    content = (ROOT / "deploy/demo/compose.yaml").read_text()
    assert content == generate(ROOT)
    assert re.search(r"(?<!\$)(?:\$\$)*\$(?!\$)", content) is None
    document = yaml.safe_load(content)
    tag = oci_compose.image_tag(oci_compose.package_version(ROOT / "pyproject.toml"))
    assert document["services"]["promcast"]["image"] == f"ghcr.io/intesarjawad/promcast:{tag}"
    original = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert document["volumes"] == original["volumes"]
    for name, service in document["services"].items():
        assert "build" not in service
        assert service.get("depends_on") == original["services"][name].get("depends_on")
        assert service.get("ports") == original["services"][name].get("ports")
        assert all(
            mount.split(":")[0] in document["volumes"] for mount in service.get("volumes", [])
        )
        configs = {entry["target"]: entry["source"] for entry in service.get("configs", [])}
        for mount in original["services"][name].get("volumes", []):
            source, target, *_ = mount.split(":")
            if source.startswith("."):
                assert (
                    document["configs"][configs[target]]["content"].replace("$$", "$")
                    == (ROOT / source).read_text()
                )
    for name, config in original["configs"].items():
        assert document["configs"][name]["content"] == config["content"].replace("$", "$$")


def fixture_root(tmp_path: Path, mount: str = "./input.txt:/input:ro") -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    (tmp_path / "input.txt").write_text("$name ${value} $$ ${PROMCAST_VERSION:-0.1.0}\n")
    (tmp_path / "docker-compose.yml").write_text(
        yaml.safe_dump(
            {
                "services": {
                    "example": {
                        "image": "example:${PROMCAST_VERSION:-9.9.9}",
                        "build": ".",
                        "volumes": [mount],
                    }
                },
                "configs": {"original": {"content": "value: $variable\n"}},
            }
        )
    )
    return tmp_path


def test_dollars_are_escaped_without_expanding_embedded_content(tmp_path: Path):
    document = yaml.safe_load(generate(fixture_root(tmp_path)))
    assert document["services"]["example"]["image"] == "example:1.2.3"


def test_release_candidate_image_uses_the_semver_tag(tmp_path: Path):
    root = fixture_root(tmp_path)
    (root / "pyproject.toml").write_text('[project]\nversion = "1.2.3rc1"\n')
    document = yaml.safe_load(generate(root))
    assert document["services"]["example"]["image"] == "example:1.2.3-rc.1"
    assert document["configs"]["inline-file-1"]["content"] == (
        "$$name $${value} $$$$ $${PROMCAST_VERSION:-0.1.0}\n"
    )
    assert document["configs"]["original"]["content"] == "value: $$variable\n"


@pytest.mark.parametrize("mount", ["./input.txt:/input", "./:/input:ro", "../:/input:ro"])
def test_rejects_mounts_that_cannot_be_inlined(tmp_path: Path, mount: str):
    with pytest.raises(ValueError, match="single-file mount"):
        generate(fixture_root(tmp_path, mount))


def test_generator_check_command():
    async def invoke():
        process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "python",
            str(ROOT / "scripts/oci_compose.py"),
            "--check",
            stdin=asyncio.subprocess.DEVNULL,
        )
        return await process.wait()

    assert asyncio.run(invoke()) == 0


@pytest.mark.parametrize("existing", [False, True])
def test_check_rejects_missing_or_stale_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
):
    fixture_root(tmp_path)
    destination = tmp_path / "deploy/demo/compose.yaml"
    if existing:
        destination.parent.mkdir(parents=True)
        destination.write_text("stale\n")
    monkeypatch.setattr(oci_compose, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["oci_compose.py", "--check"])
    with pytest.raises(SystemExit, match="1"):
        oci_compose.main()
    if existing:
        assert destination.read_text() == "stale\n"
    else:
        assert not destination.exists()
