import asyncio
import re
import shutil
import tomllib
from pathlib import Path
from typing import cast

import pytest
import yaml
from pydantic import BaseModel

from promcast.core.config import parse_config

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy/helm/promcast"


class ChartMetadata(BaseModel):
    apiVersion: str  # noqa: N815
    name: str
    version: str
    appVersion: str  # noqa: N815
    type: str


def load(path: Path) -> dict[str, object]:
    document: object = yaml.safe_load(path.read_text())
    assert isinstance(document, dict)
    return cast("dict[str, object]", document)


async def helm(*arguments: str) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        "helm",
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    assert process.returncode is not None
    return process.returncode, stdout.decode(), stderr.decode()


def test_chart_versions_follow_the_package():
    metadata = ChartMetadata.model_validate(load(CHART / "Chart.yaml"))
    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    assert metadata.apiVersion == "v2"
    assert metadata.name == "promcast"
    assert metadata.type == "application"
    assert metadata.version == version
    assert metadata.appVersion == version


def test_default_image_is_the_published_repository():
    image = load(CHART / "values.yaml")["image"]
    assert isinstance(image, dict)
    assert image["repository"] == "ghcr.io/intesarjawad/promcast"
    assert image["tag"] == ""


def test_default_config_is_a_valid_promcast_document():
    config = load(CHART / "values.yaml")["config"]
    parsed = parse_config(yaml.safe_dump(config), environment={})
    assert str(parsed.tsdb.url) == "http://prometheus:9090/"
    assert parsed.server.listen == "0.0.0.0:8787"
    assert str(parsed.model.cache_dir) == "/opt/promcast/models"
    assert [target.name for target in parsed.targets] == ["root_disk_free"]
    assert parsed.targets[0].threshold == 5e9


def test_templates_only_read_declared_values():
    declared = set(load(CHART / "values.yaml"))
    used = {
        match
        for template in (CHART / "templates").iterdir()
        for match in re.findall(r"\.Values\.([A-Za-z]+)", template.read_text())
    }
    assert used <= declared
    assert declared <= used | {"config"}


def test_ci_values_only_override_declared_keys():
    declared = load(CHART / "values.yaml")
    overrides = load(CHART / "ci/ci-values.yaml")
    assert set(overrides) <= set(declared)
    readiness = overrides["readinessProbe"]
    assert isinstance(readiness, dict)
    assert readiness["httpGet"] is None
    assert readiness["tcpSocket"] == {"port": "http"}


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm unavailable")
def test_helm_lint_passes_with_default_and_ci_values():
    for arguments in ([], ["--values", str(CHART / "ci/ci-values.yaml")]):
        code, stdout, stderr = asyncio.run(helm("lint", "--strict", str(CHART), *arguments))
        assert code == 0, stdout + stderr


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm unavailable")
def test_rendered_deployment_mounts_the_config_and_probes_health():
    code, stdout, stderr = asyncio.run(
        helm("template", "demo", str(CHART), "--set", "config.server.listen=0.0.0.0:9000")
    )
    assert code == 0, stderr
    documents = {
        (item["kind"], item["metadata"]["name"]): item
        for item in yaml.safe_load_all(stdout)
        if item
    }
    assert set(documents) == {
        ("ServiceAccount", "demo-promcast"),
        ("ConfigMap", "demo-promcast"),
        ("Service", "demo-promcast"),
        ("Deployment", "demo-promcast"),
    }
    pod = documents["Deployment", "demo-promcast"]["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"] == "ghcr.io/intesarjawad/promcast:0.1.0"
    assert container["args"] == ["serve", "--config", "/etc/promcast/promcast.yaml"]
    assert container["ports"][0]["containerPort"] == 9000
    assert container["readinessProbe"]["httpGet"]["path"] == "/healthz"
    assert documents["Service", "demo-promcast"]["spec"]["ports"][0]["port"] == 9000
    rendered = documents["ConfigMap", "demo-promcast"]["data"]["promcast.yaml"]
    assert yaml.safe_load(rendered)["server"]["listen"] == "0.0.0.0:9000"
