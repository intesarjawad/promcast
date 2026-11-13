import asyncio
import json
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class Dependency(BaseModel):
    condition: str


class MountedConfig(BaseModel):
    source: str
    target: str


class Healthcheck(BaseModel):
    test: list[str]
    interval: str
    timeout: str
    retries: int


class Service(BaseModel):
    image: str
    build: str | None = None
    entrypoint: str | None = None
    command: list[str] = Field(default_factory=list)
    volumes: list[str] = Field(default_factory=list)
    configs: list[MountedConfig] = Field(default_factory=list[MountedConfig])
    depends_on: dict[str, Dependency] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    healthcheck: Healthcheck | None = None


class InlineConfig(BaseModel):
    content: str


class Compose(BaseModel):
    services: dict[str, Service]
    volumes: dict[str, None]
    configs: dict[str, InlineConfig]


class ScrapeConfig(BaseModel):
    job_name: str
    static_configs: list[dict[str, list[str]]]


class GlobalConfig(BaseModel):
    scrape_interval: str
    evaluation_interval: str


class PrometheusConfig(BaseModel):
    global_config: GlobalConfig = Field(alias="global")
    rule_files: list[str]
    scrape_configs: list[ScrapeConfig]


@pytest.fixture
def compose() -> Compose:
    return Compose.model_validate(yaml.safe_load((ROOT / "docker-compose.yml").read_text()))


@pytest.mark.parametrize(
    "filename",
    [
        "docker-compose.yml",
        "demo/promcast.yaml",
        "demo/promcast.rules.yml",
        "demo/demo.rules.yml",
        "tests/demo/demo.rules.test.yml",
    ],
)
def test_yaml_parses(filename: str):
    assert isinstance(yaml.safe_load((ROOT / filename).read_text()), dict)


def test_services_volumes_and_healthcheck(compose: Compose):
    assert set(compose.services) == {
        "history",
        "backfill",
        "prometheus",
        "alertmanager",
        "scenario",
        "promcast",
        "grafana",
    }
    assert set(compose.volumes) == {"work", "prometheus-data"}
    assert compose.services["prometheus"].healthcheck == Healthcheck(
        test=["CMD", "wget", "-q", "-O", "/dev/null", "http://localhost:9090/-/ready"],
        interval="5s",
        timeout="2s",
        retries=24,
    )
    assert all(
        service.healthcheck is None
        for name, service in compose.services.items()
        if name != "prometheus"
    )


def test_backfill_precedes_prometheus_and_shares_storage(compose: Compose):
    services = compose.services
    assert services["backfill"].depends_on["history"].condition == "service_completed_successfully"
    assert (
        services["prometheus"].depends_on["backfill"].condition == "service_completed_successfully"
    )
    assert services["scenario"].depends_on["history"].condition == "service_completed_successfully"
    assert services["promcast"].depends_on["prometheus"].condition == "service_healthy"
    assert "prometheus-data:/prometheus" in services["backfill"].volumes
    assert "prometheus-data:/prometheus" in services["prometheus"].volumes
    assert "work:/work" in services["history"].volumes
    assert "work:/work:ro" in services["scenario"].volumes
    assert services["backfill"].entrypoint == "/bin/sh"


def test_backfill_import_runs_once(compose: Compose, tmp_path: Path):
    command = compose.services["backfill"].command[-1].replace("/prometheus", str(tmp_path))
    script = 'promtool() { echo "$@" >> "' + str(tmp_path / "calls") + '"; }\n' + command
    for _ in range(2):
        assert asyncio.run(run_shell(script)) == (0, "")
    assert (tmp_path / "calls").read_text().splitlines() == [
        f"tsdb create-blocks-from openmetrics /work/history.om {tmp_path}"
    ]


def test_backfill_rejects_interrupted_import(compose: Compose, tmp_path: Path):
    (tmp_path / ".backfill-started").touch()
    (tmp_path / "wal").mkdir()
    command = compose.services["backfill"].command[-1].replace("/prometheus", str(tmp_path))
    code, stderr = asyncio.run(run_shell(command))
    assert code != 0
    assert "interrupted" in stderr


def test_failed_import_cannot_be_retried(compose: Compose, tmp_path: Path):
    command = compose.services["backfill"].command[-1].replace("/prometheus", str(tmp_path))
    code, _ = asyncio.run(run_shell("promtool() { return 3; }\n" + command))
    assert code == 3
    assert (tmp_path / ".backfill-started").is_file()
    code, stderr = asyncio.run(run_shell(command))
    assert code != 0
    assert "interrupted" in stderr


def test_backfill_skips_existing_tsdb(compose: Compose, tmp_path: Path):
    (tmp_path / "wal").mkdir()
    command = compose.services["backfill"].command[-1].replace("/prometheus", str(tmp_path))
    assert asyncio.run(run_shell(command)) == (0, "")


def test_backfill_rejects_unmarked_blocks(compose: Compose, tmp_path: Path):
    (tmp_path / "block").mkdir()
    command = compose.services["backfill"].command[-1].replace("/prometheus", str(tmp_path))
    code, stderr = asyncio.run(run_shell(command))
    assert code != 0
    assert "unmarked" in stderr.lower()


async def run_shell(command: str) -> tuple[int | None, str]:
    process = await asyncio.create_subprocess_exec(
        "/bin/sh", "-ec", command, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    return process.returncode, stderr.decode()


def test_mounts_are_existing_files_or_declared_volumes(compose: Compose):
    for service in compose.services.values():
        for mount in service.volumes:
            source, _ = mount.split(":", maxsplit=1)
            if source.startswith("."):
                assert (ROOT / source).is_file(), source
                assert mount.endswith(":ro")
            else:
                assert source in compose.volumes
        for config in service.configs:
            assert config.source in compose.configs
    for config in compose.configs.values():
        assert isinstance(yaml.safe_load(config.content), dict)


def test_release_image_has_source_build_fallback(compose: Compose):
    assert compose.services["promcast"].image == (
        "ghcr.io/intesarjawad/promcast:${PROMCAST_VERSION:-0.1.0}"
    )
    assert compose.services["promcast"].build == "."
    raw = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert "pull_policy" not in raw["services"]["promcast"]


def test_scrapes_and_rule_mounts_match_prometheus_config(compose: Compose):
    config = PrometheusConfig.model_validate(yaml.safe_load(compose.configs["prometheus"].content))
    assert config.global_config.scrape_interval == "5s"
    assert config.global_config.evaluation_interval == "15s"
    jobs = {job.job_name: job.static_configs for job in config.scrape_configs}
    assert jobs["scenario"] == [{"targets": ["scenario:9100"]}]
    assert jobs["promcast"] == [{"targets": ["promcast:8787"]}]
    mounted_paths = {mount.split(":")[1] for mount in compose.services["prometheus"].volumes}
    assert len(config.rule_files) == 2
    assert set(config.rule_files) <= mounted_paths


def test_grafana_home_is_provisioned_and_datasource_is_named(compose: Compose):
    grafana = compose.services["grafana"]
    assert grafana.environment["GF_AUTH_ANONYMOUS_ENABLED"] == "true"
    assert grafana.environment["GF_AUTH_ANONYMOUS_ORG_ROLE"] == "Viewer"
    home = grafana.environment["GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH"]
    mounts = {mount.split(":")[1]: mount.split(":")[0] for mount in grafana.volumes}
    assert mounts[home] == "./grafana/dashboards/promcast.json"
    assert mounts["/etc/grafana/provisioning/datasources/promcast.yml"] == (
        "./grafana/provisioning/datasources/promcast.yml"
    )
    assert mounts["/etc/grafana/provisioning/dashboards/promcast.yml"] == (
        "./grafana/provisioning/dashboards/promcast.yml"
    )
    dashboard = json.loads((ROOT / mounts[home]).read_text())
    assert dashboard["uid"] == "promcast-demo"
    assert dashboard["panels"]
    datasources = yaml.safe_load(
        (ROOT / "grafana/provisioning/datasources/promcast.yml").read_text()
    )
    assert any(
        source["uid"] == "promcast" and source["url"] == "http://promcast:8787"
        for source in datasources["datasources"]
    )


def test_promtool_demo_alert_behavior(compose: Compose, tmp_path: Path):
    promtool = shutil.which("promtool")
    assert promtool is not None, "promtool must be installed to validate demo rules"
    config_path = tmp_path / "prometheus.yml"
    config_path.write_text(
        compose.configs["prometheus"].content.replace(
            "/etc/prometheus/demo/", str(ROOT / "demo") + "/"
        )
    )

    async def check_rules():
        for command in (
            ("check", "config", str(config_path)),
            ("test", "rules", str(ROOT / "tests/demo/demo.rules.test.yml")),
        ):
            process = await asyncio.create_subprocess_exec(
                promtool,
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            assert process.returncode == 0, (stdout + stderr).decode()

    asyncio.run(check_rules())
