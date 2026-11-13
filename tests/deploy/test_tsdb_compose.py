import asyncio
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import BaseModel, Field

from promcast.core.alerting import time_to_breach_s
from promcast.core.baselines import linear

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "deploy/compose"
CHECK = ROOT / "scripts/tsdb_check.sh"


class MountedConfig(BaseModel):
    source: str
    target: str


class Service(BaseModel):
    image: str
    command: list[str] = Field(default_factory=list[str])
    volumes: list[str] = Field(default_factory=list[str])
    configs: list[MountedConfig] = Field(default_factory=list[MountedConfig])
    profiles: list[str] = Field(default_factory=list[str])
    ports: list[str] = Field(default_factory=list[str])


class InlineConfig(BaseModel):
    content: str


class Compose(BaseModel):
    services: dict[str, Service]
    volumes: dict[str, None]
    configs: dict[str, InlineConfig]


class RemoteWrite(BaseModel):
    url: str
    headers: dict[str, str] = Field(default_factory=dict[str, str])


class PrometheusConfig(BaseModel):
    remote_write: list[RemoteWrite]


class TSDB(BaseModel):
    url: str
    headers: dict[str, str] = Field(default_factory=dict[str, str])


class PromcastConfig(BaseModel):
    tsdb: TSDB


class FilesystemStorage(BaseModel):
    dir: str


class CommonStorage(BaseModel):
    backend: str
    filesystem: FilesystemStorage


class Common(BaseModel):
    storage: CommonStorage


class BlockTSDB(BaseModel):
    dir: str


class BlocksStorage(BaseModel):
    storage_prefix: str
    tsdb: BlockTSDB


class Ring(BaseModel):
    replication_factor: int


class Ingester(BaseModel):
    ring: Ring


class StoreGateway(BaseModel):
    sharding_ring: Ring


class Server(BaseModel):
    http_listen_port: int


class MimirConfig(BaseModel):
    target: str
    multitenancy_enabled: bool
    server: Server
    common: Common
    blocks_storage: BlocksStorage
    ingester: Ingester
    store_gateway: StoreGateway


def compose(name: str) -> Compose:
    return Compose.model_validate(yaml.safe_load((COMPOSE / name).read_text()))


def service_config(document: Compose, service: str, target: str) -> str:
    sources = {entry.target: entry.source for entry in document.services[service].configs}
    return document.configs[sources[target]].content


def parse(content: str, model: type[BaseModel]) -> BaseModel:
    return model.model_validate(yaml.safe_load(content))


def predict_linear_breach_s(samples: list[float], step_s: int, threshold: float) -> float:
    """Solve Prometheus predict_linear for the first crossing, in seconds from origin."""
    times = [(index - len(samples) + 1) * step_s for index in range(len(samples))]
    mean_time = sum(times) / len(times)
    mean_value = sum(samples) / len(samples)
    slope = sum(
        (time - mean_time) * (value - mean_value)
        for time, value in zip(times, samples, strict=True)
    ) / sum((time - mean_time) ** 2 for time in times)
    at_origin = mean_value - slope * mean_time
    return (threshold - at_origin) / slope


def test_victoriametrics_compose_scrapes_and_persists():
    document = compose("victoriametrics.yml")
    assert set(document.services) == {"victoriametrics", "prometheus", "promcast"}
    assert set(document.volumes) == {"victoriametrics-data", "prometheus-data"}
    assert document.services["victoriametrics"].image == (
        "victoriametrics/victoria-metrics:v1.155.0"
    )
    assert (
        "victoriametrics-data:/victoria-metrics-data"
        in document.services["victoriametrics"].volumes
    )
    assert document.services["victoriametrics"].ports == ["127.0.0.1:8428:8428"]
    prometheus = parse(
        service_config(document, "prometheus", "/etc/prometheus/prometheus.yml"), PrometheusConfig
    )
    assert isinstance(prometheus, PrometheusConfig)
    assert prometheus.remote_write == [RemoteWrite(url="http://victoriametrics:8428/api/v1/write")]
    promcast = parse(service_config(document, "promcast", "/etc/promcast.yaml"), PromcastConfig)
    assert isinstance(promcast, PromcastConfig)
    assert promcast.tsdb == TSDB(url="http://victoriametrics:8428")


def test_mimir_compose_writes_and_reads_the_demo_tenant():
    document = compose("mimir.yml")
    assert set(document.services) == {"mimir", "prometheus", "promcast"}
    assert set(document.volumes) == {"mimir-data", "prometheus-data"}
    assert document.services["mimir"].image == "grafana/mimir:3.2.4"
    assert "./mimir-config.yaml:/etc/mimir/mimir.yaml:ro" in document.services["mimir"].volumes
    assert document.services["mimir"].ports == ["127.0.0.1:8080:8080"]
    prometheus = parse(
        service_config(document, "prometheus", "/etc/prometheus/prometheus.yml"), PrometheusConfig
    )
    assert isinstance(prometheus, PrometheusConfig)
    assert prometheus.remote_write == [
        RemoteWrite(
            url="http://mimir:8080/api/v1/push",
            headers={"X-Scope-OrgID": "demo"},
        )
    ]
    promcast = parse(service_config(document, "promcast", "/etc/promcast.yaml"), PromcastConfig)
    assert isinstance(promcast, PromcastConfig)
    assert promcast.tsdb == TSDB(
        url="http://mimir:8080/prometheus", headers={"X-Scope-OrgID": "demo"}
    )


def test_mimir_runs_in_one_process_on_filesystem_storage():
    config = MimirConfig.model_validate(yaml.safe_load((COMPOSE / "mimir-config.yaml").read_text()))
    assert config.target == "all"
    assert config.multitenancy_enabled is True
    assert config.server.http_listen_port == 8080
    assert config.common.storage.backend == "filesystem"
    assert config.common.storage.filesystem.dir == "/data"
    assert config.blocks_storage.storage_prefix == "blocks"
    assert config.blocks_storage.tsdb.dir.startswith("/data/")
    assert config.ingester.ring.replication_factor == 1
    assert config.store_gateway.sharding_ring.replication_factor == 1


def test_linear_breach_matches_predict_linear_on_demo_series():
    # The demo alert rule's 3600-60x60 series at its first firing evaluation.
    samples = [3600.0, 3540.0, 3480.0, 3420.0]
    step_s = 60
    prometheus_s = predict_linear_breach_s(samples, step_s, threshold=0.0)
    assert prometheus_s == pytest.approx(3420.0)

    forecast = linear(np.array(samples), step_s, horizon_steps=120, window_steps=120)
    promcast_s = time_to_breach_s(forecast[4], step_s, threshold=0.0, direction="below")
    assert promcast_s is not None
    assert promcast_s == pytest.approx(3420.0)
    assert abs(promcast_s - prometheus_s) <= 0.01 * prometheus_s


def test_check_script_is_valid_posix_shell():
    async def invoke():
        process = await asyncio.create_subprocess_exec(
            "sh",
            "-n",
            str(CHECK),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        return process.returncode, stderr.decode()

    code, stderr = asyncio.run(invoke())
    assert code == 0, stderr


def test_check_script_rejects_unknown_backend():
    async def invoke():
        process = await asyncio.create_subprocess_exec(
            "sh",
            str(CHECK),
            "prometheus",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        return process.returncode, stderr.decode()

    code, stderr = asyncio.run(invoke())
    assert code == 2
    assert "Usage: scripts/tsdb_check.sh vm|mimir" in stderr
