import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml

from promcast.core.config import Config
from promcast.shell.errors import InitTargetExists
from promcast.shell.plotting import adoption_dashboard
from promcast.shell.rules import generate_rules


def _container_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        authority = "host.docker.internal" + (f":{parsed.port}" if parsed.port else "")
        return urlunsplit(parsed._replace(netloc=authority))
    return url


def write_init(config: Config, out: Path, *, force: bool = False) -> None:
    """Write adoption files exclusively unless overwriting was explicitly selected."""
    document = config.model_dump(mode="json")
    # Runtime duration parsing accepts compact seconds, not ISO 8601 durations.
    document["cycle"] = f"{int(config.cycle.total_seconds())}s"
    targets: list[dict[str, object]] = []
    for target in config.targets:
        values = target.model_dump(mode="json")
        for name in ("step", "context", "horizon", "season", "warn", "linear_window"):
            values[name] = f"{int(getattr(target, name).total_seconds())}s"
        targets.append(values)
    document["targets"] = targets
    container_url = _container_url(str(config.tsdb.url))
    document["tsdb"] = {"url": container_url}
    compose = {
        "services": {
            "promcast": {
                "image": "ghcr.io/intesarjawad/promcast:0.1.0",
                "command": ["serve", "--config", "/etc/promcast/promcast.yaml"],
                "ports": ["8787:8787"],
                "volumes": ["./promcast.yaml:/etc/promcast/promcast.yaml:ro"],
                "extra_hosts": ["host.docker.internal:host-gateway"],
            }
        }
    }
    snippet = {
        "scrape_configs": [
            {"job_name": "promcast", "static_configs": [{"targets": ["host.docker.internal:8787"]}]}
        ],
        "rule_files": ["promcast.rules.yml"],
    }
    datasource = {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "uid": "promcast-tsdb",
                "type": "prometheus",
                "access": "proxy",
                "url": container_url,
            }
        ],
    }
    provider = {
        "apiVersion": 1,
        "providers": [
            {"name": "promcast", "type": "file", "options": {"path": "/var/lib/grafana/dashboards"}}
        ],
    }
    artifacts = {
        "promcast.yaml": yaml.safe_dump(document, sort_keys=False),
        "promcast.rules.yml": generate_rules(config),
        "prometheus-snippet.yml": yaml.safe_dump(snippet, sort_keys=False),
        "compose.yml": yaml.safe_dump(compose, sort_keys=False),
        "grafana/provisioning/datasources/promcast.yml": yaml.safe_dump(datasource),
        "grafana/provisioning/dashboards/promcast.yml": yaml.safe_dump(provider),
        "grafana/dashboards/promcast.json": json.dumps(adoption_dashboard(), indent=2) + "\n",
    }
    if not force:
        for name in artifacts:
            if (out / name).exists():
                raise InitTargetExists(str(out / name))
    for name, text in artifacts.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w" if force else "x", encoding="utf-8") as stream:
                stream.write(text)
        except FileExistsError:
            raise InitTargetExists(str(path)) from None
