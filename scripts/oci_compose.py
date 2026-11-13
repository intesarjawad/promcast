"""Generate a self-contained Compose demo with literal configuration content."""

import argparse
import re
from pathlib import Path

import yaml
from image_tag import image_tag, package_version
from pydantic import BaseModel, ConfigDict, Field, JsonValue

ROOT = Path(__file__).resolve().parents[1]


class Service(BaseModel):
    """Preserve Compose options while exposing local build and mount inputs."""

    model_config = ConfigDict(extra="allow")
    image: str
    build: JsonValue = None
    volumes: list[str] = Field(default_factory=list)
    configs: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])


class Compose(BaseModel):
    """Preserve top-level options and validate the supported inline configs."""

    model_config = ConfigDict(extra="allow")
    services: dict[str, Service]
    configs: dict[str, dict[str, str]]
    volumes: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


def generate(root: Path) -> str:
    """Inline single-file read-only mounts and pin the image tag of the package version."""
    compose = Compose.model_validate(yaml.safe_load((root / "docker-compose.yml").read_text()))
    tag = image_tag(package_version(root / "pyproject.toml"))
    sources: dict[str, str] = {}
    for service in compose.services.values():
        service.image = re.sub(r"\$\{PROMCAST_VERSION(?::-[^}]*)?\}", tag, service.image)
        if "$" in service.image:
            raise ValueError(f"Expected a literal image reference: {service.image}")
        volumes: list[str] = []
        configs = list(service.configs)
        for mount in service.volumes:
            source, target, *options = mount.split(":")
            if source in compose.volumes:
                volumes.append(mount)
                continue
            file = (root / source).resolve()
            if not file.is_relative_to(root.resolve()) or not file.is_file() or options != ["ro"]:
                raise ValueError(f"Expected a local read-only single-file mount: {mount}")
            if source not in sources:
                key = f"inline-file-{len(sources) + 1}"
                if key in compose.configs:
                    raise ValueError(f"Config name collision: {key}")
                sources[source] = key
                compose.configs[key] = {"content": file.read_text(encoding="utf-8")}
            configs.append({"source": sources[source], "target": target})
        if service.volumes:
            service.volumes = volumes
        if configs:
            service.configs = configs
    document = compose.model_dump(exclude_unset=True)
    for service in document["services"].values():
        service.pop("build", None)
    # Compose also interpolates dollar signs inside embedded source and rule text.
    return yaml.safe_dump(document, sort_keys=False, width=100).replace("$", "$$")


def main() -> None:
    """Write the artifact, or check its exact contents without changing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "deploy/demo/compose.yaml"
    content = generate(ROOT)
    if args.check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != content:
            parser.exit(1, "Compose artifact is stale; run uv run python scripts/oci_compose.py\n")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
