"""Fetch the pinned model weights and configuration and verify their digests."""

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Protocol

import httpx
import huggingface_hub

from promcast.shell.errors import ModelFilesMissingError

LOGGER = logging.getLogger(__name__)
MODELS_DIRECTORY = Path.home() / ".cache" / "promcast" / "models"
WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"


class ModelHub(Protocol):
    """Typed boundary for the Hub download API's partially typed options."""

    def hf_hub_download(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str,
        local_dir: Path,
        token: bool,
    ) -> str:
        """Download one file at an immutable revision to a local directory."""
        ...


@dataclass(frozen=True)
class ModelArtifact:
    """Immutable source revision and expected digests of the weights and configuration."""

    name: str
    repository: str
    revision: str
    sha256: str
    config_sha256: str


TOTO_MODELS = (
    ModelArtifact(
        "toto-2.0-22m",
        "Datadog/Toto-2.0-22m",
        "685e4ae3e2be8d8998025e53dd98e7fdcb296a89",
        "9cd503d82df3aa71747862688f47a31c1d0a4b80f898df6e046189016eaa21dd",
        "abeaf0fcd54aaac66757fde69ec3ddb4d3bfdcf96e0c8f767aefd03ab4c9e8d9",
    ),
    ModelArtifact(
        "toto-2.0-4m",
        "Datadog/Toto-2.0-4m",
        "8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9",
        "316660d5afb47943e531f39242e0b02ca0b8bb73be5709dfe07ca80dfce9805e",
        "7a926d130e401ab0c5fdb3564f46c8d917bd05c7b3ae26b9c22d2da2ef01d2d8",
    ),
)
COMPARE_MODEL = ModelArtifact(
    "chronos-2-small",
    "autogluon/chronos-2-small",
    "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a",
    "492290ae82bb89f9769e3479ce90b3179de1f33e600c34daa0352531538b23cd",
    "f2780468adc8b16322aa0b6e28b26476c9401ecf64ab53292b6d2b0d3f423722",
)


def verify_file(path: Path, expected_sha256: str) -> None:
    """Reject a file whose bytes differ from the pinned digest."""
    with path.open("rb") as weights:
        actual_sha256 = hashlib.file_digest(weights, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        message = (
            f"file {path} has sha256 {actual_sha256}, expected {expected_sha256}; "
            f"remove {path} and rerun python -m promcast.shell.model_download"
        )
        raise ValueError(message)


def fetch_model(
    model: ModelArtifact, models_directory: Path, hub: ModelHub = huggingface_hub
) -> None:
    """Download missing weights and configuration, then verify new and cached copies."""
    directory = models_directory / model.name
    for filename, expected_sha256 in (
        (WEIGHTS_FILENAME, model.sha256),
        (CONFIG_FILENAME, model.config_sha256),
    ):
        path = directory / filename
        if path.exists():
            verify_file(path, expected_sha256)
            LOGGER.info("Verified %s (%d bytes); no download", path, path.stat().st_size)
            continue

        directory.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        hub.hf_hub_download(
            repo_id=model.repository,
            filename=filename,
            revision=model.revision,
            local_dir=directory,
            token=False,
        )
        elapsed_seconds = perf_counter() - started
        verify_file(path, expected_sha256)
        LOGGER.info(
            "Downloaded and verified %s: %d bytes in %.3f seconds",
            path,
            path.stat().st_size,
            elapsed_seconds,
        )


def download_model(model_dir: Path, name: str) -> None:
    """Fetch a missing pinned model and report failures without requiring a checkout."""
    artifact = next(model for model in (*TOTO_MODELS, COMPARE_MODEL) if model.name == name)
    sys.stderr.write(f"Downloading {artifact.repository}@{artifact.revision} to {model_dir}\n")
    try:
        fetch_model(replace(artifact, name=model_dir.name), model_dir.parent)
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise ModelFilesMissingError(
            f"download or checksum verification failed at {model_dir}: {error}"
        ) from None


def main() -> None:
    """Fetch Toto weights and optionally the replay comparison weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-compare", action="store_true", help="also fetch the pinned Chronos replay weights"
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    models = (*TOTO_MODELS, COMPARE_MODEL) if arguments.with_compare else TOTO_MODELS
    for model in models:
        fetch_model(model, MODELS_DIRECTORY)


if __name__ == "__main__":
    main()
