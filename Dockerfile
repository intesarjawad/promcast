FROM python:3.12-slim AS base

FROM base AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /opt/promcast
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable --no-cache

# Configuration and weights must match the same pinned revision and verified digests.
RUN .venv/bin/python - <<'PY'
import logging
from pathlib import Path

from promcast.shell.model_download import TOTO_MODELS, fetch_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
for model in TOTO_MODELS:
    fetch_model(model, Path.cwd() / "models")
PY

RUN --network=none chmod -R a+rX /opt/promcast/models

FROM base AS runtime
ENV PATH=/opt/promcast/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/opt/promcast/models \
    PROMCAST_CACHE_DIR=/opt/promcast/models \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /opt/promcast
COPY --from=build /opt/promcast/.venv /opt/promcast/.venv
COPY --from=build /opt/promcast/models /opt/promcast/models
# Licence texts that the bundled wheels themselves do not carry; licences/SOURCES pins each one.
COPY licences/ /opt/promcast/licences/
USER 65532

# Fail the build if the installed package or either bundled model needs a download.
RUN --network=none python - <<'PY'
from importlib.metadata import distribution
from pathlib import Path

import torch

from promcast.core.forecaster import Toto2Forecaster

assert torch.version.cuda is None, "the image requires CPU-only torch"
for package in ("numpy", "torch", "augurs"):
    texts = [file for file in distribution(package).files or () if "licenses" in file.parts]
    assert texts and all(file.locate().is_file() for file in texts), f"{package} licences missing"
for name in ("toto-2.0-22m", "toto-2.0-4m"):
    Toto2Forecaster(Path("/opt/promcast/models") / name, name, batch_size=1, threads=1)
PY

# GluonTS imports Matplotlib, which requires a writable cache even for inference.
VOLUME ["/tmp"]
EXPOSE 8787
ENTRYPOINT ["promcast"]
CMD ["serve", "--config", "/etc/promcast.yaml"]
