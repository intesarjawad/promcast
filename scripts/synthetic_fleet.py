"""Scan a synthetic fleet with toto-2.0-22m and write the document the design assets use.

Five series have four days of 5 minute history and a fixed seed: a leak that accelerates, a
nightly spool that refills, a slow drift, and two quiet series. A sixth mount has 18 samples,
so it is still warming up. The output has the shape of `promcast scan --json`.

    uv run python scripts/synthetic_fleet.py docs/design/fleet-scan.json
    uv run python scripts/synthetic_fleet.py --check docs/design/fleet-scan.json
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from promcast.core.config import ModelConfig
from promcast.core.forecaster import Toto2Forecaster
from promcast.core.recipes import scan_target
from promcast.core.series import FloatArray, Series
from promcast.shell.discover import History, load_recipes
from promcast.shell.scan import ScanDocument, ScanRow, ScanSkipped, _forecast_rows  # pyright: ignore[reportPrivateUsage]

STEP = 300
END = 1794319200  # 2026-11-10 14:00 UTC
SAMPLES = 1152  # four days
FRESH_SAMPLES = 18
DAY = 86400
MODEL = "toto-2.0-22m"


def fleet() -> dict[str, list[tuple[dict[str, str], FloatArray]]]:
    """Build the series; every value is a function of the seed, so reruns are identical."""
    t = END + STEP * (np.arange(SAMPLES) - (SAMPLES - 1))
    rng = np.random.default_rng(7)
    age = (t - (END - 3 * 3600)) / 3600.0  # the leak started 3 h ago
    leak = np.where(age > 0, 0.012 * age + 0.010 * age**2, 0)
    leaking = 0.31 + 0.01 * np.sin(2 * np.pi * t / DAY) - leak + rng.normal(0, 0.0015, SAMPLES)
    phase = ((t - END) / DAY + 0.04) % 1.0  # a batch spool that dips each afternoon
    spool = 0.40 - 0.30 * np.exp(-(((phase - 0.10) / 0.06) ** 2)) + rng.normal(0, 0.003, SAMPLES)
    hours = np.clip((t - (END - 8 * 3600)) / 3600, 0, None)  # log rotation broke 8 h ago
    drift = (
        0.14 - 0.0075 * hours + 0.004 * np.sin(2 * np.pi * t / DAY) + rng.normal(0, 0.001, SAMPLES)
    )
    memory = 0.42 + 0.03 * np.sin(2 * np.pi * t / DAY + 1) + rng.normal(0, 0.006, SAMPLES)
    quiet = 0.61 + rng.normal(0, 0.002, SAMPLES)
    return {
        "disk": [
            ({"instance": "db-01", "mountpoint": "/var/lib/postgresql"}, leaking),
            ({"instance": "ingest-02", "mountpoint": "/var/spool/batch"}, spool),
            ({"instance": "logs-01", "mountpoint": "/var/log"}, drift),
            ({"instance": "web-03", "mountpoint": "/"}, quiet),
        ],
        "memory": [({"instance": "cache-01"}, memory)],
    }


def scan(model_dir: Path) -> ScanDocument:
    """Forecast every series through the same row builder `promcast scan` uses."""
    forecaster = Toto2Forecaster(model_dir, MODEL, batch_size=16)
    recipes = {recipe.name: recipe for recipe in load_recipes()}
    rows: list[ScanRow] = []
    skipped: list[ScanSkipped] = []
    for name, members in fleet().items():
        recipe = recipes[name]
        series = [
            Series(tuple(sorted(labels.items())), STEP, float(END), np.clip(values, 0, 1))
            for labels, values in members
        ]
        if name == "disk":
            fresh = np.full(SAMPLES, np.nan)
            fresh[-FRESH_SAMPLES:] = 0.5
            series.append(
                Series((("instance", "edge-04"), ("mountpoint", "/data")), STEP, float(END), fresh)
            )
        history = History(
            scan_target(recipe, STEP, 50),
            series,
            {item.labels: STEP for item in series},
            {item.labels: int(np.count_nonzero(~np.isnan(item.values))) for item in series},
            len(series),
        )
        found, rejected = _forecast_rows(recipe, history, forecaster, end_ts=END)
        rows += found
        skipped += rejected
    rows.sort(key=lambda r: r.promcast_breach_ts if r.promcast_breach_ts is not None else math.inf)
    return ScanDocument(
        url="http://prometheus:9090",
        model=forecaster.name,
        origin_ts=END,
        rows=rows,
        skipped=skipped,
    )


def render(model_dir: Path) -> str:
    """The committed file's exact text."""
    return scan(model_dir).model_dump_json(indent=1) + "\n"


def main() -> int:
    """Write the document, or with --check fail when the file differs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--model-dir", type=Path, default=ModelConfig().cache_dir.expanduser() / MODEL
    )
    arguments = parser.parse_args()
    text = render(arguments.model_dir)
    if not arguments.check:
        arguments.out.write_text(text, encoding="utf-8")
        return 0
    if arguments.out.read_text(encoding="utf-8") != text:
        sys.stderr.write(
            f"error: {arguments.out} differs from the generator\n"
            f"next:  run uv run python scripts/synthetic_fleet.py {arguments.out}\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
