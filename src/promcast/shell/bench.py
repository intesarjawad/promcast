"""Isolated CPU cycle measurements for the two supported Toto models."""

import asyncio
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

MODELS = ("toto-2.0-22m", "toto-2.0-4m")
CONTEXTS = (512, 1472)
SERIES_COUNT = 50
HORIZON = 120
REPEATS = 3


@dataclass(frozen=True)
class Measurement:
    """One isolated process's cycle timings and resident memory high-water mark."""

    model: str
    context_steps: int
    seconds_per_cycle: list[float]
    median_seconds_per_cycle: float
    peak_rss_bytes: int
    torch_threads: int


class BenchmarkReport(TypedDict):
    """Host metadata and the four measurements used to choose a default."""

    cpu_model: str
    platform: str
    series_count: int
    horizon_steps: int
    repeats: int
    batch_size: int
    cycle_budget_s: float
    measurements: list[Measurement]
    recommended: str


def cpu_model() -> str:
    """Read the hardware model rather than the operating system's architecture."""
    if sys.platform == "darwin":
        hardware = subprocess.run(
            ["/usr/sbin/system_profiler", "SPHardwareDataType"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in hardware.splitlines():
            key, _, value = line.partition(":")
            if key.strip() in {"Chip", "Processor Name"}:
                return value.strip()
        raise ValueError("CPU model missing from system_profiler hardware report")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() in {"model name", "Model", "Hardware"}:
                return value.strip()
    return platform.processor() or platform.machine()


def recommended(measurements: list[Measurement], budget_s: float) -> str:
    """Choose the smaller model when the full-context median reaches the cycle budget."""
    full = next(
        item
        for item in measurements
        if item.model == MODELS[0] and item.context_steps == CONTEXTS[-1]
    )
    return MODELS[0] if full.median_seconds_per_cycle < budget_s else MODELS[1]


async def benchmark(cache: Path) -> BenchmarkReport:
    """Run each model/context in a fresh process to isolate peak RSS."""
    cpu = cpu_model()
    measurements: list[Measurement] = []
    for model in MODELS:
        for context in CONTEXTS:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "promcast.shell.bench_case",
                "--model",
                model,
                "--context",
                str(context),
                "--cache-dir",
                str(cache),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode:
                raise ValueError(
                    f"benchmark failed for {model}, context {context}: {stderr.decode().strip()}"
                )
            measurements.append(Measurement(**json.loads(stdout)))
    budget_s = 60.0 if sys.platform == "darwin" else 120.0
    return {
        "cpu_model": cpu,
        "platform": platform.platform(),
        "series_count": SERIES_COUNT,
        "horizon_steps": HORIZON,
        "repeats": REPEATS,
        "batch_size": 16,
        "cycle_budget_s": budget_s,
        "measurements": measurements,
        "recommended": recommended(measurements, budget_s),
    }


def render(report: BenchmarkReport, *, json_output: bool) -> str:
    """Render measurements as a machine-readable report or a terminal table."""
    if json_output:
        document = {**report, "measurements": [asdict(item) for item in report["measurements"]]}
        return json.dumps(document, indent=2, allow_nan=False) + "\n"
    lines = [
        f"CPU: {report['cpu_model']}",
        f"{SERIES_COUNT} series, horizon {HORIZON}, {REPEATS} repeats, batch size 16",
        "Model          Context  Median seconds  Peak RSS bytes  Torch threads",
    ]
    lines.extend(
        f"{item.model:<14} {item.context_steps:>7}  {item.median_seconds_per_cycle:>14.3f}"
        f"  {item.peak_rss_bytes:>14}  {item.torch_threads:>13}"
        for item in report["measurements"]
    )
    lines.append(f"Recommended: {report['recommended']} (budget {report['cycle_budget_s']:g} s)")
    return "\n".join(lines) + "\n"
