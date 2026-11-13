"""Single-process benchmark measurement, invoked by the benchmark command."""

import argparse
import json
import resource
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from promcast.core.forecaster import Toto2Forecaster
from promcast.core.series import Series
from promcast.shell.bench import CONTEXTS, HORIZON, MODELS, REPEATS, SERIES_COUNT, Measurement


def measure(model: str, context: int, cache: Path) -> Measurement:
    """Time three complete forecast batches, excluding model load and validation."""
    forecaster = Toto2Forecaster(cache / model, model)
    steps = np.arange(context, dtype=np.float64)
    batch = [
        Series(
            (("series", str(index)),),
            60,
            float((context - 1) * 60),
            10 + index + np.sin(steps / (5 + index)) + steps / 100,
        )
        for index in range(SERIES_COUNT)
    ]
    timings: list[float] = []
    for _ in range(REPEATS):
        started = perf_counter()
        output = forecaster.forecast(batch, HORIZON)
        timings.append(perf_counter() - started)
        if len(output) != SERIES_COUNT or any(
            values.shape != (9, HORIZON)
            or not np.isfinite(values).all()
            or not (np.diff(values, axis=0) >= 0).all()
            for values in output
        ):
            raise ValueError(f"{model} produced invalid benchmark quantiles")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux reports KiB.
    peak_rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    return Measurement(
        model, context, timings, median(timings), peak_rss_bytes, torch.get_num_threads()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--context", type=int, choices=CONTEXTS, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    arguments = parser.parse_args()
    sys.stdout.write(
        json.dumps(asdict(measure(arguments.model, arguments.context, arguments.cache_dir))) + "\n"
    )
