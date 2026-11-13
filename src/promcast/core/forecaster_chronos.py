import os
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from promcast.core.series import FloatArray, Series

# ADR 0010 fixes the requested levels and the list-of-(1, horizon, 9) contract.
_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@runtime_checkable
class _QuantilePipeline(Protocol):
    def predict_quantiles(
        self, inputs: torch.Tensor, prediction_length: int, quantile_levels: list[float]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]: ...


@runtime_checkable
class _PipelineLoader(Protocol):
    def from_pretrained(self, path: str, *, device_map: str, local_files_only: bool) -> object: ...


class Chronos2Forecaster:
    """Run the replay-only comparison on local CPU weights with NaN missingness."""

    name = "chronos-2-small"

    def __init__(self, model_dir: Path, batch_size: int = 16, threads: int | None = None) -> None:
        if batch_size <= 0 or (threads is not None and threads <= 0):
            raise ValueError("batch_size and threads must be positive")
        loader: object = import_module("chronos").BaseChronosPipeline
        if not isinstance(loader, _PipelineLoader):
            raise TypeError("BaseChronosPipeline must provide from_pretrained")
        for filename in ("config.json", "model.safetensors"):
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"local Chronos model requires {model_dir / filename}")
        self._batch_size = batch_size
        torch.set_num_threads(
            threads if threads is not None else max(1, (os.cpu_count() or 2) // 2)
        )
        pipeline = loader.from_pretrained(str(model_dir), device_map="cpu", local_files_only=True)
        if not isinstance(pipeline, _QuantilePipeline):
            raise TypeError("Chronos pipeline must provide predict_quantiles")
        self._pipeline = pipeline

    def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
        """Return nine shared quantiles in input order, preserving invalid outputs."""
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        groups: dict[int, list[tuple[int, Series]]] = defaultdict(list)
        for index, series in enumerate(batch):
            groups[series.values.size].append((index, series))
        results: dict[int, FloatArray] = {}
        with torch.inference_mode():
            for group in groups.values():
                for start in range(0, len(group), self._batch_size):
                    chunk = group[start : start + self._batch_size]
                    inputs = torch.tensor(
                        np.stack([series.values for _, series in chunk]), dtype=torch.float32
                    )[:, None, :]
                    outputs, _ = self._pipeline.predict_quantiles(
                        inputs,
                        prediction_length=horizon_steps,
                        quantile_levels=_QUANTILE_LEVELS.copy(),
                    )
                    if len(outputs) != len(chunk):
                        raise ValueError("Chronos must return one output per input series")
                    for (index, _), output in zip(chunk, outputs, strict=True):
                        if output.shape != (1, horizon_steps, len(_QUANTILE_LEVELS)):
                            raise ValueError(
                                f"unexpected Chronos forecast shape: {tuple(output.shape)}"
                            )
                        result = output[0].cpu().numpy().astype(np.float64).T.copy()
                        results[index] = (
                            np.maximum.accumulate(result, axis=0)
                            if np.isfinite(result).all()
                            else result
                        )
        return [results[index] for index in range(len(batch))]
