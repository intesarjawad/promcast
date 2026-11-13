import os
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypedDict, Unpack, runtime_checkable

import numpy as np
import torch

from promcast.core.series import BoolArray, FloatArray, Series, left_pad_to_patch

# Defaults of toto2.configuration.Toto2GluonTSModelConfig (2.0.0), per ADR 0007.
_SCALER_FALLBACK_MIN_OBS = 8
_QUANTILE_REAL_CAP_K = 1e4
_QUANTILE_COUNT = 9


class Forecaster(Protocol):
    """Forecast prepared series in input order, each as nine quantiles by horizon."""

    name: str

    def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
        """Accept series whose final patch is observed after the caller's tail rule."""
        ...


class _ForecastOptions(TypedDict):
    decode_block_size: None
    has_missing_values: bool
    scaler_fallback_min_obs: int
    quantile_real_cap_k: float


@runtime_checkable
class _ForecastModel(Protocol):
    def forecast(
        self, inputs: dict[str, torch.Tensor], horizon: int, **options: Unpack[_ForecastOptions]
    ) -> torch.Tensor: ...


@runtime_checkable
class _ModelLoader(Protocol):
    def from_pretrained(
        self, path: str, *, local_files_only: bool, map_location: str
    ) -> object: ...


class Toto2Forecaster:
    """Run local Toto 2 weights on CPU with masked padding and bounded batches."""

    def __init__(
        self, model_dir: Path, name: str, batch_size: int = 16, threads: int | None = None
    ) -> None:
        if batch_size <= 0 or (threads is not None and threads <= 0):
            raise ValueError("batch_size and threads must be positive")
        for filename in ("config.json", "model.safetensors"):
            if not (model_dir / filename).is_file():
                raise FileNotFoundError(f"local Toto model requires {model_dir / filename}")
        self.name = name
        self._batch_size = batch_size
        torch.set_num_threads(
            threads if threads is not None else max(1, (os.cpu_count() or 2) // 2)
        )
        loader: object = import_module("toto2").Toto2Model
        if not isinstance(loader, _ModelLoader):
            raise TypeError("Toto2Model must provide from_pretrained")
        model = loader.from_pretrained(str(model_dir), local_files_only=True, map_location="cpu")
        if not isinstance(model, _ForecastModel):
            raise TypeError("Toto2Model must provide forecast")
        self._model = model

    def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
        """Pad and group prepared contexts, preserving order across groups and chunks."""
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        groups: dict[int, list[tuple[int, FloatArray, BoolArray]]] = defaultdict(list)
        for index, series in enumerate(batch):
            values, mask = left_pad_to_patch(series.values)
            groups[values.size].append((index, values, mask))
        results: dict[int, FloatArray] = {}
        with torch.inference_mode():
            for group in groups.values():
                for start in range(0, len(group), self._batch_size):
                    chunk = group[start : start + self._batch_size]
                    target = torch.tensor(
                        np.stack([item[1] for item in chunk]), dtype=torch.float32
                    )[:, None, :]
                    mask = torch.tensor(np.stack([item[2] for item in chunk]), dtype=torch.bool)[
                        :, None, :
                    ]
                    output = self._model.forecast(
                        {
                            "target": target,
                            "target_mask": mask,
                            "series_ids": torch.zeros((len(chunk), 1), dtype=torch.long),
                        },
                        horizon=horizon_steps,
                        decode_block_size=None,
                        has_missing_values=bool((~mask).any()),
                        scaler_fallback_min_obs=_SCALER_FALLBACK_MIN_OBS,
                        quantile_real_cap_k=_QUANTILE_REAL_CAP_K,
                    )
                    if output.shape != (_QUANTILE_COUNT, len(chunk), 1, horizon_steps):
                        raise ValueError(f"unexpected Toto forecast shape: {tuple(output.shape)}")
                    quantiles = output[:, :, 0, :].cpu().numpy().astype(np.float64)
                    for offset, (index, _, _) in enumerate(chunk):
                        result = quantiles[:, offset, :].copy()
                        # Correction must not conceal invalid model values from run_cycle.
                        results[index] = (
                            np.maximum.accumulate(result, axis=0)
                            if np.isfinite(result).all()
                            else result
                        )
        return [results[index] for index in range(len(batch))]


class FakeForecaster:
    """Deterministic last-value forecast with a square-root widening unit band."""

    name = "fake"

    def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
        """Center q50 on the final value and widen each adjacent quantile by sqrt(lead)."""
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        offsets = np.arange(-4, 5, dtype=np.float64)[:, None]
        widths = np.sqrt(np.arange(1, horizon_steps + 1, dtype=np.float64))
        return [series.values[-1] + offsets * widths for series in batch]
