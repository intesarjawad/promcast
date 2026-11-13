import hashlib
import json
from collections.abc import Callable, Iterator
from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pytest
import torch

from promcast.core.forecaster_chronos import Chronos2Forecaster
from promcast.core.series import Series

pytestmark = pytest.mark.slow

CONTEXT_LENGTH = 512
HORIZON = 37  # Cross the 16-step output patch boundary.
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WEIGHTS_SHA256 = "492290ae82bb89f9769e3479ce90b3179de1f33e600c34daa0352531538b23cd"
CONFIG_SHA256 = "885ee12976bd37d5f853027102a7f5b893675cd22725e2455c3cba7e9f0e4d40"


@runtime_checkable
class QuantilePipeline(Protocol):
    @property
    def quantiles(self) -> list[float]: ...

    @property
    def model(self) -> torch.nn.Module: ...

    def predict_quantiles(
        self,
        inputs: torch.Tensor,
        prediction_length: int,
        quantile_levels: list[float],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]: ...


@runtime_checkable
class PipelineLoader(Protocol):
    def from_pretrained(self, path: str, *, device_map: str, local_files_only: bool) -> object: ...


@pytest.fixture(scope="module")
def chronos_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    weights = Path.home() / ".cache/promcast/models/chronos-2-small/model.safetensors"
    with weights.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == WEIGHTS_SHA256

    # Exact configuration from the section 6 revision, recorded in ADR 0010.
    config = {
        "architectures": ["Chronos2Model"],
        "chronos_config": {
            "context_length": 8192,
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "max_output_patches": 64,
            "output_patch_size": 16,
            "quantiles": [0.01, 0.05, *QUANTILE_LEVELS, 0.95, 0.99],
            "time_encoding_scale": 8192,
            "use_arcsinh": True,
            "use_reg_token": True,
        },
        "chronos_pipeline_class": "Chronos2Pipeline",
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": 512,
        "dense_act_fn": "relu",
        "dropout_rate": 0.1,
        "feed_forward_proj": "relu",
        "initializer_factor": 0.05,
        "is_gated_act": False,
        "layer_norm_epsilon": 1e-6,
        "model_type": "t5",
        "num_heads": 8,
        "num_layers": 6,
        "pad_token_id": 0,
        "reg_token_id": 1,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "transformers_version": "4.49.0",
        "use_cache": True,
        "vocab_size": 2,
    }
    canonical_config = json.dumps(config, sort_keys=True).encode("utf-8")
    assert hashlib.sha256(canonical_config).hexdigest() == CONFIG_SHA256
    local_dir = tmp_path_factory.mktemp("chronos-2-small")
    (local_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (local_dir / "model.safetensors").symlink_to(weights)
    return local_dir


@pytest.fixture(scope="module")
def chronos_pipeline(chronos_model_dir: Path) -> Iterator[QuantilePipeline]:
    local_dir = chronos_model_dir
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("HF_HUB_OFFLINE", "1")
        environment.setenv("MPLCONFIGDIR", str(local_dir / "matplotlib"))
        environment.setenv("XDG_CACHE_HOME", str(local_dir / "cache"))
        loader: object = import_module("chronos").BaseChronosPipeline
        assert isinstance(loader, PipelineLoader)
        pipeline = loader.from_pretrained(str(local_dir), device_map="cpu", local_files_only=True)
        assert isinstance(pipeline, QuantilePipeline)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            yield pipeline
        finally:
            torch.set_num_threads(previous_threads)


def test_chronos_adapter_matches_verified_pipeline(
    chronos_model_dir: Path, chronos_pipeline: QuantilePipeline
) -> None:
    batch = [
        Series((), 60, 0, np.sin(np.arange(CONTEXT_LENGTH, dtype=np.float64) / 5) + offset)
        for offset in range(3)
    ]
    inputs = torch.tensor(np.stack([series.values for series in batch]), dtype=torch.float32)[
        :, None, :
    ]
    with torch.inference_mode():
        expected, _ = chronos_pipeline.predict_quantiles(inputs, HORIZON, QUANTILE_LEVELS)
    previous_threads = torch.get_num_threads()
    try:
        forecaster = Chronos2Forecaster(chronos_model_dir, batch_size=2, threads=1)
        actual = forecaster.forecast(batch, HORIZON)
    finally:
        torch.set_num_threads(previous_threads)
    for output, raw in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            output, np.maximum.accumulate(raw[0].numpy().T, axis=0), atol=1e-3
        )


@pytest.mark.parametrize("batch", [1, 3])
def test_sine_quantiles_are_finite_monotone_and_deterministic(
    chronos_pipeline: QuantilePipeline,
    batch: int,
    record_property: Callable[[str, object], None],
) -> None:
    assert not chronos_pipeline.model.training
    assert set(QUANTILE_LEVELS).issubset(chronos_pipeline.quantiles)
    steps = torch.arange(CONTEXT_LENGTH, dtype=torch.float32)
    inputs = torch.sin(steps / 5).reshape(1, 1, CONTEXT_LENGTH).repeat(batch, 1, 1)
    inputs += torch.arange(batch).reshape(batch, 1, 1)
    with torch.inference_mode():
        first, first_medians = chronos_pipeline.predict_quantiles(
            inputs, prediction_length=HORIZON, quantile_levels=QUANTILE_LEVELS
        )
        second, second_medians = chronos_pipeline.predict_quantiles(
            inputs, prediction_length=HORIZON, quantile_levels=QUANTILE_LEVELS
        )
    assert len(first) == len(second) == len(first_medians) == len(second_medians) == batch
    max_correction = 0.0
    for quantiles, repeated, median, repeated_median in zip(
        first, second, first_medians, second_medians, strict=True
    ):
        assert quantiles.shape == (1, HORIZON, len(QUANTILE_LEVELS))
        assert median.shape == (1, HORIZON)
        assert quantiles.device.type == median.device.type == "cpu"
        assert quantiles.dtype == median.dtype == torch.float32
        assert torch.isfinite(quantiles).all().item()
        assert torch.isfinite(median).all().item()
        assert torch.equal(quantiles, repeated)
        assert torch.equal(median, repeated_median)
        assert torch.equal(median, quantiles[..., QUANTILE_LEVELS.index(0.5)])
        values: npt.NDArray[np.float64] = quantiles.double().numpy()
        monotone = np.maximum.accumulate(values, axis=-1)
        assert np.isfinite(monotone).all()
        assert (np.diff(monotone, axis=-1) >= 0).all()
        max_correction = max(max_correction, float(np.max(monotone - values)))
    record_property("quantile_max_correction", max_correction)
