from collections.abc import Callable, Iterator
from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest
import torch

pytestmark = pytest.mark.slow

PATCH_SIZE = 32
HORIZON = 37  # Cross a patch boundary and require truncation of the final patch.
MASK_TOLERANCE = 1e-6
FALLBACK_MIN_OBS = 8
REAL_CAP_K = 1e4


@runtime_checkable
class QuantileHead(Protocol):
    @property
    def knots(self) -> list[float]: ...


@runtime_checkable
class ForecastModel(Protocol):
    def get_submodule(self, target: str) -> object: ...

    def forecast(
        self,
        inputs: dict[str, torch.Tensor],
        horizon: int,
        *,
        decode_block_size: None,
        scaler_fallback_min_obs: int = 0,
        quantile_real_cap_k: float = 0,
    ) -> torch.Tensor: ...


@runtime_checkable
class ModelLoader(Protocol):
    def from_pretrained(
        self, path: str, *, local_files_only: bool, map_location: str
    ) -> object: ...


@pytest.fixture(scope="module")
def toto_model(toto_model_dir: Path) -> Iterator[ForecastModel]:
    local_dir = toto_model_dir
    # The model imports matplotlib; keep its incidental caches in temporary storage.
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("MPLCONFIGDIR", str(local_dir / "matplotlib"))
        environment.setenv("XDG_CACHE_HOME", str(local_dir / "cache"))
        loader: object = import_module("toto2").Toto2Model
    assert isinstance(loader, ModelLoader)
    model = loader.from_pretrained(str(local_dir), local_files_only=True, map_location="cpu")
    assert isinstance(model, ForecastModel)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield model
    finally:
        torch.set_num_threads(previous_threads)


def sine_context(length: int) -> torch.Tensor:
    steps = torch.arange(length, dtype=torch.float32)
    return (10 + torch.sin(steps / 5) + steps / 100).reshape(1, 1, length)


def inputs_for(target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "target": target,
        "target_mask": mask,
        "series_ids": torch.zeros(target.shape[:-1], dtype=torch.long),
    }


def assert_forecast(output: torch.Tensor, batch: int = 1) -> None:
    assert output.shape == (9, batch, 1, HORIZON)
    assert torch.isfinite(output).all().item()
    assert (output[1:] >= output[:-1]).all().item()


@pytest.mark.parametrize("batch", [1, 3])
def test_shape_sorted_and_deterministic(toto_model: ForecastModel, batch: int) -> None:
    head = toto_model.get_submodule("output_head")
    assert isinstance(head, QuantileHead)
    assert head.knots == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    target = sine_context(128).repeat(batch, 1, 1)
    target += torch.arange(batch).reshape(batch, 1, 1)
    inputs = inputs_for(target, torch.ones_like(target, dtype=torch.bool))
    first = toto_model.forecast(inputs, HORIZON, decode_block_size=None)
    second = toto_model.forecast(inputs, HORIZON, decode_block_size=None)
    assert_forecast(first, batch)
    assert torch.equal(first, second)


def test_last_five_masked_zeros_are_treated_as_observed(
    toto_model: ForecastModel, record_property: Callable[[str, object], None]
) -> None:
    target = sine_context(128)
    target[..., -5:] = 0
    observed = torch.ones_like(target, dtype=torch.bool)
    masked = observed.clone()
    masked[..., -5:] = False
    with_mask = toto_model.forecast(inputs_for(target, masked), HORIZON, decode_block_size=None)
    without_mask = toto_model.forecast(
        inputs_for(target, observed), HORIZON, decode_block_size=None
    )
    assert_forecast(with_mask)
    difference = (with_mask - without_mask).abs().max().item()
    record_property("tail_mask_max_difference", difference)
    assert difference < MASK_TOLERANCE, difference


@pytest.mark.parametrize("value", [0.0, 7.0])
def test_constant_series_is_finite(toto_model: ForecastModel, value: float) -> None:
    target = torch.full((1, 1, 128), value)
    output = toto_model.forecast(
        inputs_for(target, torch.ones_like(target, dtype=torch.bool)),
        HORIZON,
        decode_block_size=None,
    )
    assert_forecast(output)


def test_scaler_fallback_and_real_cap(
    toto_model: ForecastModel, record_property: Callable[[str, object], None]
) -> None:
    target = sine_context(128)
    target[..., :100] = 0
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., :100] = False
    inputs = inputs_for(target, mask)
    default = toto_model.forecast(inputs, HORIZON, decode_block_size=None)
    protected = toto_model.forecast(
        inputs,
        HORIZON,
        decode_block_size=None,
        scaler_fallback_min_obs=FALLBACK_MIN_OBS,
        quantile_real_cap_k=REAL_CAP_K,
    )
    assert_forecast(default)
    assert_forecast(protected)
    difference = (default - protected).abs().max().item()
    record_property("scaler_options_max_difference", difference)

    # The forced final patch includes four zeros despite the caller's mask.
    effective_mask = mask.clone()
    effective_mask[..., -PATCH_SIZE:] = True
    observed = target[effective_mask]
    scale = observed.double().std().float().clamp_min(1e-6)
    lower = observed.min() - REAL_CAP_K * scale
    upper = observed.max() + REAL_CAP_K * scale
    assert (protected >= lower).all().item()
    assert (protected <= upper).all().item()

    tight_k = 0.01
    tight = toto_model.forecast(
        inputs,
        HORIZON,
        decode_block_size=None,
        scaler_fallback_min_obs=FALLBACK_MIN_OBS,
        quantile_real_cap_k=tight_k,
    )
    fallback_only = toto_model.forecast(
        inputs,
        HORIZON,
        decode_block_size=None,
        scaler_fallback_min_obs=FALLBACK_MIN_OBS,
    )
    assert_forecast(tight)
    expected = fallback_only.clamp(
        observed.min() - tight_k * scale, observed.max() + tight_k * scale
    )
    assert torch.equal(tight, expected)
    assert not torch.equal(tight, fallback_only)


def test_scaler_fallback_changes_a_partially_observed_leading_patch(
    toto_model: ForecastModel, record_property: Callable[[str, object], None]
) -> None:
    target = sine_context(128)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., 4:100] = False
    target[~mask] = 0
    inputs = inputs_for(target, mask)
    default = toto_model.forecast(inputs, HORIZON, decode_block_size=None)
    fallback = toto_model.forecast(
        inputs,
        HORIZON,
        decode_block_size=None,
        scaler_fallback_min_obs=FALLBACK_MIN_OBS,
    )
    assert_forecast(default)
    assert_forecast(fallback)
    difference = (default - fallback).abs().max().item()
    record_property("partial_patch_fallback_max_difference", difference)
    assert difference > MASK_TOLERANCE, difference


@pytest.mark.parametrize("observed", [32, 33, 40, 63])
@pytest.mark.parametrize("padding_value", [-1e6, 17.0, 1e6])
def test_left_padding_is_ignored(
    toto_model: ForecastModel,
    observed: int,
    padding_value: float,
    record_property: Callable[[str, object], None],
) -> None:
    target = torch.zeros(1, 1, 64)
    target[..., -observed:] = sine_context(observed)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[..., -observed:] = True
    altered = target.clone()
    altered[~mask] = padding_value
    baseline = toto_model.forecast(
        inputs_for(target, mask),
        HORIZON,
        decode_block_size=None,
        scaler_fallback_min_obs=FALLBACK_MIN_OBS,
        quantile_real_cap_k=REAL_CAP_K,
    )
    changed = toto_model.forecast(
        inputs_for(altered, mask),
        HORIZON,
        decode_block_size=None,
        scaler_fallback_min_obs=FALLBACK_MIN_OBS,
        quantile_real_cap_k=REAL_CAP_K,
    )
    assert_forecast(baseline)
    assert_forecast(changed)
    difference = (baseline - changed).abs().max().item()
    record_property("padding_max_difference", difference)
    assert difference < MASK_TOLERANCE, difference
