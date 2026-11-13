from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal

from promcast.core.config import parse_config
from promcast.core.forecast import run_cycle
from promcast.core.forecaster import FakeForecaster, Forecaster, Toto2Forecaster
from promcast.core.series import Series, Skipped


@pytest.fixture(autouse=True)
def restore_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    yield
    torch.set_num_threads(previous)


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, torch.Tensor]] = []
        self.options: list[dict[str, object]] = []

    def forecast(
        self, inputs: dict[str, torch.Tensor], horizon: int, **options: object
    ) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        self.calls.append(inputs)
        self.options.append(options)
        # Deliberately unordered quantiles exercise the wrapper's monotonicity guard.
        offsets = torch.tensor([0, 2, 1, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
        centers = inputs["target"][:, :, -1]
        return (offsets[:, None, None, None] + centers[None, :, :, None]).repeat(1, 1, 1, horizon)


@pytest.fixture
def recording_forecaster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Toto2Forecaster, RecordingModel]:
    model = RecordingModel()

    class Loader:
        @staticmethod
        def from_pretrained(path: str, *, local_files_only: bool, map_location: str) -> object:
            assert path == str(tmp_path)
            assert local_files_only
            assert map_location == "cpu"
            return model

    def import_toto(name: str) -> SimpleNamespace:
        assert name == "toto2"
        return SimpleNamespace(Toto2Model=Loader)

    monkeypatch.setattr("promcast.core.forecaster.import_module", import_toto)
    for filename in ("config.json", "model.safetensors"):
        (tmp_path / filename).touch()
    return Toto2Forecaster(tmp_path, "test", batch_size=2, threads=1), model


def context(length: int, value: float) -> Series:
    values = np.full(length, value, dtype=np.float64)
    if length > 32:
        values[0] = np.nan
    return Series((), 60, 0, values)


def test_grouping_chunking_masking_order_and_quantile_guard(
    recording_forecaster: tuple[Toto2Forecaster, RecordingModel],
) -> None:
    forecaster, model = recording_forecaster
    batch = [context(length, value) for length, value in [(65, 5), (32, 6), (70, 7), (96, 8)]]
    before = [series.values.copy() for series in batch]
    output = forecaster.forecast(batch, 37)
    assert forecaster.name == "test"
    assert [tuple(call["target"].shape) for call in model.calls] == [
        (2, 1, 96),
        (1, 1, 96),
        (1, 1, 64),
    ]
    for call, options in zip(model.calls, model.options, strict=True):
        assert call["target"].dtype == torch.float32
        assert call["target_mask"].dtype == torch.bool
        assert call["series_ids"].dtype == torch.long
        assert torch.count_nonzero(call["series_ids"]) == 0
        assert torch.all(call["target"][~call["target_mask"]] == 0)
        assert torch.all(call["target_mask"][..., -32:])
        assert options == {
            "decode_block_size": None,
            "has_missing_values": True,
            "scaler_fallback_min_obs": 8,
            "quantile_real_cap_k": 1e4,
        }
    for index, quantiles in enumerate(output):
        assert quantiles.shape == (9, 37)
        assert quantiles.dtype == np.float64
        assert_array_equal(quantiles[:, 0], np.array([0, 2, 2, 3, 4, 5, 6, 7, 8]) + index + 5)
        assert_array_equal(batch[index].values, before[index])


def test_fully_observed_context_disables_missing_path(
    recording_forecaster: tuple[Toto2Forecaster, RecordingModel],
) -> None:
    forecaster, model = recording_forecaster
    forecaster.forecast([Series((), 60, 0, np.ones(64))], 1)
    assert model.options[0]["has_missing_values"] is False


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_noninitial_invalid_quantile_survives_guard_and_is_rejected(
    recording_forecaster: tuple[Toto2Forecaster, RecordingModel],
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    forecaster, model = recording_forecaster
    original_forecast = model.forecast

    def invalid_forecast(
        inputs: dict[str, torch.Tensor], horizon: int, **options: object
    ) -> torch.Tensor:
        output = original_forecast(inputs, horizon, **options)
        output[4, 0, 0, 0] = bad_value
        return output

    monkeypatch.setattr(model, "forecast", invalid_forecast)
    example = Path(__file__).parents[1] / "fixtures/config.yaml"
    target = parse_config(example.read_text(encoding="utf-8"), environment={}).targets[0]
    series = Series((), 300, 0, np.arange(128, 0, -1, dtype=np.float64))
    output = forecaster.forecast([series], 1)[0]
    assert not np.isfinite(output[4, 0])
    assert run_cycle(target, [series], forecaster, 0) == ([], [Skipped((), "model_output")])


def test_empty_batch_does_not_call_model(
    recording_forecaster: tuple[Toto2Forecaster, RecordingModel],
) -> None:
    forecaster, model = recording_forecaster
    assert forecaster.forecast([], 37) == []
    assert not model.calls


@pytest.mark.parametrize("threads", [None, 3])
def test_thread_configuration(
    recording_forecaster: tuple[Toto2Forecaster, RecordingModel],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    threads: int | None,
) -> None:
    monkeypatch.setattr("promcast.core.forecaster.os.cpu_count", lambda: 8)
    Toto2Forecaster(tmp_path, "test", threads=threads)
    assert torch.get_num_threads() == (4 if threads is None else threads)


@pytest.mark.parametrize("options", [{"batch_size": 0}, {"threads": 0}, {"threads": -1}])
def test_invalid_execution_settings(tmp_path: Path, options: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Toto2Forecaster(tmp_path, "test", **options)


def test_local_configuration_is_required(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        Toto2Forecaster(tmp_path, "test")


def test_fake_forecaster_has_constant_median_and_widening_ordered_band() -> None:
    forecaster: Forecaster = FakeForecaster()
    output = forecaster.forecast([context(64, 7), context(32, -2)], 12)
    assert forecaster.name == "fake"
    for result, value in zip(output, [7, -2], strict=True):
        assert result.shape == (9, 12)
        assert_array_equal(result[4], np.full(12, value))
        assert np.all(np.diff(result, axis=0) > 0)
        assert np.all(np.diff(result[-1] - result[0]) > 0)
    assert forecaster.forecast([], 12) == []


def test_forecasters_reject_nonpositive_horizon(
    recording_forecaster: tuple[Toto2Forecaster, RecordingModel],
) -> None:
    for forecaster in (FakeForecaster(), recording_forecaster[0]):
        with pytest.raises(ValueError, match="horizon_steps"):
            forecaster.forecast([], 0)


@pytest.mark.slow
@pytest.mark.parametrize("toto_model_dir", ["toto-2.0-22m"], indirect=True)
def test_toto22m_sine_golden(toto_model_dir: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures/golden/toto22m_sine.npz"
    with np.load(fixture) as golden:
        values = np.asarray(golden["values"], dtype=np.float64)
        expected = np.asarray(golden["quantiles"], dtype=np.float64)
    assert values.shape == (512,)
    forecaster: Forecaster = Toto2Forecaster(toto_model_dir, "toto-2.0-22m", threads=1)
    output = forecaster.forecast([Series((), 60, 0, values)], 96)[0]
    assert output.shape == (9, 96)
    assert np.isfinite(output).all()
    assert np.all(np.diff(output, axis=0) >= 0)
    assert_allclose(output, expected, atol=1e-3, rtol=0)


@pytest.mark.slow
def test_real_model_grouped_batches_match_individual_forecasts(toto_model_dir: Path) -> None:
    forecaster = Toto2Forecaster(toto_model_dir, "test", batch_size=2, threads=1)
    batch = [context(length, value) for length, value in [(65, 5), (32, 6), (70, 7), (96, 8)]]
    grouped = forecaster.forecast(batch, 37)
    for series, actual in zip(batch, grouped, strict=True):
        expected = forecaster.forecast([series], 37)[0]
        assert_allclose(actual, expected, atol=1e-3, rtol=0)
