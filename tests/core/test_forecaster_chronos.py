from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from promcast.core.forecaster import Forecaster
from promcast.core.forecaster_chronos import Chronos2Forecaster
from promcast.core.series import Series
from promcast.shell.cli import load_forecaster


class RecordingPipeline:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []
        self.bad_value: float | None = None
        self.malformation = ""

    def predict_quantiles(
        self, inputs: torch.Tensor, prediction_length: int, quantile_levels: list[float]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        assert torch.is_inference_mode_enabled()
        assert quantile_levels == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.inputs.append(inputs)
        offsets = torch.tensor([0, 2, 1, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
        outputs = [
            (row[:, -1, None, None] + offsets).repeat(1, prediction_length, 1) for row in inputs
        ]
        if self.bad_value is not None:
            outputs[0][0, 0, 4] = self.bad_value
        if self.malformation == "count":
            outputs.pop()
        if self.malformation == "shape":
            outputs[0] = outputs[0][..., :8]
        return outputs, [output[..., 4] for output in outputs]


@pytest.fixture
def recording_chronos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Forecaster, RecordingPipeline]]:
    pipeline = RecordingPipeline()

    class Loader:
        @staticmethod
        def from_pretrained(path: str, *, device_map: str, local_files_only: bool) -> object:
            assert path == str(tmp_path)
            assert device_map == "cpu"
            assert local_files_only
            return pipeline

    def import_chronos(name: str) -> SimpleNamespace:
        assert name == "chronos"
        return SimpleNamespace(BaseChronosPipeline=Loader)

    monkeypatch.setattr("promcast.core.forecaster_chronos.import_module", import_chronos)
    for filename in ("config.json", "model.safetensors"):
        (tmp_path / filename).touch()
    previous_threads = torch.get_num_threads()
    try:
        yield load_forecaster(tmp_path, "chronos-2-small", batch_size=2, threads=1), pipeline
    finally:
        torch.set_num_threads(previous_threads)


def test_chronos_groups_chunks_and_preserves_order_missingness_and_quantile_axis(
    recording_chronos: tuple[Forecaster, RecordingPipeline],
) -> None:
    forecaster, pipeline = recording_chronos
    batch: list[Series] = []
    for length, value in [(64, 5), (32, 6), (64, 7), (64, 8)]:
        values = np.full(length, value, dtype=np.float64)
        values[0] = np.nan
        batch.append(Series((), 60, 0, values))
    before = [series.values.copy() for series in batch]
    outputs = forecaster.forecast(batch, 37)
    assert forecaster.name == "chronos-2-small"
    assert [tuple(inputs.shape) for inputs in pipeline.inputs] == [
        (2, 1, 64),
        (1, 1, 64),
        (1, 1, 32),
    ]
    assert all(inputs.dtype == torch.float32 for inputs in pipeline.inputs)
    assert all(torch.isnan(inputs[..., 0]).all() for inputs in pipeline.inputs)
    for index, output in enumerate(outputs):
        assert output.shape == (9, 37)
        assert output.dtype == np.float64
        assert_array_equal(output[:, 0], np.array([0, 2, 2, 3, 4, 5, 6, 7, 8]) + index + 5)
        assert_array_equal(batch[index].values, before[index])
    assert forecaster.forecast([], 37) == []
    assert len(pipeline.inputs) == 3


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_chronos_preserves_noninitial_invalid_quantiles(
    recording_chronos: tuple[Forecaster, RecordingPipeline], bad_value: float
) -> None:
    forecaster, pipeline = recording_chronos
    pipeline.bad_value = bad_value
    result = forecaster.forecast([Series((), 60, 0, np.ones(64))], 37)[0]
    assert not np.isfinite(result[4, 0])


@pytest.mark.parametrize("malformation", ["count", "shape"])
def test_chronos_rejects_malformed_output(
    recording_chronos: tuple[Forecaster, RecordingPipeline], malformation: str
) -> None:
    forecaster, pipeline = recording_chronos
    pipeline.malformation = malformation
    with pytest.raises(ValueError, match=r"one output|shape"):
        forecaster.forecast([Series((), 60, 0, np.ones(64))], 37)


@pytest.mark.parametrize("options", [{"batch_size": 0}, {"threads": 0}])
def test_chronos_rejects_invalid_execution_settings(
    tmp_path: Path, options: dict[str, int]
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Chronos2Forecaster(tmp_path, **options)


def test_chronos_requires_positive_horizon(
    recording_chronos: tuple[Forecaster, RecordingPipeline],
) -> None:
    with pytest.raises(ValueError, match="horizon_steps"):
        recording_chronos[0].forecast([], 0)
