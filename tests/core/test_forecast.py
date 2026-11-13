import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from promcast.core.config import TargetConfig
from promcast.core.forecast import run_cycle
from promcast.core.forecaster import FakeForecaster
from promcast.core.series import FloatArray, Series, Skipped


@pytest.fixture
def target() -> TargetConfig:
    return TargetConfig.model_validate(
        {
            "name": "free_space",
            "query": "free_space",
            "step": "1m",
            "context": "128m",
            "horizon": "10m",
            "season": "1h",
            "direction": "below",
            "threshold": 0,
            "quantile": 0.9,
            "warn": "5m",
            "method": "toto",
            "linear_window": "10m",
            "max_series": 2,
        }
    )


def context(name: str, values: FloatArray | None = None, end_ts: float = 10000) -> Series:
    return Series(
        (("instance", name),),
        60,
        end_ts,
        np.arange(128, 0, -1, dtype=np.float64) if values is None else values,
    )


class RecordingForecaster(FakeForecaster):
    def __init__(self) -> None:
        self.batches: list[list[Series]] = []

    def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
        self.batches.append(batch)
        return super().forecast(batch, horizon_steps)


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        (np.array([], dtype=np.float64), "missing"),
        (np.full(128, np.nan), "missing"),
        (np.concatenate([np.full(65, np.nan), np.arange(63, 0, -1)]), "missing"),
        (np.concatenate([np.array([np.inf]), np.arange(127, 0, -1)]), "missing"),
        (np.arange(128, dtype=np.float64), "counter"),
        (np.ones(128), "counter"),
        (np.arange(63, 0, -1, dtype=np.float64), "too_short"),
        (
            np.concatenate([np.arange(96, 0, -1), np.full(9, np.nan), np.arange(23, 0, -1)]),
            "tail_missing",
        ),
    ],
)
def test_input_skip_reasons(target: TargetConfig, values: FloatArray, reason: str) -> None:
    model = RecordingForecaster()
    series = context("skipped", values)
    forecasts, skipped = run_cycle(target, [series], model, 10000)
    assert not forecasts
    assert [(item.labels, item.reason) for item in skipped] == [(series.labels, reason)]
    assert not model.batches


def test_batch_cap_follows_filtering_in_input_order(target: TargetConfig) -> None:
    model = RecordingForecaster()
    sources = [
        context("empty", np.array([])),
        context("z"),
        context("a"),
        context("b"),
        context("short", np.arange(63, 0, -1, dtype=np.float64)),
    ]
    forecasts, skipped = run_cycle(target, sources, model, 10000)
    assert skipped == [
        Skipped(sources[0].labels, "missing"),
        Skipped(sources[3].labels, "capped"),
        Skipped(sources[4].labels, "too_short"),
    ]
    assert len(model.batches) == 1
    assert [item.labels for item in model.batches[0]] == [sources[1].labels, sources[2].labels]
    assert [(item.labels, item.model) for item in forecasts] == [
        (source.labels, model_name) for source in sources[1:3] for model_name in ("fake", "linear")
    ]


def test_origin_tail_fill_missingness_and_linear_window(target: TargetConfig) -> None:
    values = np.arange(128, 0, -1, dtype=np.float64)
    values[10] = np.nan
    values[-5] = np.nan
    values[-2:] = np.nan
    original = values.copy()
    model = RecordingForecaster()
    forecasts, skipped = run_cycle(target, [context("node", values)], model, 10000)
    assert not skipped
    learned, baseline = forecasts
    assert learned.origin_ts == baseline.origin_ts == 9880
    assert learned.missing_fraction == baseline.missing_fraction == 2 / 126
    assert learned.naive_mae == baseline.naive_mae == 1
    assert learned.target == "free_space"
    assert learned.step_s == 60
    assert learned.quantiles.shape == (9, 10)
    assert_array_equal(learned.quantiles[4], np.full(10, 3))
    assert_allclose(baseline.quantiles, np.tile(np.arange(2, -8, -1), (9, 1)), atol=1e-12)
    prepared = model.batches[0][0].values
    assert prepared[-3] == 6
    assert np.isnan(prepared[10])
    assert not np.isnan(prepared[-32:]).any()
    assert_array_equal(values, original)


def test_now_cutoff_retains_exact_sample_and_discards_future_infinity(target: TargetConfig) -> None:
    values = np.arange(130, 0, -1, dtype=np.float64)
    values[-1] = np.inf
    forecasts, skipped = run_cycle(
        target, [context("node", values, 10120)], FakeForecaster(), 10000
    )
    assert not skipped
    assert forecasts[0].origin_ts == 10000
    assert_array_equal(forecasts[0].quantiles[4], np.full(10, 3))


@pytest.mark.parametrize("now_ts", [0, 9999, 9999.5, 10000])
def test_empty_after_trim_and_fractional_cutoff(target: TargetConfig, now_ts: float) -> None:
    series = context("node")
    forecasts, skipped = run_cycle(target, [series], FakeForecaster(), now_ts)
    if now_ts == 0:
        assert skipped == [Skipped(series.labels, "missing")]
        assert not forecasts
    else:
        assert not skipped
        assert forecasts[0].origin_ts == (9940 if now_ts < 10000 else 10000)


def test_exact_missing_and_observation_limits_are_accepted(target: TargetConfig) -> None:
    values = np.arange(128, 0, -1, dtype=np.float64)
    values[:64] = np.nan
    forecasts, skipped = run_cycle(target, [context("node", values)], FakeForecaster(), 10000)
    assert not skipped
    assert forecasts[0].missing_fraction == 0.5


def test_linear_method_uses_configured_window_without_calling_model(target: TargetConfig) -> None:
    target = target.model_copy(update={"method": "linear"})
    values = np.arange(128, 0, -1, dtype=np.float64)
    values[:-10] = 1000
    model = RecordingForecaster()
    forecasts, skipped = run_cycle(target, [context("node", values)], model, 10000)
    assert not skipped
    assert len(forecasts) == 1
    assert forecasts[0].model == "linear"
    assert_allclose(forecasts[0].quantiles, np.tile(np.arange(0, -10, -1), (9, 1)))
    assert not model.batches


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_model_output_is_skipped_per_series(
    target: TargetConfig, bad_value: float
) -> None:
    class NonfiniteForecaster(RecordingForecaster):
        def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
            output = super().forecast(batch, horizon_steps)
            output[0][0, 0] = bad_value
            return output

    sources = [context("bad"), context("good"), context("capped")]
    model = NonfiniteForecaster()
    forecasts, skipped = run_cycle(target, sources, model, 10000)
    assert skipped == [
        Skipped(sources[2].labels, "capped"),
        Skipped(sources[0].labels, "model_output"),
    ]
    assert len(model.batches) == 1
    assert [item.labels for item in model.batches[0]] == [item.labels for item in sources[:2]]
    assert len(forecasts) == 2
    assert all(item.labels == sources[1].labels for item in forecasts)


def test_empty_cycle_does_not_invoke_forecaster(target: TargetConfig) -> None:
    model = RecordingForecaster()
    assert run_cycle(target, [], model, 10000) == ([], [])
    assert not model.batches


def test_eight_tail_gaps_are_filled_but_still_count_as_missing(target: TargetConfig) -> None:
    values = np.arange(128, 0, -1, dtype=np.float64)
    values[-9:-1] = np.nan
    model = RecordingForecaster()
    forecasts, skipped = run_cycle(target, [context("node", values)], model, 10000)
    assert not skipped
    assert forecasts[0].missing_fraction == 8 / 128
    assert_array_equal(model.batches[0][0].values[-9:-1], np.full(8, 10))


def test_sparse_linear_window_is_too_short(target: TargetConfig) -> None:
    target = target.model_copy(update={"linear_window": target.step * 2})
    values = np.arange(128, 0, -1, dtype=np.float64)
    values[-2] = np.nan
    series = context("node", values)
    forecasts, skipped = run_cycle(target, [series], FakeForecaster(), 10000)
    assert not forecasts
    assert skipped == [Skipped(series.labels, "too_short")]


def test_150_second_linear_window_includes_three_60_second_slots(target: TargetConfig) -> None:
    target = target.model_copy(update={"linear_window": target.step * 2.5, "method": "linear"})
    values = np.arange(128, 0, -1, dtype=np.float64)
    values[-3:] = [10, 6, 6]
    forecasts, skipped = run_cycle(target, [context("node", values)], FakeForecaster(), 10000)
    assert not skipped
    assert_allclose(forecasts[0].quantiles[4, :2], [10 / 3, 4 / 3])


def test_output_quantiles_are_monotone(target: TargetConfig) -> None:
    class CrossingForecaster(FakeForecaster):
        def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
            return [output[::-1].copy() for output in super().forecast(batch, horizon_steps)]

    forecasts, skipped = run_cycle(target, [context("node")], CrossingForecaster(), 10000)
    assert not skipped
    assert np.all(np.diff(forecasts[0].quantiles, axis=0) >= 0)


@pytest.mark.parametrize("malformation", ["count", "shape"])
def test_forecaster_contract_errors_fail_loudly(target: TargetConfig, malformation: str) -> None:
    class MalformedForecaster(FakeForecaster):
        def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
            if malformation == "count":
                return []
            return [np.ones((8, horizon_steps)) for _ in batch]

    with pytest.raises(ValueError, match=r"one output|shape"):
        run_cycle(target, [context("node")], MalformedForecaster(), 10000)


@pytest.mark.parametrize("now_ts", [np.nan, np.inf, -np.inf])
def test_nonfinite_now_is_rejected(target: TargetConfig, now_ts: float) -> None:
    with pytest.raises(ValueError, match="now_ts must be finite"):
        run_cycle(target, [], FakeForecaster(), now_ts)
