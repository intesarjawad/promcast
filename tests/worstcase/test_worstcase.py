import asyncio

import httpx
import numpy as np
import pytest
import respx
from numpy.testing import assert_allclose, assert_array_equal

from promcast.core.config import TargetConfig, TSDBConfig
from promcast.core.forecast import PreparationPolicy, run_cycle
from promcast.core.forecaster import FakeForecaster
from promcast.core.series import FloatArray, Series, Skipped
from promcast.shell.tsdb import RangeGrid, TsdbError, range_query

_TSDB_URL = "https://prom.example/prometheus/api/v1/query_range"


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


def _range_query() -> dict[str, object]:
    config = TSDBConfig.model_validate({"url": "https://prom.example/prometheus/"})

    async def run() -> dict[str, object]:
        async with httpx.AsyncClient() as client:
            return await range_query(client, config, "up", RangeGrid(0, 60, 15))

    return asyncio.run(run())


def test_constant_series_is_a_counter_unless_semantics_rule_it_out(target: TargetConfig) -> None:
    series = context("constant", np.full(128, 0.25))
    forecasts, skipped = run_cycle(target, [series], FakeForecaster(), 10000)
    assert not forecasts
    assert skipped == [Skipped(series.labels, "counter")]

    forecasts, skipped = run_cycle(
        target, [series], FakeForecaster(), 10000, policy=PreparationPolicy(known_gauge=True)
    )
    assert not skipped
    learned, baseline = forecasts
    assert_array_equal(learned.quantiles[4], np.full(10, 0.25))
    assert_allclose(learned.quantiles[8], 0.25 + 4 * np.sqrt(np.arange(1, 11)))
    assert_allclose(baseline.quantiles, np.full((9, 10), 0.25), rtol=0, atol=1e-12)


def test_1e15_scale_bytes_keep_absolute_and_step_magnitude(target: TargetConfig) -> None:
    values = 1e15 - np.arange(128, dtype=np.float64) * 1e9
    forecasts, skipped = run_cycle(target, [context("bytes", values)], FakeForecaster(), 10000)
    assert not skipped
    learned, baseline = forecasts
    last = 1e15 - 127e9
    assert_array_equal(learned.quantiles[4], np.full(10, last))
    assert_allclose(learned.quantiles[8], last + 4 * np.sqrt(np.arange(1, 11)), rtol=0, atol=0.5)
    assert learned.naive_mae == pytest.approx(1e9)
    assert_allclose(baseline.quantiles[4, :2], [last - 1e9, last - 2e9], rtol=0, atol=2.0)


def test_negative_gauge_keeps_sign_and_slope(target: TargetConfig) -> None:
    values = -10.0 - 0.5 * np.arange(128, dtype=np.float64)
    forecasts, skipped = run_cycle(target, [context("celsius", values)], FakeForecaster(), 10000)
    assert not skipped
    learned, baseline = forecasts
    assert_array_equal(learned.quantiles[4], np.full(10, -73.5))
    assert_allclose(learned.quantiles[8], -73.5 + 4 * np.sqrt(np.arange(1, 11)))
    assert learned.naive_mae == 0.5
    assert_allclose(baseline.quantiles[4, 0], -74.0, rtol=0, atol=1e-12)


def test_just_under_half_missing_is_accepted(target: TargetConfig) -> None:
    values = np.concatenate([np.full(63, np.nan), 1000.0 - np.arange(65, dtype=np.float64)])
    forecasts, skipped = run_cycle(target, [context("gappy", values)], FakeForecaster(), 10000)
    assert not skipped
    learned, _ = forecasts
    assert learned.missing_fraction == 63 / 128
    assert learned.origin_ts == 10000
    assert_array_equal(learned.quantiles[4], np.full(10, 936.0))


def test_just_over_half_missing_is_skipped_before_observation_count(target: TargetConfig) -> None:
    values = np.concatenate([np.full(65, np.nan), 1000.0 - np.arange(63, dtype=np.float64)])
    series = context("sparse", values)
    forecasts, skipped = run_cycle(target, [series], FakeForecaster(), 10000)
    assert not forecasts
    assert skipped == [Skipped(series.labels, "missing")]


def test_all_nan_tail_trims_origin_without_counting_as_missing(target: TargetConfig) -> None:
    values = np.concatenate([np.arange(96, 0, -1, dtype=np.float64), np.full(32, np.nan)])
    forecasts, skipped = run_cycle(target, [context("tail", values)], FakeForecaster(), 10000)
    assert not skipped
    learned, baseline = forecasts
    assert learned.origin_ts == baseline.origin_ts == 8080
    assert learned.missing_fraction == 0
    assert_array_equal(learned.quantiles[4], np.full(10, 1.0))


def test_single_sample_is_never_forecast(target: TargetConfig) -> None:
    series = context("single", np.array([5.0]))
    forecasts, skipped = run_cycle(target, [series], FakeForecaster(), 10000)
    assert not forecasts
    assert skipped == [Skipped(series.labels, "counter")]

    forecasts, skipped = run_cycle(
        target, [series], FakeForecaster(), 10000, policy=PreparationPolicy(known_gauge=True)
    )
    assert not forecasts
    assert skipped == [Skipped(series.labels, "too_short")]


def test_label_churn_between_cycles_is_not_carried_over(target: TargetConfig) -> None:
    model = FakeForecaster()
    first = [context("a"), context("b")]
    forecasts, skipped = run_cycle(target, first, model, 10000)
    assert not skipped
    assert [(item.labels, item.model) for item in forecasts] == [
        (source.labels, name) for source in first for name in ("fake", "linear")
    ]

    second = [context("b"), context("c", np.arange(10, 0, -1, dtype=np.float64))]
    forecasts, skipped = run_cycle(target, second, model, 10000)
    assert skipped == [Skipped(second[1].labels, "too_short")]
    assert [item.labels for item in forecasts] == [second[0].labels] * 2
    assert all(item.labels != first[0].labels for item in forecasts)


def test_five_hundred_series_cap_keeps_input_order(target: TargetConfig) -> None:
    target = target.model_copy(update={"max_series": 500})
    sources = [context(f"node-{index:03d}") for index in range(501)]
    forecasts, skipped = run_cycle(target, sources, FakeForecaster(), 10000)
    assert skipped == [Skipped(sources[-1].labels, "capped")]
    assert len(forecasts) == 1000
    assert [item.labels for item in forecasts[::2]] == [source.labels for source in sources[:500]]
    assert [item.model for item in forecasts[:2]] == ["fake", "linear"]
    assert forecasts[-1].labels == sources[499].labels


def test_vector_response_is_not_accepted_as_a_matrix() -> None:
    body = {
        "status": "success",
        "data": {"resultType": "vector", "result": [{"metric": {"job": "a"}, "value": [0, "1"]}]},
    }
    with respx.mock as router:
        router.get(_TSDB_URL).respond(200, json=body)
        with pytest.raises(TsdbError, match="invalid query_range response") as caught:
            _range_query()
    assert caught.value.status == 200
    assert caught.value.body.startswith("invalid query_range response: ")
    assert '"resultType":"vector"' in caught.value.body


def test_html_502_is_reported_with_a_bounded_body() -> None:
    html = "<html><body><h1>502 Bad Gateway</h1></body></html>" + "x" * 1000
    with respx.mock as router:
        router.get(_TSDB_URL).respond(502, text=html)
        with pytest.raises(TsdbError) as caught:
            _range_query()
    assert caught.value.status == 502
    assert str(caught.value).startswith("TSDB returned HTTP 502: <html>")
    assert caught.value.body == html[:512] + "..."
    assert len(caught.value.body) == 515


def test_future_samples_are_discarded_by_the_clock_cutoff(target: TargetConfig) -> None:
    values = np.arange(130, 0, -1, dtype=np.float64)
    future = context("future", values, 20000)
    forecasts, skipped = run_cycle(target, [future], FakeForecaster(), 10000)
    assert not forecasts
    assert skipped == [Skipped(future.labels, "missing")]

    skewed = context("skewed", values, 10120)
    forecasts, skipped = run_cycle(target, [skewed], FakeForecaster(), 10000)
    assert not skipped
    assert forecasts[0].origin_ts == 10000
    assert forecasts[0].missing_fraction == 0
    assert_array_equal(forecasts[0].quantiles[4], np.full(10, 3.0))
