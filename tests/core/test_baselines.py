from typing import Literal

import augurs
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from promcast.core.alerting import time_to_breach_s
from promcast.core.baselines import ets, linear, seasonal_naive, static_margin


def test_linear_exact_line_and_window():
    values = np.array([999, 999, 5, 7, 9, 11], dtype=np.float64)
    result = linear(values, step_s=10, horizon_steps=3, window_steps=4)
    assert result.shape == (9, 3)
    assert_allclose(result, np.tile([13, 15, 17], (9, 1)), atol=1e-12)


def test_linear_retains_elapsed_time_across_irregular_gaps():
    # Observations at -40, -30 and 0 seconds have mean x=-70/3, mean y=4.
    # Centered sums are Sxy=150 and Sxx=2600/3, giving slope 9/52.
    values = np.array([1, 3, np.nan, np.nan, 8], dtype=np.float64)
    expected = 4 + (np.array([10, 20, 30]) + 70 / 3) * (9 / 52)
    assert_allclose(linear(values, 10, 3, 5), np.tile(expected, (9, 1)))


def test_seasonal_naive_repeats_last_complete_period():
    history = np.array([999, 1, 3, 2, 1, 3, 2], dtype=np.float64)
    assert_array_equal(seasonal_naive(history, 10, 8, 3), np.tile([1, 3, 2, 1, 3, 2, 1, 3], (9, 1)))


def test_seasonal_naive_short_history_uses_last_observation():
    history = np.array([1, 7, np.nan], dtype=np.float64)
    assert_array_equal(seasonal_naive(history, 10, 4, 5), np.full((9, 4), 7.0))


@pytest.mark.parametrize("direction", ["above", "below"])
@pytest.mark.parametrize(("distance", "crosses"), [(1.9, True), (2, True), (2.1, False)])
def test_static_margin_crosses_first_step_exactly_within_margin(
    direction: Literal["above", "below"], distance: float, crosses: bool
) -> None:
    last = 10 - distance if direction == "above" else 10 + distance
    result = static_margin(np.array([last, np.nan]), 3, 2, direction)
    assert result.shape == (9, 3)
    assert_array_equal(result, np.full((9, 3), last + (2 if direction == "above" else -2)))
    assert time_to_breach_s(result[4], 10, 10, direction) == (10.0 if crosses else None)


def test_ets_sine_accuracy_and_quantile_interpolation():
    period = 60
    values = 10 + np.sin(2 * np.pi * np.arange(240) / period)
    original = values.copy()
    result = ets(values, 10, 60, period)
    expected = 10 + np.sin(2 * np.pi * np.arange(240, 300) / period)
    assert result.shape == (9, 60)
    assert np.mean(np.abs(result[4] - expected)) < 0.05
    assert np.all(np.isfinite(result))
    assert np.all(np.diff(result, axis=0) >= 0)
    for index in range(5):
        assert_allclose(result[index], result[0] + index / 4 * (result[4] - result[0]))
        assert_allclose(result[index + 4], result[4] + index / 4 * (result[8] - result[4]))
    assert_array_equal(values, original)


def test_ets_interpolates_gaps_without_compressing_time():
    values = 10 + np.sin(2 * np.pi * np.arange(180) / 60)
    values[80:83] = np.nan
    original = values.copy()
    filled = values.copy()
    filled[80:83] = np.interp([80, 81, 82], [79, 83], values[[79, 83]])
    assert_allclose(ets(values, 10, 12, 60), ets(filled, 10, 12, 60))
    assert_array_equal(values, original)


def test_ets_uses_only_last_three_seasons():
    recent = 10 + np.sin(2 * np.pi * np.arange(180) / 60)
    values = np.concatenate([np.full(60, 999.0), recent])
    assert_allclose(ets(values, 10, 12, 60), ets(recent, 10, 12, 60))


def test_ets_short_history_has_finite_nonseasonal_forecast():
    result = ets(np.arange(1, 21, dtype=np.float64), 10, 8, 60)
    assert result.shape == (9, 8)
    assert np.all(np.isfinite(result))
    assert np.all(np.diff(result, axis=0) >= 0)


@pytest.mark.parametrize("history_steps", [60, 119, 120])
@pytest.mark.parametrize("leading_missing_steps", [0, 60])
def test_ets_requires_two_full_seasons_for_mstl(history_steps: int, leading_missing_steps: int):
    period = 60
    horizon = 12
    values = 10 + np.sin(2 * np.pi * np.arange(history_steps) / period)
    history = np.concatenate([np.full(leading_missing_steps, np.nan), values])
    expected_model = (
        augurs.AutoETS(1, "ZZN") if history_steps < 2 * period else augurs.MSTL.ets([period])
    )
    expected_model.fit(values)
    expected = expected_model.predict(horizon, level=0.8)
    lower, upper = expected.lower(), expected.upper()
    assert lower is not None
    assert upper is not None

    result = ets(history, 10, horizon, period)

    assert result.shape == (9, horizon)
    assert np.all(np.isfinite(result))
    assert np.all(np.diff(result, axis=0) >= 0)
    assert_allclose(result[0], lower)
    assert_allclose(result[4], expected.point())
    assert_allclose(result[8], upper)


def test_linear_requires_two_observations_in_selected_window():
    with pytest.raises(ValueError, match="two observed"):
        linear(np.array([1, 2, np.nan, 3]), 10, 2, 2)
