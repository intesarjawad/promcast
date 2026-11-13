from typing import Literal

import numpy as np
import pytest

from promcast.core.alerting import already_breached, quantile_row, time_to_breach_s


@pytest.mark.parametrize(
    ("direction", "row", "expected"),
    [
        ("above", [1, 2, 3], 20.0),
        ("below", [3, 2, 1], 20.0),
        ("above", [0, 1], None),
        ("below", [4, 3], None),
        ("above", [2], 10.0),
        ("below", [2], 10.0),
    ],
)
def test_first_inclusive_crossing(
    direction: Literal["above", "below"], row: list[float], expected: float | None
) -> None:
    assert time_to_breach_s(np.array(row, dtype=np.float64), 10, 2, direction) == expected


@pytest.mark.parametrize("direction", ["above", "below"])
def test_empty_row_never_crosses(direction: Literal["above", "below"]) -> None:
    assert time_to_breach_s(np.array([], dtype=np.float64), 10, 2, direction) is None


@pytest.mark.parametrize(
    ("direction", "value", "expected"),
    [
        ("above", 3, True),
        ("above", 2, True),
        ("above", 1, False),
        ("below", 1, True),
        ("below", 2, True),
        ("below", 3, False),
    ],
)
def test_already_breached(
    direction: Literal["above", "below"], value: float, expected: bool
) -> None:
    assert already_breached(value, 2, direction) is expected


@pytest.mark.parametrize("quantile", [0.5, 0.6, 0.7, 0.8, 0.9])
@pytest.mark.parametrize("direction", ["above", "below"])
def test_direction_selects_quantile_row(quantile: float, direction: Literal["above", "below"]):
    forecast = np.repeat(np.arange(9, dtype=np.float64)[:, None], 3, axis=1)
    offset = round((quantile - 0.5) * 10)
    index = 4 + offset if direction == "above" else 4 - offset
    np.testing.assert_array_equal(quantile_row(forecast, quantile, direction), forecast[index])


def test_quantile_row_rejects_unsupported_level():
    with pytest.raises(ValueError, match="quantile"):
        quantile_row(np.zeros((9, 2)), 0.95, "above")
