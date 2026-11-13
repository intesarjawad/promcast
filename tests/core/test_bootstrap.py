import numpy as np
import pytest
from numpy.testing import assert_array_equal

from promcast.core.bootstrap import bootstrap_ci, decision_rule, paired_warning_difference


def test_seed_zero_cluster_bootstrap_is_deterministic_and_wider_for_correlated_series() -> None:
    correlated = [np.ones(10)] + [np.zeros(1) for _ in range(9)]
    balanced = [np.array([1.0, 0.0]) for _ in range(10)]
    first = bootstrap_ci(correlated, lambda x: float(np.mean(x)))
    second = bootstrap_ci(correlated, lambda x: float(np.mean(x)))
    narrow = bootstrap_ci(balanced, lambda x: float(np.mean(x)))
    assert first == second
    assert first[1] - first[0] > narrow[1] - narrow[0]
    assert narrow == (0.5, 0.5)


def test_missed_incidents_are_zero_in_paired_warning_time() -> None:
    differences = paired_warning_difference(np.array([120.0, 0, 60]), np.array([0.0, 180, 60]))
    assert_array_equal(differences, [120, -180, 0])
    assert float(np.mean(differences)) == -20
    with pytest.raises(ValueError, match="zero for misses"):
        paired_warning_difference(np.array([np.nan]), np.array([0.0]))


@pytest.mark.parametrize(
    ("interval", "rate", "expected"),
    [
        ((1, 10), 1.5, "beats"),
        ((-1, 1), 0, "ties"),
        ((0, 1), 2, "ties"),
        ((-10, -1), 0, "loses"),
        ((1, 10), 1.51, "loses"),
    ],
)
def test_preregistered_decision_rule(
    interval: tuple[float, float], rate: float, expected: str
) -> None:
    assert decision_rule(interval, rate, 1) == expected


def test_bootstrap_keeps_empty_series_and_rejects_undefined_inputs() -> None:
    assert bootstrap_ci([np.array([]), np.ones(2)], lambda x: float(np.median(x))) == (1, 1)
    with pytest.raises(ValueError, match="requires incidents"):
        bootstrap_ci([np.array([])], lambda x: float(np.mean(x)))
