from dataclasses import replace

import pytest

from promcast.core.operating import (
    MethodScore,
    OperatingPoint,
    choose_operating_point,
    operating_points,
    pool_scores,
)


def test_selection_prioritizes_budget_recall_lead_then_smaller_warning() -> None:
    points = [OperatingPoint("linear", w) for w in (600, 1200, 1800, 2700)]
    base = MethodScore(10, 5, 0, 5, 5, 1, [100])
    scores = {
        points[0]: base,
        points[1]: replace(base, leads_s=[200]),
        points[2]: replace(base, leads_s=[200]),
        points[3]: replace(base, caught_early=10, n_episodes=20),
    }
    assert choose_operating_point(scores, 1) == points[1]
    scores[points[0]] = replace(base, caught_early=6)
    assert choose_operating_point(scores, 1) == points[0]


def test_pooled_zero_incidents_minimizes_false_rate_then_warning() -> None:
    points = [OperatingPoint("linear", w) for w in (600, 1200, 1800)]
    quiet = MethodScore(0, 0, 0, 0, 0, 2)
    noisy = replace(quiet, n_episodes=2)
    scores = {
        points[0]: pool_scores([noisy, quiet]),
        points[1]: pool_scores([quiet, quiet]),
        points[2]: pool_scores([quiet, quiet]),
    }
    assert choose_operating_point(scores, 1) == points[1]
    assert scores[points[0]].false_alerts_per_week == 0.5
    assert scores[points[0]].n_series == 2
    assert scores[points[0]].weeks == 4


def test_pooling_keeps_no_incident_series_exposure_and_false_alerts() -> None:
    pooled = pool_scores(
        [MethodScore(1, 1, 0, 1, 1, 1, [60], [], [60]), MethodScore(0, 0, 0, 10, 0, 2)]
    )
    assert pooled.recall == 1
    assert pooled.precision == 1 / 11
    assert pooled.false_alerts_per_week == 10 / 3
    assert pooled.warning_times_s == [60]


def test_no_feasible_point_is_explicit() -> None:
    with pytest.raises(ValueError, match="no calibration operating point"):
        choose_operating_point({OperatingPoint("linear", 600): MethodScore(1, 0, 0, 10, 0, 1)}, 1)


def test_protocol_sweep_has_all_windows_quantiles_and_margins() -> None:
    linear = operating_points("linear", 7200)
    assert {p.window_s for p in linear} == {3600, 21600, 86400}
    assert {p.warn_s for p in linear} == {600, 1200, 1800, 2700, 3600, 5400, 7200}
    assert {p.quantile for p in operating_points("toto-2.0-22m", 7200)} == {0.5, 0.7, 0.9}
    assert {p.margin_k for p in operating_points("static-margin", 7200)} == {0.5, 1, 2, 3}
    assert {p.warn_s for p in operating_points("linear", 300)} == {300}
