from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest

from promcast.core.forecast import Forecast
from promcast.core.selfaudit import SelfAudit
from promcast.core.series import Labels, Series

LABELS = (("instance", "a"),)


def forecast(
    origin: float = 0, model: str = "toto", labels: Labels = LABELS, scale: float = 2
) -> Forecast:
    quantiles = np.repeat(np.arange(1, 10, dtype=np.float64)[:, None], 120, axis=1)
    return Forecast("space", labels, origin, 60, quantiles, 0, model, scale)


@pytest.mark.parametrize("offset", [-30, 0, 30])
@pytest.mark.parametrize("actual", [1, 9, 10])
def test_half_step_inclusive_hits_and_original_error_scale(offset: int, actual: float):
    audit = SelfAudit([600], 86400)
    audit.record(forecast())
    assert audit.score(569, actual, LABELS) == []
    assert audit.score(600, actual, ()) == []
    points = audit.score(600 + offset, actual, LABELS)
    assert len(points) == 1
    assert points[0].covered == (actual <= 9)
    assert points[0].scaled_error == abs(actual - 5) / 2
    assert audit.score(600 + offset, actual, LABELS) == []
    audit.record(forecast())
    assert audit.score(600, actual, LABELS) == []


def test_pooling_methods_missing_values_and_undefined_scale():
    audit = SelfAudit([600, 7200], 86400)
    audit.record(forecast(scale=0))
    audit.record(forecast(model="linear"))
    audit.record(forecast(labels=()))
    for missing in (np.nan, np.inf, -np.inf):
        assert audit.score(600, missing, LABELS) == []
    audit.score(600, 10, LABELS)
    audit.score(600, 5, ())
    linear, toto = audit.summaries(600)
    assert linear.samples == 1
    assert linear.coverage_ratio is None
    assert linear.mase == 2.5
    assert toto.samples == 2
    assert toto.coverage_ratio == 0.5
    assert toto.mase == 0
    assert audit.summaries(600 + 86400)[1].samples == 2
    assert audit.summaries(601 + 86400) == ()
    audit.score(7200, 5, LABELS)
    assert len(audit.summaries(601 + 86400)) == 2
    audit.summaries(86401 + 7200)
    assert audit.score(7200, 5, LABELS) == []


def test_history_catches_up_mature_leads_without_future_leakage():
    audit = SelfAudit([600, 1800], 86400)
    audit.record(forecast())
    values = np.full(31, 5.0)
    values[10] = np.nan
    audit.score_series(Series(LABELS, 60, 1800, values))
    assert [point.lead_s for point in audit.summaries(1800)] == [1800]
    values[10] = 10
    audit.score_series(Series(LABELS, 60, 1800, values))
    summaries = audit.summaries(1800)
    assert [point.coverage_ratio for point in summaries] == [0, 1]
    assert [point.samples for point in summaries] == [1, 1]


def test_short_horizon_and_off_grid_leads_are_not_scored():
    audit = SelfAudit([61, 600, 7200], 86400)
    item = forecast()
    audit.record(replace(item, quantiles=item.quantiles[:, :10]))
    assert audit.score(61, 5, LABELS) == []
    assert len(audit.score(600, 5, LABELS)) == 1
    assert audit.score(7200, 5, LABELS) == []


def test_stationary_24_hour_coverage():
    # Known Gaussian quantiles isolate scoring accuracy from model calibration.
    random = np.random.default_rng(42)
    actuals = random.normal(size=24 * 60 + 1)
    leads = [600, 1800, 3600, 7200]
    audit = SelfAudit(leads, 86400)
    quantiles = np.repeat(
        np.array([NormalDist().inv_cdf(level / 10) for level in range(1, 10)])[:, None],
        120,
        axis=1,
    )
    for minute, value in enumerate(actuals):
        timestamp = minute * 60
        audit.score(timestamp, float(value), LABELS)
        audit.record(Forecast("stationary", LABELS, timestamp, 60, quantiles, 0, "toto", 1))
        audit.summaries(timestamp)
    summaries = audit.summaries(86400)
    assert len(summaries) == 4
    assert [summary.samples for summary in summaries] == [1431, 1411, 1381, 1321]
    for summary in summaries:
        assert summary.coverage_ratio is not None
        assert 0.6 <= summary.coverage_ratio <= 0.95


@pytest.mark.parametrize(
    ("leads", "window"), [([], 1), ([0], 1), ([-1], 1), ([60], 0), ([60], np.inf)]
)
def test_invalid_configuration(leads: list[int], window: float):
    with pytest.raises(ValueError, match="must"):
        SelfAudit(leads, window)
