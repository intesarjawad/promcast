import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from numpy.testing import assert_array_equal

from promcast.core.config import TargetConfig, parse_config, parse_fragment
from promcast.core.forecast import run_cycle
from promcast.core.forecaster import FakeForecaster
from promcast.core.operating import MethodScore, OperatingPoint
from promcast.core.replay import (
    Episode,
    Incident,
    MethodResult,
    ReplayResult,
    ReplaySpec,
    auto_threshold,
    detect_incidents,
    episodes,
    forecast_at,
    replay,
    replay_sweep,
    rolling_origins,
    score,
    split_index,
)
from promcast.core.report import recommend, report_json, report_markdown
from promcast.core.series import Series


@pytest.fixture
def spec() -> ReplaySpec:
    return ReplaySpec("below", 4, 10, 60, 1, 24, threshold=0, budget_per_week=1e6)


def test_ramp_lead_includes_persistence_delay(spec: ReplaySpec) -> None:
    # Threshold is crossed at minute 20. The strict 6m rule becomes true at
    # minute 15 and fires at minute 16, leaving exactly four minutes of lead.
    values = 20 - np.arange(40, dtype=np.float64)
    result = replay(
        values,
        replace(spec, calibration_fraction=0.1),
        [OperatingPoint("linear", 360, window_s=3600)],
    )
    test = result.methods["linear"].test
    assert test.n_incidents == test.caught_early == 1
    assert test.leads_s == [240]
    assert test.warning_times_s == [240]
    assert test.recall == test.precision == 1
    assert test.coverage_80 == 1
    assert test.wql_by_lead == [0.0] * 10


def test_late_episode_is_matched_but_not_caught(spec: ReplaySpec) -> None:
    result = score([Episode(120, 180, 360)], [Incident(120, 300)], 1, spec)
    assert result.late == 1
    assert result.recall == 0
    assert result.precision == 1
    assert result.warning_times_s == [0]
    assert result.reset_s == [60]


def test_too_early_episode_is_false_and_multiple_matches_count_incident_once(
    spec: ReplaySpec,
) -> None:
    result = score(
        [Episode(0, 60, 120), Episode(600, 660, 720), Episode(720, 780, 840)],
        [Incident(900, 1080)],
        2,
        spec,
    )
    assert result.n_episodes == 3
    assert result.matched == 2
    assert result.caught_early == 1
    assert result.precision == 2 / 3
    assert result.false_alerts_per_week == 0.5
    assert result.warning_times_s == [240]


def test_episode_ignores_breached_origins_and_resets_at_missing_cycle(spec: ReplaySpec) -> None:
    times = np.array([0, 60, 120, 180, 240, 360, 420], dtype=np.float64)
    condition = np.ones(7, dtype=np.bool_)
    breached = np.array([False, False, True, True, False, False, False])
    assert episodes(times, condition, breached, spec) == [
        Episode(0, 60, 120),
        Episode(360, 420, 480),
    ]
    assert episodes(times, condition, np.ones(7, dtype=np.bool_), spec) == []


def test_episodes_require_consecutive_conditions(spec: ReplaySpec) -> None:
    times = np.arange(5, dtype=np.float64) * 60
    assert episodes(
        times, np.array([True, False, True, True, False]), np.zeros(5, dtype=np.bool_), spec
    ) == [Episode(120, 180, 240)]
    assert (
        episodes(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.bool_),
            np.array([], dtype=np.bool_),
            spec,
        )
        == []
    )


def test_incidents_minimum_missing_and_horizon_merge(spec: ReplaySpec) -> None:
    values = np.ones(40)
    values[2:4] = -1
    values[6:9] = -1
    values[10] = np.nan
    values[12:15] = -1
    values[24:27] = -1
    result = detect_incidents(np.arange(40, dtype=np.float64) * 60, values, 0, spec)
    # The last pair is exactly a horizon apart and must remain separate.
    assert result == [Incident(360, 840), Incident(1440, 1560)]
    above = replace(spec, direction="above")
    assert detect_incidents(np.arange(40, dtype=np.float64) * 60, -values, 0, above) == result


class RecordingModel(FakeForecaster):
    def __init__(self) -> None:
        self.contexts: list[Series] = []

    def forecast(self, batch: list[Series], horizon_steps: int):
        self.contexts.extend(batch)
        return super().forecast(batch, horizon_steps)


def test_future_sentinel_cannot_reach_model(spec: ReplaySpec) -> None:
    model = RecordingModel()
    fixed = replace(spec, context_steps=64)
    values = np.arange(100, dtype=np.float64)
    first = forecast_at(values, 70, fixed, model)
    values[71:] = -1e30
    second = forecast_at(values, 70, fixed, model)
    assert_array_equal(first, second)
    assert_array_equal(model.contexts[0].values, np.arange(7, 71))
    assert model.contexts[0].end_ts == 70 * 60
    with pytest.raises(ValueError, match="origin"):
        forecast_at(values, values.size, fixed, model)
    with pytest.raises(ValueError, match="origin"):
        forecast_at(values, 1, fixed, model)


def test_model_predictions_reused_across_operating_points(spec: ReplaySpec) -> None:
    model = RecordingModel()
    fixed = replace(spec, context_steps=64)
    values = np.sin(np.arange(250, dtype=np.float64)) + 10
    points = [
        OperatingPoint("fake", warn, quantile)
        for warn in (300, 600)
        for quantile in (0.5, 0.7, 0.9)
    ]
    replay(values, fixed, points, {"fake": model})
    assert len(model.contexts) == len(rolling_origins(values.size, fixed))


def test_test_split_does_not_change_threshold_or_selected_point(spec: ReplaySpec) -> None:
    values = 10 + np.sin(np.arange(1000, dtype=np.float64) / 10)
    fixed = replace(spec, threshold=None, auto_percentile=20)
    points = [OperatingPoint("linear", warn, window_s=3600) for warn in (300, 600)]
    first = replay(values, fixed, points)
    values[first.split :] = 1e6
    second = replay(values, fixed, points)
    assert first.threshold == second.threshold
    assert first.methods["linear"].point == second.methods["linear"].point
    assert first.methods["linear"].calibration == second.methods["linear"].calibration
    assert first.methods["linear"].test != second.methods["linear"].test


def test_static_margin_sweep_is_scored_like_linear(spec: ReplaySpec) -> None:
    values = 10 + np.sin(np.arange(500, dtype=np.float64) / 10)
    points = [
        OperatingPoint("static-margin", warn, margin_k=k)
        for warn in (300, 600)
        for k in (0.5, 1, 2, 3)
    ]
    result = replay(values, replace(spec, threshold=9.5), points)
    static = result.methods["static-margin"]
    assert len(static.sweep) == 8
    assert static.calibration.n_incidents > 0
    assert static.test.n_incidents > 0
    assert static.point in static.sweep


def test_reports_are_byte_identical_and_show_zero_calibration_incidents(spec: ReplaySpec) -> None:
    values = np.ones(1000)
    points = [OperatingPoint("linear", 300, window_s=3600), OperatingPoint("seasonal-naive", 300)]
    first, second = replay(values, spec, points), replay(values, spec, points)
    assert report_json(first) == report_json(second)
    assert report_markdown(first) == report_markdown(second)
    assert '"n_incidents_cal": 0' in report_json(first)
    assert "n_incidents_cal" in report_markdown(first)
    assert "Series-weeks" in report_markdown(first)
    assert "False alerts/week" in report_markdown(first)


def method_result(name: str, caught: int) -> MethodResult:
    point = OperatingPoint(name, 300, window_s=3600 if name == "linear" else 0)
    calibration = MethodScore(10, caught, 0, caught, caught, 1, [100] * caught)
    # Test scores deliberately oppose calibration ranking.
    test = replace(calibration, caught_early=10 - caught)
    return MethodResult(point, calibration, test)


def test_recommend_uses_calibration_linear_window_and_names_replay_only_winner(
    spec: ReplaySpec,
) -> None:
    methods = {
        name: method_result(name, caught)
        for name, caught in (("linear", 8), ("toto-2.0-22m", 5), ("ets", 10))
    }
    result = ReplayResult(spec, 0, 30, methods)
    target = TargetConfig.model_validate(
        {
            "name": "disk",
            "query": "disk",
            "step": "1m",
            "context": "128m",
            "horizon": "10m",
            "season": "24h",
            "direction": "below",
            "threshold": 0,
            "quantile": 0.9,
            "warn": "5m",
            "method": "toto",
            "linear_window": "10m",
            "max_series": 1,
        }
    )
    text = recommend(result, target)
    fragment = parse_fragment(text, environment={})
    assert fragment.targets[0].method == "linear"
    assert fragment.targets[0].linear_window.total_seconds() == 3600
    assert "# Replay-only calibration winner: ets" in text


def test_percentile_and_split_validation(spec: ReplaySpec) -> None:
    values = np.arange(100, dtype=np.float64)
    fixed = replace(spec, threshold=None)
    split = split_index(100, fixed)
    assert split == 30
    assert auto_threshold(values, split, fixed) == np.percentile(values[:30], 1)
    with pytest.raises(ValueError, match="nonempty"):
        split_index(1, fixed)


def test_fallbacks_recorded(spec: ReplaySpec) -> None:
    values = np.ones(65)
    result = replay(
        values,
        replace(spec, season_steps=0, context_steps=64),
        [OperatingPoint("seasonal-naive", 300), OperatingPoint("ets", 300)],
    )
    assert result.methods["seasonal-naive"].fallbacks == ("last_value",)
    assert result.methods["ets"].fallbacks == ("nonseasonal_ets",)


def test_reports_include_incident_cluster_intervals_and_flag_test_budget(spec: ReplaySpec) -> None:
    values = 20 - np.arange(40, dtype=np.float64)
    result = replay(
        values,
        replace(spec, calibration_fraction=0.1),
        [OperatingPoint("linear", 360, window_s=3600)],
    )
    assert report_json(result) == report_json(result)
    assert json.loads(report_json(result))["methods"]["linear"]["intervals"]["recall_95"] == [
        1.0,
        1.0,
    ]
    assert "[240.0000, 240.0000]" in report_markdown(result)
    method = result.methods["linear"]
    noisy = replace(method.test, n_episodes=method.test.n_episodes + 1)
    flagged = replace(
        result,
        spec=replace(result.spec, budget_per_week=0),
        methods={"linear": replace(method, test=noisy)},
    )
    assert "above 1.5 x budget" in report_markdown(flagged)
    assert '"test_budget_exceeded": true' in report_json(flagged)


def test_gappy_linear_history_skips_origins_without_two_observations(spec: ReplaySpec) -> None:
    values = np.full(100, np.nan)
    values[5] = 10
    values[70:73] = [3, 2, 1]
    result = replay(values, spec, [OperatingPoint("linear", 300, window_s=120)])
    assert result.methods["linear"].test.n_episodes == 1
    assert result.methods["linear"].test.matched == 0
    assert result.methods["linear"].test.coverage_80 == 1
    assert result.methods["linear"].calibration.coverage_80 is None


class NonfiniteModel(FakeForecaster):
    def forecast(self, batch: list[Series], horizon_steps: int):
        return [np.full((9, horizon_steps), np.nan) for _ in batch]


def test_nonfinite_model_forecasts_never_become_episodes(spec: ReplaySpec) -> None:
    result = replay(
        np.ones(100),
        replace(spec, context_steps=64),
        [OperatingPoint("fake", 300)],
        {"fake": NonfiniteModel()},
    )
    assert result.methods["fake"].test.n_episodes == 0
    assert result.methods["fake"].test.coverage_80 is None


def test_test_sweep_reuses_forecasts_and_preserves_calibration_selection(spec: ReplaySpec):
    model = RecordingModel()
    fixed = replace(spec, context_steps=64)
    values = np.sin(np.arange(250, dtype=np.float64)) + 10
    points = [
        OperatingPoint("fake", warn, quantile) for warn in (300, 600) for quantile in (0.5, 0.9)
    ]
    result = replay(values, fixed, points, {"fake": model})
    method = result.methods["fake"]
    assert len(model.contexts) == len(rolling_origins(values.size, fixed))
    assert set(method.test_sweep) == set(points)
    assert method.test_sweep[method.point] == method.test
    for point in points:
        single = replay(values, fixed, [point], {"fake": FakeForecaster()})
        assert method.test_sweep[point] == single.methods["fake"].test


def test_secondary_labels_do_not_change_primary_scores_or_selection(spec: ReplaySpec):
    values = 20 - np.arange(40, dtype=np.float64)
    fixed = replace(spec, calibration_fraction=0.1)
    points = [OperatingPoint("linear", warn, window_s=3600) for warn in (300, 360)]
    primary = replay(values, fixed, points)
    labelled = replay(values, fixed, points, labelled_incidents=[Incident(10 * 60, 10 * 60)])
    before, after = primary.methods["linear"], labelled.methods["linear"]
    assert after.point == before.point
    assert after.calibration == before.calibration
    assert after.test == before.test
    assert after.test_sweep == before.test_sweep
    assert after.labelled_test is not None
    assert after.labelled_test.n_incidents == 1
    assert after.labelled_test.warning_times_s == [0]
    assert before.point.warn_s == 300
    assert before.test.warning_times_s == [180]


def test_sweep_scores_every_point_without_choosing_and_matches_replay(spec: ReplaySpec) -> None:
    values = 10 + np.sin(np.arange(1000, dtype=np.float64) / 10)
    fixed = replace(spec, threshold=None, auto_percentile=20)
    points = [OperatingPoint("linear", warn, window_s=3600) for warn in (300, 600)]
    sweep = replay_sweep(values, replace(fixed, budget_per_week=0.0), points)
    assert [entry.point for entry in sweep["linear"]] == points
    chosen = replay(values, fixed, points).methods["linear"]
    entry = next(e for e in sweep["linear"] if e.point == chosen.point)
    assert (entry.calibration, entry.test) == (chosen.calibration, chosen.test)


@pytest.mark.parametrize("model", ["toto-2.0-4m", "toto-2.0-22m"])
def test_recommended_model_and_gauge_survive_configuration_round_trip(model: str):
    original = parse_config(Path("demo/promcast.yaml").read_text(), environment={})
    target = original.targets[0].model_copy(update={"known_gauge": True})
    spec = ReplaySpec("below", 1440, 120, 60, 2, 1440, threshold=0, budget_per_week=100)
    result = ReplayResult(spec, 0, 100, {model: method_result(model, 8)})
    text = recommend(result, target)
    fragment = parse_fragment(text, environment={})
    assert fragment.targets[0].known_gauge is True
    assert fragment.model is not None
    assert fragment.model.name == model
    document = yaml.safe_load(Path("demo/promcast.yaml").read_text())
    document.update(yaml.safe_load(text))
    applied = parse_config(yaml.safe_dump(document), environment={})
    assert applied.model.name == model
    series = Series((), 60, 1700000000, np.full(1440, 0.5))
    forecasts, skipped = run_cycle(applied.targets[0], [series], FakeForecaster(), series.end_ts)
    assert len(forecasts) == 2
    assert not skipped
