import numpy as np
import pytest
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from promcast.core.config import TargetConfig
from promcast.core.forecast import Forecast
from promcast.core.series import Labels, Skipped
from promcast.shell.exposition import AuditSample, Collector
from promcast.shell.store import CycleCompletion, Store

LABELS = (("instance", "node-a"),)


@pytest.fixture
def target() -> TargetConfig:
    return TargetConfig.model_validate(
        {
            "name": "space",
            "query": "space",
            "step": "1m",
            "context": "128m",
            "horizon": "1h",
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


def forecast(model: str = "toto", breach: bool = True, labels: Labels = LABELS) -> Forecast:
    row = np.linspace(10, -49 if breach else 1, 60)
    return Forecast("space", labels, 10000, 60, np.tile(row, (9, 1)), 0.25, model)


def exposition(store: Store) -> tuple[Collector, CollectorRegistry]:
    collector = Collector(store, "0.1.0", "toto-2.0-22m", "abc")
    registry = CollectorRegistry()
    registry.register(collector)
    return collector, registry


@pytest.mark.parametrize(
    ("breach", "actual", "expected"),
    [
        (True, 10, 10660),
        (False, 10, None),
        (True, 0, None),
        (True, -1, None),
    ],
)
def test_absolute_breach_and_suppression(
    target: TargetConfig, breach: bool, actual: float, expected: float | None
) -> None:
    store = Store()
    store.update(target, [forecast(breach=breach)], [], {LABELS: actual}, 12000)
    _, registry = exposition(store)
    assert (
        registry.get_sample_value(
            "promcast_breach_timestamp_seconds",
            {
                "target": "space",
                "instance": "node-a",
                "method": "toto",
            },
        )
        == expected
    )


def test_exact_metric_families_and_values(target: TargetConfig) -> None:
    store = Store()
    skipped = Skipped((("instance", "node-b"),), "capped")
    store.update(target, [forecast(), forecast("linear")], [skipped], {LABELS: 10}, 12000)
    collector, registry = exposition(store)
    collector.observe_cycle("toto", 2)
    collector.replace_audit((AuditSample("space", "10m", "toto", 0.5, 8, 0.75),))
    scrape = generate_latest(registry).decode()
    families = {line.split()[2] for line in scrape.splitlines() if line.startswith("# TYPE")}
    assert families == {
        "promcast_forecast_origin_timestamp_seconds",
        "promcast_forecast_value",
        "promcast_breach_timestamp_seconds",
        "promcast_last_success_timestamp_seconds",
        "promcast_series_missing_fraction",
        "promcast_series_skipped",
        "promcast_series_dropped_total",
        "promcast_selfaudit_coverage_ratio",
        "promcast_selfaudit_mase",
        "promcast_selfaudit_samples",
        "promcast_cycle_duration_seconds",
        "promcast_build_info",
    }
    samples = [sample for family in registry.collect() for sample in family.samples]
    identities = [(sample.name, tuple(sorted(sample.labels.items()))) for sample in samples]
    assert len(identities) == len(set(identities))
    labels = {"target": "space", "instance": "node-a"}
    assert registry.get_sample_value("promcast_forecast_origin_timestamp_seconds", labels) == 10000
    assert registry.get_sample_value("promcast_series_missing_fraction", labels) == 0.25
    for lead, value in [("10m", 1), ("30m", -19), ("1h", -49), ("end", -49)]:
        for quantile in range(1, 10):
            assert (
                registry.get_sample_value(
                    "promcast_forecast_value",
                    {
                        **labels,
                        "lead": lead,
                        "quantile": f"{quantile / 10:.1f}",
                    },
                )
                == value
            )
    assert (
        registry.get_sample_value("promcast_cycle_duration_seconds_count", {"model": "toto"}) == 1
    )
    assert registry.get_sample_value("promcast_cycle_duration_seconds_sum", {"model": "toto"}) == 2
    assert (
        registry.get_sample_value("promcast_selfaudit_samples", {"target": "space", "lead": "10m"})
        == 8
    )


def test_last_success_and_stale_series(target: TargetConfig) -> None:
    store = Store()
    _, registry = exposition(store)
    labels = {"target": "space"}
    store.update(target, [], [], {}, 11000)
    assert registry.get_sample_value("promcast_last_success_timestamp_seconds", labels) is None
    store.update(target, [forecast()], [], {LABELS: 10}, 12000)
    old = store.snapshot()
    assert registry.get_sample_value("promcast_last_success_timestamp_seconds", labels) == 12000
    store.update(target, [], [Skipped(LABELS, "capped")], {}, 13000)
    assert registry.get_sample_value("promcast_last_success_timestamp_seconds", labels) == 12000
    assert registry.get_sample_value("promcast_series_dropped_total", labels) == 1
    assert "# TYPE promcast_forecast_value" in generate_latest(registry).decode()
    assert not any(
        family.samples for family in registry.collect() if family.name == "promcast_forecast_value"
    )
    assert old[0].forecasts
    store.update(target, [forecast(labels=())], [], {(): 10}, 14000)
    assert registry.get_sample_value("promcast_last_success_timestamp_seconds", labels) == 14000
    assert registry.get_sample_value("promcast_series_dropped_total", labels) == 1
    assert (
        registry.get_sample_value(
            "promcast_series_skipped",
            {
                **labels,
                "instance": "node-a",
                "reason": "capped",
            },
        )
        is None
    )


def test_store_copies_forecast_input(target: TargetConfig) -> None:
    store = Store()
    item = forecast()
    store.update(target, [item], [], {LABELS: 10}, 12000)
    item.quantiles[:] = 123
    assert store.snapshot()[0].forecasts[0].quantiles[0, 0] == 10


def test_source_label_collisions_and_value_escaping(target: TargetConfig) -> None:
    labels = (
        ("method", "GET"),
        ("exported_method", "original"),
        ("exported_exported_method", "also original"),
        ("target", "source target"),
        ("quantile", "source quantile"),
        ("lead", "source lead"),
        ("reason", "source reason"),
        ("__name__", "source metric"),
        ("path", 'C:\\metrics\n"disk"'),
    )
    store = Store()
    store.update(
        target, [forecast(labels=labels)], [Skipped(labels, "missing")], {labels: 10}, 12000
    )
    _, registry = exposition(store)
    scrape = generate_latest(registry).decode()
    samples = [
        sample for family in text_string_to_metric_families(scrape) for sample in family.samples
    ]
    preserved = {
        "target": "space",
        "exported_exported_exported_method": "GET",
        "exported_method": "original",
        "exported_exported_method": "also original",
        "exported_target": "source target",
        "exported_quantile": "source quantile",
        "exported_lead": "source lead",
        "exported_reason": "source reason",
        "exported___name__": "source metric",
        "path": 'C:\\metrics\n"disk"',
    }
    breach = next(
        sample for sample in samples if sample.name == "promcast_breach_timestamp_seconds"
    )
    assert breach.labels == {**preserved, "method": "toto"}
    value = next(sample for sample in samples if sample.name == "promcast_forecast_value")
    assert value.labels == {**preserved, "quantile": "0.1", "lead": "10m"}
    skipped = next(sample for sample in samples if sample.name == "promcast_series_skipped")
    assert skipped.labels == {**preserved, "reason": "missing"}
    assert r'path="C:\\metrics\n\"disk\""' in scrape


def test_store_explicit_drop_count_replaces_inferred_count(target: TargetConfig) -> None:
    store = Store()
    store.update(target, [], [Skipped(LABELS, "capped")], {}, CycleCompletion(12000, 70))
    store.update(target, [], [], {}, CycleCompletion(13000, 70))
    assert store.snapshot()[0].dropped_total == 140
    store.update(target, [], [Skipped(LABELS, "capped")], {}, 14000)
    assert store.snapshot()[0].dropped_total == 141
    with pytest.raises(ValueError, match="nonnegative"):
        store.update(target, [], [], {}, CycleCompletion(15000, -1))
    assert store.snapshot()[0].dropped_total == 141


def test_below_uses_complementary_quantile(target: TargetConfig) -> None:
    store = Store()
    item = forecast()
    item.quantiles[1:] = 100
    store.update(target, [item], [], {LABELS: 10}, 12000)
    _, registry = exposition(store)
    assert (
        registry.get_sample_value(
            "promcast_breach_timestamp_seconds",
            {
                "target": "space",
                "instance": "node-a",
                "method": "toto",
            },
        )
        == 10660
    )


def test_linear_target_exports_primary_values(target: TargetConfig) -> None:
    store = Store()
    store.update(
        target.model_copy(update={"method": "linear"}),
        [forecast("linear")],
        [],
        {LABELS: 10},
        12000,
    )
    _, registry = exposition(store)
    assert (
        registry.get_sample_value(
            "promcast_forecast_value",
            {
                "target": "space",
                "instance": "node-a",
                "quantile": "0.5",
                "lead": "end",
            },
        )
        == -49
    )


def test_short_horizon_omits_unavailable_leads(target: TargetConfig) -> None:
    store = Store()
    item = forecast()
    short = Forecast(item.target, (), item.origin_ts, 60, item.quantiles[:, :5], 0, item.model)
    store.update(target, [short], [], {(): 10}, 12000)
    _, registry = exposition(store)
    values = next(
        family for family in registry.collect() if family.name == "promcast_forecast_value"
    )
    assert {sample.labels["lead"] for sample in values.samples} == {"end"}
