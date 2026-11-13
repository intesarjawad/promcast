from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue

ROOT = Path(__file__).resolve().parents[2]


class Target(BaseModel):
    model_config = ConfigDict(extra="allow")
    ref_id: str = Field(alias="refId")
    expr: str
    legend: str | None = Field(default=None, alias="legendFormat")
    instant: bool = False
    datasource: dict[str, str] | None = None


class Mapping(BaseModel):
    type: str
    options: dict[str, JsonValue]


class Defaults(BaseModel):
    unit: str | None = None
    no_value: str | None = Field(default=None, alias="noValue")
    mappings: list[Mapping] = Field(default_factory=list[Mapping])


class Override(BaseModel):
    matcher: dict[str, str]
    properties: list[dict[str, JsonValue]]


class FieldConfig(BaseModel):
    defaults: Defaults = Field(default_factory=Defaults)
    overrides: list[Override] = Field(default_factory=list[Override])


class Panel(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    type: str
    title: str
    datasource: dict[str, str] | None = None
    field_config: FieldConfig = Field(default_factory=FieldConfig, alias="fieldConfig")
    targets: list[Target] = Field(default_factory=list[Target])
    panels: list["Panel"] = Field(default_factory=list["Panel"])
    collapsed: bool = False


class Dashboard(BaseModel):
    model_config = ConfigDict(extra="allow")
    uid: str
    refresh: str
    time: dict[str, str]
    panels: list[Panel]


@pytest.fixture
def dashboard() -> Dashboard:
    return Dashboard.model_validate_json((ROOT / "grafana/dashboards/promcast.json").read_text())


def panel(dashboard: Dashboard, title: str) -> Panel:
    return next(
        item for item in [*dashboard.panels, *dashboard.panels[-1].panels] if item.title == title
    )


def test_datasources_name_prometheus_and_promcast():
    document = yaml.safe_load((ROOT / "grafana/provisioning/datasources/promcast.yml").read_text())
    sources = {source["uid"]: source for source in document["datasources"]}
    assert set(sources) == {"prometheus", "promcast"}
    assert sources["prometheus"]["url"] == "http://prometheus:9090"
    assert sources["promcast"]["url"] == "http://promcast:8787"
    assert all(source["type"] == "prometheus" for source in sources.values())


def test_time_window_and_refresh(dashboard: Dashboard):
    assert dashboard.time == {"from": "now-6h", "to": "now+2h"}
    assert dashboard.refresh == "30s"
    ids = [item.id for item in [*dashboard.panels, *dashboard.panels[-1].panels]]
    assert len(ids) == len(set(ids))


def test_first_screen_holds_forecast_and_breach_stat_above_collapsed_trust_row(
    dashboard: Dashboard,
):
    *first_screen, trust = dashboard.panels
    assert [item.title for item in first_screen][:2] == [
        "Synthetic free disk space and forecasts",
        "Breach in",
    ]
    assert trust.type == "row"
    assert trust.title == "Trust"
    assert trust.collapsed is True
    assert [item.title for item in trust.panels] == ["Self-audit", "Missingness"]


def test_forecast_panel_draws_cone_median_and_threshold(dashboard: Dashboard):
    forecast = panel(dashboard, "Synthetic free disk space and forecasts")
    legends = {target.ref_id: target.legend for target in forecast.targets}
    assert legends == {
        "A": "Actual",
        "B": "q10",
        "C": "q50",
        "D": "q90",
        "E": "Full disk threshold",
    }
    queries = {target.legend: target.expr for target in forecast.targets}
    assert queries["q10"] == 'promcast_forecast_q10{target="demo_disk_free"}'
    assert queries["q50"] == 'promcast_forecast_q50{target="demo_disk_free"}'
    assert queries["q90"] == 'promcast_forecast_q90{target="demo_disk_free"}'
    band = next(
        item
        for item in forecast.field_config.overrides
        if item.matcher == {"id": "byName", "options": "q90"}
    )
    assert {"id": "custom.fillBelowTo", "value": "q10"} in band.properties


def test_breach_in_has_two_prometheus_rows_in_minutes_with_text_when_absent(
    dashboard: Dashboard,
):
    stat = panel(dashboard, "Breach in")
    assert stat.type == "stat"
    assert stat.model_extra is not None
    assert stat.model_extra["timeFrom"] == "5m"
    defaults = stat.field_config.defaults
    assert defaults.unit == "suffix: min"
    assert defaults.no_value == "unavailable"
    assert any(
        mapping.type == "value"
        and mapping.options["-1"] == {"text": "no breach forecast", "index": 0}
        for mapping in defaults.mappings
        if "-1" in mapping.options
    )
    assert any(
        mapping.type == "special"
        and mapping.options["match"] == "null"
        and mapping.options["result"] == {"text": "unavailable", "index": 1}
        for mapping in defaults.mappings
        if "match" in mapping.options
    )
    assert [(target.legend, target.datasource, target.instant) for target in stat.targets] == [
        ("Promcast", {"type": "prometheus", "uid": "prometheus"}, True),
        ("predict_linear[1h]", {"type": "prometheus", "uid": "prometheus"}, True),
    ]
    for method, target in zip(("toto", "linear"), stat.targets, strict=True):
        assert target.expr == (
            f'clamp_min((promcast_breach_timestamp_seconds{{target="demo_disk_free",method="{method}"}}'
            " - time()) / 60, 0) or on() (vector(-1) and on() "
            '(time() - promcast_last_success_timestamp_seconds{target="demo_disk_free"} <= 180) '
            'and on() (up{job="promcast"} == 1))'
        )


def test_forecast_state_names_each_condition_in_text(dashboard: Dashboard):
    state = panel(dashboard, "Forecast state")
    texts = {target.ref_id: target.legend for target in state.targets}
    assert texts == {
        "A": "warming up: first forecast within one cycle",
        "B": "stale since",
        "C": "already past threshold",
        "D": "forecast current",
    }
    warming = next(target for target in state.targets if target.ref_id == "A")
    assert warming.expr == (
        'vector(1) unless on() promcast_last_success_timestamp_seconds{target="demo_disk_free"}'
    )
    assert state.field_config.defaults.no_value == "unavailable"
    stale_unit = next(
        item
        for item in state.field_config.overrides
        if item.matcher == {"id": "byFrameRefID", "options": "B"}
    )
    assert stale_unit.properties == [{"id": "unit", "value": "dateTimeAsLocal"}]


def test_trust_tables_show_text_when_empty(dashboard: Dashboard):
    for title, fragment in (("Self-audit", "self-audit"), ("Missingness", "unavailable")):
        table = panel(dashboard, title)
        assert fragment in str(table.field_config.defaults.no_value)
        assert all(target.instant for target in table.targets)
    audit = panel(dashboard, "Self-audit")
    assert [target.expr for target in audit.targets] == [
        'promcast_selfaudit_coverage_ratio{target="demo_disk_free"}',
        'promcast_selfaudit_mase{target="demo_disk_free"}',
        'promcast_selfaudit_samples{target="demo_disk_free"}',
    ]


def test_query_failures_have_unavailable_fallbacks(dashboard: Dashboard):
    for title in ("Forecast state", "Breach in"):
        defaults = panel(dashboard, title).field_config.defaults
        assert defaults.no_value == "unavailable"
        for mapping in defaults.mappings:
            if mapping.options.get("match") == "null":
                assert "unavailable" in str(mapping.options["result"])
    current = next(
        target
        for target in panel(dashboard, "Forecast state").targets
        if target.legend == "forecast current"
    )
    assert "time() - promcast_last_success_timestamp_seconds" in current.expr
    assert "<= 180" in current.expr
    assert "demo_disk_free_bytes > 0" in current.expr


def test_zero_skips_and_measurements_are_visible_in_missingness(dashboard: Dashboard):
    missing = panel(dashboard, "Missingness")
    assert missing.field_config.defaults.no_value == "unavailable"
    skipped = next(target for target in missing.targets if target.ref_id == "B")
    assert "or" in skipped.expr
    assert "0 * promcast_series_missing_fraction" in skipped.expr
    assert missing.model_extra is not None
    transforms = missing.model_extra["transformations"]
    organize = next(item["options"] for item in transforms if item["id"] == "organize")
    assert organize["indexByName"]["Value #A"] == 0
    assert organize["indexByName"]["Value #B"] == 1


def test_successful_absence_of_a_breach_is_distinct_from_missing_query_data(dashboard: Dashboard):
    breach = panel(dashboard, "Breach in")
    assert any(
        mapping.type == "value"
        and "-1" in mapping.options
        and "no breach forecast" in str(mapping.options["-1"])
        for mapping in breach.field_config.defaults.mappings
    )
    for target in breach.targets:
        assert "clamp_min(" in target.expr
        assert "vector(-1)" in target.expr
        assert "promcast_last_success_timestamp_seconds" in target.expr
        assert 'up{job="promcast"} == 1' in target.expr
