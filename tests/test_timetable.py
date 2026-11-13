import importlib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Scripts import each other by name; pytest puts scripts/ on the path (pyproject.toml).
timetable: ModuleType = importlib.import_module("timetable")
load = importlib.import_module("render_board").load


def test_a_train_stops_on_the_rule_at_its_first_step_at_or_under_it():
    assert timetable.arrive([0.2, 0.1, 0.05, 0.03], 0.05) == ([0.2, 0.1, 0.05], 2)
    assert timetable.arrive([0.2, 0.1, 0.04], 0.05) == ([0.2, 0.1, 0.05], 2)
    assert timetable.arrive([0.2, 0.3], 0.05) == ([0.2, 0.3], None)


def test_level_between_steps_is_linear_and_holds_past_the_end():
    path = [0.4, 0.2]
    assert timetable.level_at(0.6, path, 300, 0) == 0.6
    assert timetable.level_at(0.6, path, 300, 150) == pytest.approx(0.5)
    assert timetable.level_at(0.6, path, 300, 300) == pytest.approx(0.4)
    assert timetable.level_at(0.6, path, 300, 450) == pytest.approx(0.3)
    assert timetable.level_at(0.6, path, 300, 9000) == pytest.approx(0.2)


def test_free_space_rises_up_the_sheet_on_one_log_scale():
    sheet = timetable.WIDE
    assert sheet.y(0.05) < sheet.y(0.1) < sheet.y(0.6)
    assert sheet.y(0.4) - sheet.y(0.2) == pytest.approx(sheet.y(0.2) - sheet.y(0.1))
    assert sheet.x(-sheet.before_s) == sheet.left
    assert sheet.x(sheet.after_s) == sheet.right


def test_every_train_arrives_when_the_scan_says_it_breaches():
    doc = load(ROOT / "docs/design/fleet-scan.json").document
    paths = timetable.load_paths(ROOT / "docs/design/fleet-paths.json")
    step = paths["step_s"]
    for train in timetable.trains(doc, paths):
        row = train.row
        _, q10 = timetable.arrive(train.quantiles[timetable.PAGE_Q], row.threshold)
        arrived = None if q10 is None else doc.origin_ts + (q10 + 1) * step
        assert arrived == row.promcast_breach_ts, row.labels
        assert train.linear is not None
        _, lin = timetable.arrive(train.linear, row.threshold)
        arrived = None if lin is None else doc.origin_ts + (lin + 1) * step
        assert arrived == row.linear_breach_ts, row.labels


def test_the_bar_says_what_the_gap_between_the_stamps_means():
    doc = load(ROOT / "docs/design/fleet-scan.json").document
    said = {timetable.name(row.labels): timetable.verdict(row) for row in doc.rows}
    assert said["db-01"] == "45 min unpaged"
    assert said["logs-01"] == "40 min where predict_linear paged first"
    assert said["ingest-02"] == "a page Promcast never sends"
    assert said["web-03"] is None


def test_a_gap_reads_in_minutes_then_hours():
    assert timetable.minutes(45 * 60) == "45 min"
    assert timetable.minutes(59 * 60) == "59 min"
    assert timetable.minutes(95 * 60) == "1h 35m"


def test_the_social_sheet_ends_where_its_forecast_does():
    sheet = timetable.SOCIAL
    assert sheet.x(sheet.after_s) == sheet.right
    assert sheet.x(0) - sheet.left == pytest.approx((sheet.right - sheet.left) / 4)


def test_every_arrival_falls_on_both_page_sheets():
    doc = load(ROOT / "docs/design/fleet-scan.json").document
    arrivals = [
        ts - doc.origin_ts
        for row in doc.rows
        for ts in (row.promcast_breach_ts, row.linear_breach_ts)
        if ts is not None
    ]
    for sheet in (timetable.WIDE, timetable.NARROW):
        assert max(arrivals) <= sheet.after_s


def test_the_reading_agrees_with_the_sheet():
    doc = load(ROOT / "docs/design/fleet-scan.json").document
    paths = timetable.load_paths(ROOT / "docs/design/fleet-paths.json")
    db, logs, *_, web, _ = timetable.trains(doc, paths)
    step, origin = paths["step_s"], doc.origin_ts
    assert timetable.reading(db, step, origin, 4080) == {
        "p": "6.2 % free",
        "d": "10.5 % free",
        "say": "Pages in",
        "left": "7m",
    }
    assert timetable.reading(db, step, origin, 6300)["left"] == "30 min unpaged"
    assert timetable.reading(db, step, origin, 7200) == {
        "p": "15:15",
        "d": "16:00",
        "say": "",
        "left": "45 min unpaged",
    }
    assert timetable.reading(logs, step, origin, 13500)["left"] == "predict_linear 40 min early"
    assert timetable.reading(web, step, origin, 7200)["left"] == "not in this forecast"
