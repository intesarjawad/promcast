import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from promcast.core.series import (
    Series,
    Skipped,
    align,
    apply_tail_rule,
    end_at_origin,
    is_monotone_counter,
    left_pad_to_patch,
)


def test_range_query_gaps_nan_and_infinities(caplog: pytest.LogCaptureFixture) -> None:
    path = Path(__file__).parents[1] / "fixtures/range_query.json"
    matrix = json.loads(path.read_text(encoding="utf-8"))
    accepted, skipped = align(matrix, step_s=10, end_ts=1040, context_steps=5)
    assert [series.labels for series in accepted] == [
        (("instance", "middle"), ("job", "node")),
        (("instance", "tail"), ("job", "node")),
        (("instance", "nan"),),
    ]
    assert_array_equal(accepted[0].values, [10, 9, np.nan, 7, 6])
    assert_array_equal(accepted[1].values, [10, 9, 8, np.nan, np.nan])
    assert_array_equal(accepted[2].values, [np.nan, 9, np.nan, 7, 6])
    assert all(series.step_s == 10 and series.end_ts == 1040 for series in accepted)
    assert all(series.values.dtype == np.float64 for series in accepted)
    assert skipped == [
        Skipped((("instance", "positive_infinity"),), "missing"),
        Skipped((("instance", "negative_infinity"),), "missing"),
    ]
    assert not caplog.records


def test_align_uses_exact_grid_and_last_duplicate_without_mutating_input() -> None:
    matrix: dict[str, object] = {
        "resultType": "matrix",
        "result": [
            {"metric": {}, "values": [[90, "1"], [105, "2"], [110, "3"], [110, "4"], [130, "5"]]}
        ],
    }
    before = json.dumps(matrix)
    accepted, skipped = align(matrix, 10, 120, 3)
    assert_array_equal(accepted[0].values, [np.nan, 4, np.nan])
    assert not skipped
    assert json.dumps(matrix) == before


def test_align_empty_and_all_missing_series() -> None:
    assert align({"resultType": "matrix", "result": []}, 10, 120, 3) == ([], [])
    accepted, skipped = align(
        {"resultType": "matrix", "result": [{"metric": {}, "values": []}]}, 10, 120, 3
    )
    assert_array_equal(accepted[0].values, [np.nan] * 3)
    assert end_at_origin(accepted[0]) is None
    assert not skipped


@pytest.mark.parametrize(
    ("start", "offset"), [(0.1, 1e-9), (1700000000.1, 1e-5), (1e12 + 0.1, 1e-4)]
)
def test_align_fractional_grid_keeps_first_sample_and_rejects_nearby_off_grid(
    start: float, offset: float
) -> None:
    accepted, skipped = align(
        {
            "resultType": "matrix",
            "result": [
                {"metric": {}, "values": [[start, "1"], [start + 1, "2"], [start + offset, "9"]]}
            ],
        },
        1,
        start + 1,
        2,
    )
    assert not skipped
    assert_array_equal(accepted[0].values, [1, 2])


@pytest.mark.parametrize("value", ["+Inf", "-Inf", float("inf"), float("-inf")])
def test_align_rejects_infinity_even_outside_window(value: str | float) -> None:
    accepted, skipped = align(
        {"resultType": "matrix", "result": [{"metric": {}, "values": [[0, value], [120, "1"]]}]},
        10,
        120,
        3,
    )
    assert not accepted
    assert skipped == [Skipped((), "missing")]


@pytest.mark.parametrize(
    ("step", "end", "count"), [(0, 1, 3), (-1, 1, 3), (1, 1, 0), (1, np.inf, 3)]
)
def test_align_rejects_invalid_grid(step: int, end: float, count: int) -> None:
    with pytest.raises(ValueError, match=r"positive.*finite"):
        align({"resultType": "matrix", "result": []}, step, end, count)


@pytest.mark.parametrize(
    "document",
    [
        '{"resultType": "vector", "result": []}',
        '{"resultType": "matrix", "result": [{"metric": {}, "values": [[1, "bad"]]}]}',
        '{"resultType": "matrix", "result": [{"metric": {}, "values": [["NaN", "1"]]}]}',
    ],
)
def test_align_validates_matrix_boundary(document: str) -> None:
    matrix: dict[str, object] = json.loads(document)
    with pytest.raises(ValidationError):
        align(matrix, 10, 120, 3)


@pytest.mark.parametrize(("field", "value"), [("reason", "counter"), ("labels", ())])
def test_skip_records_are_frozen(field: str, value: object) -> None:
    skipped = Skipped((("instance", "node"),), "missing")
    with pytest.raises(FrozenInstanceError):
        setattr(skipped, field, value)


def test_end_at_origin_updates_timestamp_without_filling_internal_gaps() -> None:
    values = np.array([1, np.nan, 3, np.nan, np.nan], dtype=np.float64)
    series = Series((("job", "node"),), 10, 1040, values)
    result = end_at_origin(series)
    assert result is not None
    assert result.end_ts == 1020
    assert result.labels == series.labels
    assert result.step_s == 10
    assert_array_equal(result.values, [1, np.nan, 3])
    assert_array_equal(series.values, [1, np.nan, 3, np.nan, np.nan])
    assert not np.shares_memory(series.values, result.values)


@pytest.mark.parametrize("values", [[], [np.nan], [np.nan, np.nan]])
def test_end_at_origin_has_no_origin_without_observations(values: list[float]) -> None:
    assert end_at_origin(Series((), 10, 100, np.array(values, dtype=np.float64))) is None


@pytest.mark.parametrize("missing_count", [0, 1, 8, 9])
def test_tail_rule_missing_limit_and_preservation(missing_count: int) -> None:
    values = np.arange(64, dtype=np.float64)
    values[4] = np.nan
    values[32 : 32 + missing_count] = np.nan
    original = values.copy()
    result = apply_tail_rule(values)
    if missing_count > 8:
        assert result is None
    else:
        assert result is not None
        filled, count = result
        assert count == missing_count
        assert_array_equal(filled[:32], original[:32])
        assert_array_equal(filled[32 : 32 + missing_count], [31] * missing_count)
        assert not np.isnan(filled[-32:]).any()
        assert not np.shares_memory(filled, values)
    assert_array_equal(values, original)


def test_tail_rule_uses_last_observation_across_earlier_gaps() -> None:
    values = np.arange(64, dtype=np.float64)
    values[30:34] = np.nan
    result = apply_tail_rule(values)
    assert result is not None
    filled, count = result
    assert count == 2
    assert_array_equal(filled[30:35], [np.nan, np.nan, 29, 29, 34])


@pytest.mark.parametrize("values", [[], [np.nan, 1], [np.nan] * 40])
def test_tail_rule_rejects_unfillable_context(values: list[float]) -> None:
    assert apply_tail_rule(np.array(values, dtype=np.float64)) is None


def test_tail_rule_short_context_and_custom_patch() -> None:
    result = apply_tail_rule(np.array([1, np.nan, 2, np.nan]), patch=3, max_missing=2)
    assert result is not None
    assert_array_equal(result[0], [1, 1, 2, 2])
    assert result[1] == 2


@pytest.mark.parametrize(("length", "expected"), [(0, 64), (31, 64), (32, 64), (64, 64), (65, 96)])
def test_padding_length_mask_and_input_preservation(length: int, expected: int) -> None:
    values = np.arange(length, dtype=np.float64)
    if length:
        values[0] = np.nan
    original = values.copy()
    padded, mask = left_pad_to_patch(values)
    assert padded.size == expected
    assert padded.dtype == np.float64
    assert mask.dtype == np.bool_
    offset = expected - length
    assert_array_equal(padded[:offset], np.zeros(offset))
    assert not mask[:offset].any()
    assert_array_equal(mask[offset:], ~np.isnan(values))
    assert_array_equal(padded[offset:], np.nan_to_num(values))
    assert_array_equal(values, original)


def test_custom_padding_patch_rounds_minimum_up() -> None:
    padded, mask = left_pad_to_patch(np.ones(40), patch=20)
    assert padded.size == 80
    assert mask[-40:].all()
    assert not mask[:-40].any()


@pytest.mark.parametrize("patch", [0, -1])
def test_patch_must_be_positive(patch: int) -> None:
    with pytest.raises(ValueError, match="patch must be positive"):
        left_pad_to_patch(np.ones(32), patch)
    with pytest.raises(ValueError, match="patch must be positive"):
        apply_tail_rule(np.ones(32), patch)


def test_tail_missing_limit_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="max_missing nonnegative"):
        apply_tail_rule(np.ones(32), max_missing=-1)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], False),
        ([np.nan], False),
        ([1], True),
        ([1, 1, 1], True),
        ([1, 2, 3], True),
        ([3, 2, 1], False),
        ([1, 3, 2], False),
    ],
)
def test_counter_requires_nondecreasing_observations(values: list[float], expected: bool) -> None:
    assert is_monotone_counter(np.array(values, dtype=np.float64)) is expected


@pytest.mark.parametrize(("missing", "expected"), [(0, True), (4, True), (5, False), (6, False)])
def test_counter_observed_fraction_is_strict(missing: int, expected: bool) -> None:
    values = np.arange(100, dtype=np.float64)
    values[:missing] = np.nan
    assert is_monotone_counter(values) is expected


def test_known_gauge_overrides_monotone_counter_heuristic():
    values = np.full(288, 0.4)
    assert is_monotone_counter(values)
    assert not is_monotone_counter(values, known_gauge=True)
