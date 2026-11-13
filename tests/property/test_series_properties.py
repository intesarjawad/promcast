import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

from promcast.core.series import align, apply_tail_rule, left_pad_to_patch

finite_values = st.floats(allow_nan=False, allow_infinity=False, width=32)
missing_or_finite = st.one_of(st.just(float("nan")), finite_values)


@given(
    samples=st.lists(st.tuples(st.integers(-100, 100), missing_or_finite), max_size=150),
    step_s=st.integers(1, 3600),
    end_ts=st.integers(0, 2_000_000_000),
    context_steps=st.integers(1, 2048),
)
def test_align_preserves_grid_length(
    samples: list[tuple[int, float]], step_s: int, end_ts: int, context_steps: int
) -> None:
    matrix: dict[str, object] = {
        "resultType": "matrix",
        "result": [
            {
                "metric": {},
                "values": [[end_ts + offset * step_s, str(value)] for offset, value in samples],
            }
        ],
    }
    accepted, skipped = align(matrix, step_s, float(end_ts), context_steps)
    assert len(accepted) == 1
    assert not skipped
    series = accepted[0]
    assert series.values.size == context_steps
    expected = np.full(context_steps, np.nan)
    for offset, value in samples:
        slot = context_steps - 1 + offset
        if 0 <= slot < context_steps:
            expected[slot] = value
    assert_array_equal(series.values, expected)
    assert series.end_ts == end_ts


@given(st.lists(missing_or_finite, min_size=1, max_size=256))
def test_tail_rule_never_returns_nan_in_final_patch(samples: list[float]) -> None:
    values = np.array(samples, dtype=np.float64)
    original = values.copy()
    result = apply_tail_rule(values)
    tail_start = max(0, len(samples) - 32)
    missing = int(np.count_nonzero(np.isnan(values[tail_start:])))
    first_observed = np.flatnonzero(~np.isnan(values))
    fillable = bool(first_observed.size and first_observed[0] <= tail_start)
    assert (result is not None) == (missing <= 8 and fillable)
    if result is not None:
        filled, count = result
        assert count == missing
        assert np.isfinite(filled[-32:]).all()
        assert_array_equal(filled[:tail_start], values[:tail_start])
        observed = ~np.isnan(values)
        assert_array_equal(filled[observed], values[observed])
    assert_array_equal(values, original)


@given(st.lists(missing_or_finite, max_size=256))
def test_padding_preserves_observations_and_masks_missing_slots(samples: list[float]) -> None:
    values = np.array(samples, dtype=np.float64)
    padded, mask = left_pad_to_patch(values)
    assert padded.size >= 64
    assert padded.size % 32 == 0
    assert padded.shape == mask.shape
    assert np.isfinite(padded).all()
    offset = padded.size - values.size
    assert not mask[:offset].any()
    assert_array_equal(padded[:offset], np.zeros(offset))
    assert_array_equal(mask[offset:], ~np.isnan(values))
    assert_array_equal(padded[mask], values[~np.isnan(values)])
    assert_array_equal(padded[~mask], np.zeros(np.count_nonzero(~mask)))


@given(st.lists(finite_values, min_size=32, max_size=256))
def test_padding_never_enters_final_patch_for_observed_context(samples: list[float]) -> None:
    values = np.array(samples, dtype=np.float64)
    padded, mask = left_pad_to_patch(values)
    assert mask[-32:].all()
    assert_array_equal(padded[-32:], values[-32:])
