from dataclasses import dataclass
from math import ulp
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Labels = tuple[tuple[str, str], ...]
SkipReason = Literal["missing", "counter", "too_short", "tail_missing", "model_output", "capped"]

# ADR 0007 explains the patch and observation limits.
_PATCH_SIZE = 32
_MIN_CONTEXT_STEPS = 64
_MAX_TAIL_MISSING = 8
_COUNTER_OBSERVED_FRACTION = 0.95
# Bound subtraction roundoff to four timestamp ULPs and a millionth of a step.
_MAX_GRID_ERROR_STEPS = 1e-6
_GRID_ROUNDOFF_ULPS = 4


@dataclass(frozen=True)
class Series:
    """Regularly spaced observations ending at end_ts, with NaN for missing slots."""

    labels: Labels
    step_s: int
    end_ts: float
    values: FloatArray


@dataclass(frozen=True)
class Skipped:
    """Series identity and rejection reason shared by alignment and run_cycle."""

    labels: Labels
    reason: SkipReason


class _MatrixSeries(BaseModel):
    metric: dict[str, str]
    values: list[tuple[Annotated[float, Field(allow_inf_nan=False)], float]]


class _MatrixResult(BaseModel):
    result_type: Literal["matrix"] = Field(alias="resultType")
    result: list[_MatrixSeries]


def align(
    matrix_result: dict[str, object], step_s: int, end_ts: float, context_steps: int
) -> tuple[list[Series], list[Skipped]]:
    """Align query_range's data object to an inclusive grid ending at end_ts.

    Off-grid and out-of-window samples are ignored. Duplicate timestamps use the
    last supplied sample. Infinities anywhere in a series reject that series.
    """
    if step_s <= 0 or context_steps <= 0 or not np.isfinite(end_ts):
        raise ValueError("step_s and context_steps must be positive and end_ts finite")
    matrix = _MatrixResult.model_validate(matrix_result)
    accepted: list[Series] = []
    skipped: list[Skipped] = []
    start_ts = end_ts - (context_steps - 1) * step_s
    for item in matrix.result:
        labels = tuple(
            sorted((key, value) for key, value in item.metric.items() if key != "__name__")
        )
        if any(np.isinf(value) for _, value in item.values):
            skipped.append(Skipped(labels, "missing"))
            continue
        values = np.full(context_steps, np.nan, dtype=np.float64)
        for timestamp, value in item.values:
            slot = (timestamp - start_ts) / step_s
            if not -1 < slot < context_steps:
                continue
            nearest = round(slot)
            tolerance = min(
                step_s * _MAX_GRID_ERROR_STEPS,
                _GRID_ROUNDOFF_ULPS * max(ulp(timestamp), ulp(start_ts), ulp(end_ts)),
            )
            if (
                0 <= nearest < context_steps
                and abs(timestamp - (start_ts + nearest * step_s)) <= tolerance
            ):
                values[nearest] = value
        accepted.append(Series(labels, step_s, end_ts, values))
    return accepted, skipped


def end_at_origin(series: Series) -> Series | None:
    """Trim trailing missing slots and move end_ts to the last observation."""
    observed = np.flatnonzero(~np.isnan(series.values))
    if not observed.size:
        return None
    stop = int(observed[-1]) + 1
    return Series(
        series.labels,
        series.step_s,
        series.end_ts - (series.values.size - stop) * series.step_s,
        series.values[:stop].copy(),
    )


def apply_tail_rule(
    values: FloatArray, patch: int = _PATCH_SIZE, max_missing: int = _MAX_TAIL_MISSING
) -> tuple[FloatArray, int] | None:
    """Forward-fill the final patch, preserving earlier gaps and the input array.

    Return None if the missing limit is exceeded or no earlier observation can
    seed a missing slot. The count retains missingness before filling.
    """
    if patch <= 0 or max_missing < 0:
        raise ValueError("patch must be positive and max_missing nonnegative")
    if not values.size:
        return None
    start = max(0, values.size - patch)
    missing = int(np.count_nonzero(np.isnan(values[start:])))
    if missing > max_missing:
        return None
    filled = values.copy()
    previous = np.nan
    earlier = np.flatnonzero(~np.isnan(values[:start]))
    if earlier.size:
        previous = float(values[earlier[-1]])
    for index in range(start, values.size):
        if np.isnan(filled[index]):
            if np.isnan(previous):
                return None
            filled[index] = previous
        else:
            previous = float(filled[index])
    return filled, missing


def left_pad_to_patch(values: FloatArray, patch: int = _PATCH_SIZE) -> tuple[FloatArray, BoolArray]:
    """Left-pad to a patch multiple of at least 64; zero and mask missing slots.

    Inputs of at least one patch keep every padding slot outside the final patch.
    This function does not enforce the caller's minimum observation count.
    """
    if patch <= 0:
        raise ValueError("patch must be positive")
    length = ((max(_MIN_CONTEXT_STEPS, values.size) + patch - 1) // patch) * patch
    padded = np.zeros(length, dtype=np.float64)
    mask = np.zeros(length, dtype=np.bool_)
    start = length - values.size
    observed = ~np.isnan(values)
    padded[start:] = np.where(observed, values, 0.0)
    mask[start:] = observed
    return padded, mask


def is_monotone_counter(values: FloatArray, *, known_gauge: bool = False) -> bool:
    """Flag a likely counter only when metric semantics have not established a gauge."""
    observed = values[~np.isnan(values)]
    return bool(
        not known_gauge
        and values.size
        and observed.size / values.size > _COUNTER_OBSERVED_FRACTION
        and np.all(np.diff(observed) >= 0)
    )
