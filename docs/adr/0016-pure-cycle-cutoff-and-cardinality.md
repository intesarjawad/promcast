# 0016: Bound pure forecast cycles by time and eligible cardinality

Status: accepted, 2026-11-05.

`run_cycle` composes the series preparation rules, forecaster protocol and linear
baseline in `core.forecast.run_cycle`. It accepts aligned `Series` objects,
returns frozen `Forecast` records and reuses `core.series.Skipped` from ADR 0014.
It performs no I/O and reads no clock.

## Time cutoff

Discard samples whose timestamps are strictly greater than the supplied
`now_ts` before moving the series end to its last observation. Keep a sample
exactly at `now_ts`. Preserve the input grid rather than moving observations
onto a different grid. An empty context after trimming, like an all-missing
context, produces `Skipped(reason="missing")`. A future infinity cannot reject
a context once that sample has been discarded. Input arrays remain unchanged.

After origin alignment, reject infinite observations and missing fractions over
0.5, monotone counters, contexts with fewer than 64 observations, and final
patches that fail the tail rule. Missingness and observation counts are measured
before tail filling and exclude the trailing missing slots removed by
`end_at_origin`. The linear baseline fits the original observations in the
configured trailing `linear_window`, preserving gaps and ignoring tail fills.
Fewer than two observations in that window also produces `too_short`, since
there is no defined linear fit.

## Cardinality and inference

Apply `max_series` after input skip filtering. Preserve the caller's deterministic
configuration order; do not sort labels or rank values. Extend the shared
`SkipReason` literal with `capped` and emit that reason for every otherwise
eligible context beyond the limit. A failed model output does not free a slot
or trigger another inference pass.

For `method="toto"`, pass all retained prepared contexts in one call to the
forecaster, which owns length grouping and bounded inference batches. Return
model and linear forecasts next to each other in series order. For
`method="linear"`, return only the linear forecast without invoking the model.
Forecast origins are the last retained observation, never the wall clock.
Enforce monotone quantiles along axis zero. Non-finite output rejects that
series with `model_output`; a wrong output count or shape raises `ValueError`
as a forecaster protocol violation. Output rejection discards both forecasts
for that series. Skip records follow input filtering order, followed by model
output rejections in batch order.

## Verification

`tests/core/test_forecast.py` exercises all six skip reasons with deterministic
forecasters, cutoff equality and empty-after-trim behavior, input order and cap
filtering, batching, tail filling and missingness, origin timestamps, the
configured linear window, method routing, output finiteness and protocol errors.
It also verifies that invalid inputs beyond the cap keep their input skip reason
and that failed model outputs do not admit capped contexts to a second batch.

Verification on 2026-11-05:

- `UV_CACHE_DIR=/tmp/promcast-uv-cache uv run pytest tests/core/test_forecast.py`:
  29 passed.
- `UV_CACHE_DIR=/tmp/promcast-uv-cache make check`: formatting and lint passed,
  pyright reported zero errors and warnings, and 257 tests passed with 43
  deselected by the standard slow/browser marker filter.

The cache path is a local choice. No dependency or check configuration
changes are required.
