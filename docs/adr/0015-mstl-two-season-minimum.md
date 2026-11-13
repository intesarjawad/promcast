# 0015: Require two full seasons before fitting MSTL

Status: accepted, 2026-11-05.

The first design selected MSTL once one season was available. MSTL
decomposition is undefined below two full seasons; fitting the pinned augurs
implementation in that interval raises `series has less than two periods`.

Decision: `ets` uses `augurs.AutoETS(1, "ZZN")` whenever its fit
window contains fewer than `2 * season_steps` slots. At exactly two full
seasons it uses `augurs.MSTL.ets([season_steps])`. This extends the existing
nonseasonal fallback through the interval from one to two seasons.

Leading missing slots remain excluded before measuring history. Interior gaps
retain their elapsed slots and are interpolated as before. The trailing
three-season fit limit, 80% prediction interval and nine-row array contract
remain unchanged. No new fallback metadata is added: callers can identify the
fallback from the supplied history span after excluding leading missing slots,
using the amended two-season threshold.

Regression tests compare the lower, point and upper forecasts against AutoETS
at exactly one season and one slot short of two, and against MSTL at exactly
two seasons. Each boundary is checked with and without leading missing slots.
