# 0014: Return skip records from the pure series core

Status: accepted, 2026-11-05.

`align` was first specified as returning only `list[Series]`, yet infinite
samples also need a reason and a log line. Logging in the core would violate
the pure-core architecture.

The signature becomes:

```python
def align(
    matrix_result: dict[str, object], step_s: int, end_ts: float, context_steps: int
) -> tuple[list[Series], list[Skipped]]:
```

The first list contains accepted series; the second contains frozen `Skipped`
records with sorted labels excluding `__name__` and a `SkipReason`. Any infinite
sample rejects its entire series with reason `missing`; NaN remains a missing
slot. Each list preserves input order. `matrix_result` is the query_range `data`
object containing `resultType: matrix` and `result`. Alignment uses the inclusive
grid ending at `end_ts`, ignores off-grid and out-of-window samples, and keeps
the last supplied sample for duplicate timestamps.

The core emits no logs. The shell logs when consuming these records.
`run_cycle` must import this same `Skipped` and `SkipReason` from `core.series`
for alignment skips and its other skip reasons, avoiding a second skip type.
Tail filling returns the filled count so downstream missingness still includes
those slots. A tail gap without an earlier observation cannot be forward-filled
and returns `None`.

Number 0014 is the next unreserved ADR number: 0011 through
0013 are reserved. ADR 0007's measured padding and tail-patch rules stand.
