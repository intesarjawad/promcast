# 0017: Group range query bounds in a frozen dataclass

Status: accepted, 2026-11-05.

`range_query` needs six values. The repository limits functions to five
arguments and that limit takes precedence, so `range_query` takes a frozen `RangeGrid` containing
`start_ts`, `end_ts` and `step_s`. Validation and chunking remain unchanged.
