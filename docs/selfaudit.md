# Self-audit

Promcast keeps forecasts in memory and compares them with later observations.
All audit state resets on restart. Nothing is written to the TSDB.

The exported leads are `10m`, `30m`, `1h`, and `end`. Fixed leads outside the
horizon or off the sample grid are omitted. When `end` coincides with a fixed
lead, both labels report the same score without counting it twice internally.

`promcast_selfaudit_coverage_ratio{target,lead,method="toto"}` is the fraction
of actuals inside the inclusive q10 to q90 interval. The linear comparator
has no interval, so it exports error only. Coverage estimates interval trust;
it does not establish that future forecasts will remain calibrated.

`promcast_selfaudit_mase{target,lead,method}` averages each forecast's absolute
median error divided by its original context's naive one-step MAE. The scale
uses adjacent observed pairs before tail filling; it never bridges gaps.
Zero or unavailable scales contribute coverage and sample counts but are
excluded from MASE. MASE is absent if no positive scales are available.

`promcast_selfaudit_samples{target,lead}` counts scored forecasts across
series, using the larger method count rather than adding paired forecasts.
Coverage and error are pooled across series for each target, lead and method.
All results cover the preceding 24 hours, including the cutoff timestamp,
using the forecast's due time. Missing actuals do not enter the denominator.

Matching accepts an actual within half a sample step of the due time. Each
origin and method is recorded once, each lead is scored once, and re-fetched
history can fill a previously missing actual. Duplicate forecasts preserve
the first prediction. Old forecasts expire after the window plus the maximum
lead; scored points expire after the window, including when fetches fail.

Only q10, q50 and q90 at the requested leads are retained. For a target capped
at `S` series, cycle interval `C` seconds, window `W` seconds, maximum lead `H`
seconds, `M` methods and `L` distinct leads, retained forecast records are
bounded by `S * M * (ceil((W + H) / C) + 1)` and scored points by that bound
times `L`. This also bounds label churn because every record has an expiry.
If the sample step exceeds the cycle interval, repeated origins reduce storage.
This is an entry bound, not a measured byte allocation.

