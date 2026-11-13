# 0005: Score replay on a fixed protocol shared by every method

Status: accepted, 2026-11-09.

## Context

A forecast-based alert is only worth running if it warns earlier than the
rules a team already has, at the same false-alert cost. Comparing methods at
one shared quantile, or tuning on the data that is then scored, would make any
method look better than it is. The protocol below fixes every choice before a
result exists, and `core.replay`, `core.operating` and `core.bootstrap`
implement it.

## Decision

- **Split.** The first 30 % of each series' timeline is calibration, the rest
  is test. `--threshold auto` takes the 99th (above) or 1st (below) percentile
  of the calibration split only.
- **Origins.** One origin every `cycle` steps, starting once a full context
  exists. Each forecast receives a copy of the context ending at its origin and
  the same tail rule as live serving (ADR 0007), so no method sees a later
  sample.
- **Firing time.** The origin at which the breach condition has held for
  `for_cycles` consecutive origins, matching the generated alert's `for:`.
  Lead is measured from that firing time, not from the first origin where the
  condition held. Origins where the actual is already past the threshold are
  ignored.
- **Incidents.** The actual stays past the threshold for at least 3
  consecutive steps; incidents closer than one horizon merge. This is the
  primary score on every corpus. On NAB a secondary table uses each timestamp
  in `labels/combined_labels.json` as an incident start. NAB's
  `combined_windows.json` is never an incident start: each window is centred
  on its label, so it opens hours before the anomaly and would turn early noise
  into lead.
- **Budget.** Each method's operating point is chosen on calibration only: the
  highest recall with false alerts per series-week at or below `--budget`
  (default 1.0), ties to the longer median lead, then the smaller `warn`. If no
  point meets the budget the run fails rather than report an over-budget
  point. If the calibration split has no incident, the point with the fewest
  calibration false alerts, then the smallest `warn`, is chosen and the report
  prints `n_incidents_cal=0`. An evidence run pools calibration across all
  series of a corpus; series without incidents still add their false alerts
  and series-weeks.
- **Static-margin baseline.** A static warning threshold fires when the last
  observed value is within `k` calibration-split standard deviations of the
  threshold, `k` in {0.5, 1, 2, 3}. This is the warning level most teams run
  today; a forecast has to beat it, not only `predict_linear`.
- **Baseline fallbacks.** Seasonal-naive without a full season, and ETS with
  no season or fewer than two seasons of history, fall back to a non-seasonal
  form and the report records the fallback. ETS cannot take NaN, so interior
  gaps are linearly interpolated for that baseline only.
- **Statistics.** 95 % percentile cluster-bootstrap intervals, 2,000
  resamples, seed 0, resampling whole series with replacement, because
  incidents within one series are correlated. The paired comparison against
  `predict_linear` uses warning time per incident (lead if caught early, else
  0) over all incidents, so it never conditions on incidents both methods
  caught.

## Consequences

Every method, including the model under test, is scored at its own
calibration-chosen operating point under one false-alert budget, and the test
split is never used for selection. A method whose test false-alert rate
exceeds 1.5 times the budget is flagged rather than hidden. The protocol is
frozen in `docs/protocol.md` under the `protocol-v1` tag; changing it after
results exist needs a new tag, an ADR, and both versions' headline rows.
