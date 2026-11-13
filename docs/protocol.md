# Replay protocol, version 1

This document is pre-registered under the git tag `protocol-v1`. `scripts/evidence.py`
refuses to run while it differs from the tagged revision by a single byte. It has
three parts: the protocol, the per-corpus parameters, and the decision record behind
them (ADR 0005, copied verbatim).

## 1. Protocol

**Replay proves lead time** (protocol, all methods identical):

- Origins every `cycle` across the range; each forecast sees only samples at or before its origin.
- Split: the first 30 % of each series' timeline is calibration, the rest is test. Thresholds set with `--threshold auto` use the 99th (above) or 1st (below) percentile of the calibration split only.
- Incident: the actual stays past the threshold for at least 3 consecutive steps; incidents closer than one horizon merge. This definition is the primary score on every corpus, NAB included. With `--incidents FILE` (NAB `labels/combined_labels.json`), a secondary table treats each labelled anomaly timestamp as an incident start. Never use NAB's `combined_windows.json` start as the incident start: NAB centres each window on its label, so the window opens hours before the anomaly (verified 2026-11-04: `ec2_cpu_utilization_24ae8d` window 2014-02-26 13:45 to 2014-02-27 06:25 around the label 2014-02-26 22:05), which would turn early noise into fake lead.
- Firing time: the origin at which the breach condition has held for `for_cycles` consecutive origins. Origins where the actual is already past the threshold are ignored.
- Caught early: firing time in `[incident.start - horizon, incident.start)`. Lead = `incident.start - firing_ts`. Firing inside the incident counts as "late", not caught.
- An episode is matched when its firing time lies in `[incident.start - horizon, incident.end]` of some incident (late firings count as matched for precision: the problem was real). Unmatched episodes are false alerts.
- Precision = matched firing episodes / all episodes; recall = incidents caught early / incidents; reset time = episode end minus incident end; false alerts per week = unmatched episodes per series-week.
- Pooling: an evidence run (`scripts/evidence.py`) pools calibration episodes and incidents across all series of a corpus and chooses one operating point per method for that corpus; `promcast replay` on a single target chooses per target. Series with no calibration incident still contribute their false alerts and series-weeks. If a pooled calibration split has zero incidents, the operating point is the one with the fewest calibration false alerts, then the smallest `warn`, and the report prints `n_incidents_cal=0`.
- Operating point per method: sweep `warn` in {10m, 20m, 30m, 45m, 60m, 90m, 120m} (capped at horizon), quantile in {0.5, 0.7, 0.9} for probabilistic methods, window in {1h, 6h, 24h} for linear, and margin `k` in {0.5, 1, 2, 3} calibration-split standard deviations for the static warning threshold (fires when the last observed value is within `k` sd of the threshold; this is what most teams run today as a "warning" level, and the baseline a forecast must beat); choose the highest calibration recall with false alerts per week ≤ `--budget` (default 1.0); ties go to longer median lead, then smaller `warn`.
- Statistics: 95 % cluster bootstrap intervals (2,000 resamples, seed 0, resampling whole series with replacement, because incidents within one series are correlated) on median lead and recall. Paired comparison versus linear uses warning time per incident (lead if caught early, else 0) over all incidents, so it never conditions on incidents both methods caught. Test-split false alerts per week are printed beside the calibration budget for every method; a method whose test rate exceeds 1.5 times the budget is flagged in the table. `n` (incidents, series, series-weeks) printed on every row.
- Forecast quality beside alert quality: interval coverage of q10 to q90 and weighted quantile loss per lead.
- Corpora: E1 synthetic scenario suite with known ground truth; E2 BOOM gauge series (in-domain, contamination caveat printed on every table); E3 NAB AWS CloudWatch and machine metrics (MIT, public data, which Toto 2.0's base pretraining excluded): threshold incidents as the primary table, labelled anomaly timestamps as a secondary table, with the caveat that NAB labels have documented flaws (Wu and Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress", IEEE TKDE 35, 2023; arXiv 2020) and many are not forecastable by any method; E4 a homelab node_exporter history, recorded from day 0 (aggregate table only). Per-corpus step, context, horizon, season, cycle and origin caps are fixed in section 2 so no run depends on a judgement call.
- Methods: Toto-2.0-22m and 4m, Chronos-2 small (`autogluon/chronos-2-small`, 28M parameters, Apache-2.0, replay only through the optional `compare` extra; it shows whether any gain belongs to Toto or to foundation models in general), `predict_linear`, seasonal-naive, ETS, static warning threshold. Chronos-2 small rows on a corpus carry a contamination caveat if the training-data audit in ADR 0010 finds that corpus in its training sets; if that model's contract check fails, the method is dropped and the README says so.
- Pre-registration: this section, the parameter table in section 2 and ADR 0005 are copied to `docs/protocol.md` and tagged `protocol-v1` before the first E3 or E4 run. `evidence.py` refuses to run E3 or E4 unless `docs/protocol.md` is byte-identical to its `protocol-v1` version, and prints the tag's SHA in `docs/evidence.md`. A protocol change after results exist needs a new tag, an ADR, and both versions' headline rows published.
- Pre-registered decision rule (primary endpoint): paired warning time per incident, Toto-2.0-22m minus `predict_linear`, each at its calibration-chosen operating point, averaged over all E3 threshold incidents in the test split. Its interval is a 95 % percentile cluster-bootstrap interval on that mean difference (2,000 resamples, seed 0), resampling whole series with replacement. Verdict `beats` iff the interval's lower bound > 0 and Toto's test false-alert rate ≤ 1.5 × budget; `ties` iff the interval contains 0; `loses` otherwise (upper bound < 0, or a positive interval with Toto's test false-alert rate above 1.5 × budget). The same rule is computed on E4 as a secondary endpoint. `evidence.py` prints the verdict word on the first line of `docs/evidence.md`; README copy branches on that word only.
- Publishing rule: the README shows the E3 and E4 tables first. If Toto does not beat linear at the budget, the README says so in the first evidence sentence, and the default quantile is not tuned on test data.

## 2. Parameters per corpus

Replay parameters are fixed per corpus (steps are in the corpus's own sampling interval; replay cycle = one origin every `cycle` steps; `for_cycles` = 2 everywhere; direction `above` unless stated):

| Corpus | step | context | horizon | season | cycle | threshold | size cap |
|---|---|---|---|---|---|---|---|
| E1 synthetic | 1m | 1440 | 120 | 1440 | 5 | per scenario, defined in `scenarios.py` (direction per scenario) | 8 scenarios x 14 d |
| E2 BOOM | native `freq` | 1024 | 64 | 1 d in steps if ≤ 2048, else none (seasonal-naive and ETS record their fallback) | 16 | auto | 1 variate per gauge series, last 4,096 steps |
| E3 NAB | 5m (native) | 576 (2 d) | 24 (2 h) | 288 | 1 | auto | the listed files, full length |
| E4 homelab | 5m | 1152 (4 d) | 72 (6 h) | 288 | 1 | auto (`below` for filesystem and memory) | everything recorded |

Gaps (E3 and E4): each CSV goes onto its 5m grid through `promcast.core.series.align`, the function that aligns a live `query_range` result, with the grid ending at the file's last timestamp. A row off that grid is ignored, a repeated timestamp keeps its last row, a slot with no row is missing, and an infinite value rejects the file. Missing slots are never filled, interpolated or dropped from the timeline. Replay then treats them as a live cycle would: no method forecasts from an origin whose own sample is missing, and a model context goes through the shared tail rule (ADR 0007), so a forecast is skipped rather than invented.

## 3. Decision record

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
