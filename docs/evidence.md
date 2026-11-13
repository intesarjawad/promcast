# Evidence

Decision rule budget: 1.0000 false alerts per series-week.
CPU model: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz.

## e1

- Series: 7
- Wall time: 55683.681 s
- Dataset SHA-256:
  - synthetic; no dataset files
- Caveats:
  - Synthetic scenario: linear_leak. Seed: 0.
  - Synthetic scenario: accelerating_leak. Seed: 0.
  - Synthetic scenario: daily_seasonal_peak. Seed: 0.
  - Synthetic scenario: step_change. Seed: 0.
  - Synthetic scenario: saturation_curve. Seed: 0.
  - Synthetic scenario: noisy_flat. Seed: 0.
  - Synthetic scenario: gappy_accelerating_leak. Seed: 0.

### Threshold incidents

| Method | n | Incidents | Recall | Precision | False alerts/week | Median lead (s) | Recall 95% CI | Median lead 95% CI (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| chronos-2-small | 7 | 5 | 0.4000 | 1.0000 | 0.0000 | 60.00 | [0.0000, 1.0000] | [60.0000, 60.0000] |
| ets | 7 | 5 | 0.0000 | 0.0000 | 0.0000 | unavailable | [0.0000, 0.0000] | unavailable |
| linear | 7 | 5 | 0.4000 | 1.0000 | 0.0000 | 240.00 | [0.0000, 1.0000] | [60.0000, 420.0000] |
| seasonal-naive | 7 | 5 | 0.0000 | 0.0000 | 0.0000 | unavailable | [0.0000, 0.0000] | unavailable |
| static-margin | 7 | 5 | 0.0000 | 0.0000 | 1.0204 | unavailable | [0.0000, 0.0000] | unavailable |
| toto-2.0-22m | 7 | 5 | 0.6000 | 1.0000 | 0.0000 | 60.00 | [0.0000, 1.0000] | [60.0000, 420.0000] |
| toto-2.0-4m | 7 | 5 | 0.8000 | 1.0000 | 0.0000 | 60.00 | [0.0000, 1.0000] | [60.0000, 420.0000] |
