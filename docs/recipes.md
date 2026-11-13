# Metric recipes

`promcast scan [URL]` finds capacity queries from metric names. It probes the
argument, `PROMETHEUS_URL`, or the local Prometheus-compatible endpoints in that
order. Scan only reads from the TSDB. On first use it downloads pinned model
weights and configuration into the local model cache and verifies checksums.
Later scans reuse those files; `--svg FILE` also writes the requested board export.

The bundled recipes are `disk`, `memory`, `pvc`, `windows-disk`,
`prometheus-self` and `prometheus-tsdb`. Their `source` URLs identify the exporter
source or metric documentation checked on 2026-11-05. The TSDB recipe returns no
series unless size-based retention is enabled. The thresholds are starting
points for configuration, not operational guarantees.

For an authenticated endpoint, use `promcast scan --config promcast.yaml` or
`promcast init --config promcast.yaml --out ./promcast`. The file supplies the
TSDB endpoint, headers, credential-file references and optional TLS CA bundle;
discovery still selects targets from recipes. Start with a full configuration
containing `targets: []`. See [authentication configuration](config.md#authenticated-discovery).
A positional URL cannot be combined with `--config`, so credentials stay with
the configured endpoint.

## Write a recipe

Put a YAML file in a local directory:

```yaml
name: disk
requires: [node_filesystem_avail_bytes, node_filesystem_size_bytes]
query: node_filesystem_avail_bytes / node_filesystem_size_bytes
direction: below
threshold: 0.05
unit: ratio
show: [instance, mountpoint]
source: https://github.com/prometheus/node_exporter/blob/master/collector/filesystem_common.go
```

Run `promcast scan --recipes ./recipes`. Files are validated strictly; extra
fields fail. A matching name replaces the bundled recipe. All required metric
names must appear in `/api/v1/label/__name__/values`. Requirements should name
the underlying gauges, with the primary gauge first. The primary gauge supplies
scrape timestamps and real sample counts. Queries must retain labels that
identify those samples. Use exporter-documented gauges rather than counters.

For a single query, use `promcast scan --query 'free_bytes' --below 0`.
`--max-series` defaults to 50 and accepts 1 through 500. Candidate selection
uses `bottomk` below a threshold or `topk` above it, anchored to one integer range
end. The footer gives the exact rerun when candidates were omitted. Scan prints
at most 20 forecast rows; `--json` includes every forecast and skipped series.
`docs/schema/scan.schema.json` defines that JSON. `--redact` replaces label
values deterministically in both formats; it does not change init files.

## History and output

Scan infers scrape intervals from raw timestamps, rounding the median gap to
whole seconds. It tries 5m, 1m and 15s grids without going finer than the measured
scrape interval. Each query covers at most 2,048 grid points. Forecasting needs
32 real samples and 32 observed grid slots. Fewer samples produce a warming-up
row. Between 32 and 63 observations the row states its short history and caps
both breach searches at that history. At a 15s step the normal 512-step horizon
is 2h 8m. Padding is masked outside the final patch, as measured in ADR 0007.
The strip shows successive risk-quantile values over the same row horizon.

Known recipe gauges bypass the unknown-metric counter heuristic, so a quiet,
constant file-descriptor ratio still produces a forecast. Ad hoc queries retain
counter detection. Missing model files download automatically to
`~/.cache/promcast/models`, or `PROMCAST_CACHE_DIR` when set. Offline use needs
a copy of the verified model folder in that cache. Progress goes to stderr and
JSON to stdout.
The scan footer estimates an upper bound for thirty days of replay origins
using inference throughput measured in the current run. It excludes network
time and report generation. Automatic replay measures its available history
and prints a more specific ETA before starting.

## Adopt and replay

`promcast init URL --out ./promcast` writes configuration, generated alert rules,
a Prometheus scrape snippet, a Compose service and Grafana provisioning files.
Existing files cause an error before any file is overwritten; `--force` permits
replacement. Loopback TSDB addresses become `host.docker.internal` for the
container, with an explicit host-gateway mapping. Review the scrape address for
the host running Prometheus. Mount `grafana/provisioning` at
`/etc/grafana/provisioning` and `grafana/dashboards` at
`/var/lib/grafana/dashboards` in an existing Grafana installation. Provisioning
uses Grafana defaults and includes forecast, breach, state and trust panels.

`promcast replay --auto --out ./replays` uses `promcast/promcast.yaml` when
present, or discovers recipe targets. `--config FILE` selects a different
configuration. It reads up to 30 days, trims unavailable leading history and
replays each candidate series separately under `<target>/<series-number>/`.
It measures an inference probe and prints ETA before replay starts. Series
without enough calibration and test history cannot be replayed. Existing
explicit `replay --source ...` commands retain their behavior.

## Test and contribute

Add `tests/fixtures/labels_<name>.json` as a metric-name API response and
`tests/fixtures/range_<name>.json` as a representative matrix response. Run:

```sh
uv run pytest tests/recipes tests/shell/test_discover.py tests/cli/test_scan.py
make check
```

Test matching, both candidate directions, scrape interval selection and threshold
behavior. Check each new query with `promtool check rules` and test it against
a real Prometheus. Include a source URL and the date its metric names were
checked. Open a pull request with the recipe, fixtures and verification results.

All bundled offline fixtures are synthetic. In particular,
`labels_stock_prometheus.json` contains the four documented metric names needed
by the two self recipes, not a complete live label capture.
