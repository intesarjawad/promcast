# Homelab recorder

I run this recorder on Linux x86_64 with Docker 29 and Compose 5. It records
host metrics every 30 seconds into a named volume with 400 days of retention.
I leave it running for at least 21 days before the homelab replay.

## Linux recorder: one-time setup

From the repository root, I set the Linux host's LAN or VPN IPv4 address and
create a private targets directory outside the checkout:

```sh
export RECORDER_BIND_ADDRESS='<linux-lan-or-vpn-ip>'
export RECORDER_TARGETS_DIR="$HOME/.config/promcast/targets"
mkdir -p "$RECORDER_TARGETS_DIR"
chmod 755 "$RECORDER_TARGETS_DIR"
export RECORDER_URL="http://${RECORDER_BIND_ADDRESS}:9090"
```

I replace the address placeholder before running these commands. Without
`RECORDER_BIND_ADDRESS`, Prometheus listens on `127.0.0.1:9090` and I use
`export RECORDER_URL=http://localhost:9090` on the recorder itself. I allow
access to port 9090 only from my LAN or VPN.

The included node_exporter 1.12.1 container records the Linux host, so I do not
install a second exporter there. Both services use host networking.
node_exporter uses the host PID namespace and mounts `/` read-only at `/host`
with `--path.rootfs=/host`, preserving the filesystem label `mountpoint="/"`.
Its listener stays on `127.0.0.1:9100`.

## Mac: one-time setup

On the Mac, I install node_exporter 1.12.1 and start its service:

```sh
brew install node_exporter
node_exporter --version
brew services start node_exporter
```

I confirm the version is 1.12.1 before recording. I use a stable LAN/VPN address
or DNS name for the Mac and allow TCP 9100 from the recorder. On the Linux
recorder, I replace the address below and create its discovery file:

```sh
cat > "$RECORDER_TARGETS_DIR/mac.json" <<'JSON'
[
  {"targets": ["<mac-lan-or-vpn-address>:9100"], "labels": {"os": "darwin"}}
]
JSON
chmod 644 "$RECORDER_TARGETS_DIR/mac.json"
```

I add further hosts with separate `*.json` files in the same directory, using
the same format and `os: linux` for Linux hosts running node_exporter 1.12.1.
I use each address only once, including the recorder's existing loopback
target. The directory is mounted read-only and Prometheus discovers changes
without editing `prometheus.yml` or restarting. I keep these files readable by
the container user.

For an optional Windows host, I install windows_exporter as a service, allow
TCP 9182 from the recorder, and add its address on port 9182 with `os: windows`.
I record it but exclude its different metric names from the first homelab replay.

## Start and verify

On the Linux recorder, from the repository root:

```sh
docker compose -f homelab/compose.yml config -q
docker compose -f homelab/compose.yml up -d
docker compose -f homelab/compose.yml ps
docker compose -f homelab/compose.yml exec prometheus prometheus --version
docker compose -f homelab/compose.yml exec prometheus promtool check config /etc/prometheus/prometheus.yml
curl -fsS "$RECORDER_URL/api/v1/status/flags" | jq -e '.data["storage.tsdb.retention.time"] == "400d"'
curl -s "$RECORDER_URL/api/v1/query?query=up" | jq '.data.result | length'
curl -fsSG "$RECORDER_URL/api/v1/query" --data-urlencode 'query=up' | jq -e '.status == "success" and (.data.result | length > 0) and all(.data.result[]; .value[1] == "1")'
```

After discovery and scraping, the count must match the enabled hosts, including
the recorder. The final check requires every host to be up. I inspect failures
with `docker compose -f homelab/compose.yml logs --tail=100` and
`curl -fsS "$RECORDER_URL/api/v1/targets" | jq '.data.activeTargets[] | {scrapeUrl, health, lastError}'`.
There is no Prometheus self-scrape, so the host count includes only exporters.

I export the same `RECORDER_URL` in each shell that queries this recorder.
I keep the named volume when stopping or recreating containers; I never use
`docker compose down -v` while collecting history.

## Replay queries

I retain the original metric names and mountpoint labels. These are the exact
queries for the homelab replay:

```promql
node_filesystem_avail_bytes{mountpoint="/"}
node_filesystem_avail_bytes{mountpoint="/System/Volumes/Data"}
node_load1
rate(node_network_receive_bytes_total{device!~"lo.*"}[5m])
node_memory_MemAvailable_bytes
```

The first filesystem query selects Linux root; the second selects the macOS
APFS data volume. Memory availability is Linux-only. I use `--threshold auto`
for replay and the `below` direction for filesystem and memory series.
After at least five minutes of scraping, I verify each applicable query returns
series, substituting each expression below:

```sh
curl -fsSG "$RECORDER_URL/api/v1/query" \
  --data-urlencode 'query=node_filesystem_avail_bytes{mountpoint="/"}' \
  | jq -e '.status == "success" and (.data.result | length > 0)'
```

Raw series stay on my machines. I replace host labels with anonymous names
before publishing aggregates. The demo's `record.override.yml` serves a
separate recording setup; I do not need it for this recorder.
