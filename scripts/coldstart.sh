#!/usr/bin/env bash
set -euo pipefail
exec </dev/null

usage() {
    echo 'Usage: scripts/coldstart.sh --path uvx|docker|oci [--wheel FILE | --pypi VERSION] [--prometheus URL | --stock]'
}
fail() { echo "$*" >&2; exit 2; }
path=uvx
wheel=
version=
release=false
url=
stock=false
demo=false
while (($#)); do
    case "$1" in
        --path|--wheel|--pypi|--prometheus)
            (($# >= 2)) && [[ -n "$2" ]] || { usage >&2; exit 2; }
            case "$1" in
                --path) path=$2 ;; --wheel) wheel=$2 ;;
                --pypi) version=$2; release=true ;; --prometheus) url=$2 ;;
            esac
            shift 2 ;;
        --stock) stock=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ "$path" =~ ^(uvx|docker|oci)$ ]] || fail 'Invalid --path'
[[ -z "$wheel" || -z "$version" ]] || fail 'Choose --wheel or --pypi'
[[ -z "$wheel" || "$path" == uvx ]] || fail '--wheel requires --path uvx'
[[ -z "$url" || "$stock" == false ]] || fail 'Choose --prometheus or --stock'
[[ "$path" != oci || ( -z "$url" && "$stock" == false ) ]] || fail 'OCI requires the demo stack'
root=$(cd "$(dirname "$0")/.." && pwd)
for command in curl jq; do command -v "$command" >/dev/null || fail "Missing command: $command"; done
if [[ "$path" == uvx ]]; then command -v uvx >/dev/null || fail 'Missing command: uvx'; fi
if [[ -z "$version" ]]; then
    version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/pyproject.toml")
fi
[[ -n "$version" ]] || fail 'Missing package version'
if [[ "$path" != uvx ]]; then
    command -v python3 >/dev/null || fail 'Missing command: python3'
    tag=$(python3 "$root/scripts/image_tag.py" "$version") || exit 2
    image="ghcr.io/intesarjawad/promcast:$tag"
fi
if [[ -n "$wheel" ]]; then
    [[ -f "$wheel" ]] || fail "Wheel not found: $wheel"
    wheel=$(cd "$(dirname "$wheel")" && pwd)/$(basename "$wheel")
fi
platform=$(uname -s)
work=$(mktemp -d)
pids=()
containers=()
compose=()
cleanup() {
    local status=$?
    for pid in ${pids[@]+"${pids[@]}"}; do kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
    for container in ${containers[@]+"${containers[@]}"}; do docker rm -f "$container" >/dev/null || true; done
    if ((${#compose[@]})); then
        if ! "${compose[@]}" down -v >/dev/null; then
            echo 'Compose cleanup failed' >&2
            if ((status == 0)); then status=1; fi
        fi
    fi
    rm -rf "$work"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export UV_CACHE_DIR="$work/uv" PROMCAST_CACHE_DIR="$work/models"
mkdir -p "$UV_CACHE_DIR" "$PROMCAST_CACHE_DIR"

request() { curl --fail --silent --show-error --connect-timeout 1 --max-time 3 "$@"; }
wait_ready() {
    local started=$SECONDS
    until request "$url/-/ready" >/dev/null 2>&1; do
        ((SECONDS - started < 120)) || fail 'Prometheus readiness timed out'
        sleep 1
    done
}
native_prometheus() {
    local os arch archive release expected actual directory
    case "$platform" in
        Darwin) os=darwin; arch=$(uname -m) ;;
        MINGW*|MSYS*|CYGWIN*) os=windows; arch=amd64 ;;
        *) fail "Unsupported native platform: $platform" ;;
    esac
    [[ "$arch" != x86_64 ]] || arch=amd64
    directory="prometheus-3.15.1.$os-$arch"
    archive="$directory.tar.gz"
    [[ "$os" != windows ]] || archive="$directory.zip"
    release=https://github.com/prometheus/prometheus/releases/download/v3.15.1
    curl -fLsS "$release/$archive" -o "$work/$archive"
    curl -fLsS "$release/sha256sums.txt" -o "$work/sha256sums.txt"
    expected=$(awk -v file="$archive" '$2 == file {print $1}' "$work/sha256sums.txt")
    [[ "$expected" =~ ^[[:xdigit:]]{64}$ ]] || fail 'Missing archive checksum'
    if command -v sha256sum >/dev/null; then
        actual=$(sha256sum "$work/$archive")
    else
        actual=$(shasum -a 256 "$work/$archive")
    fi
    [[ "${actual%% *}" == "$expected" ]] || fail 'Prometheus archive checksum mismatch'
    if [[ "$os" == windows ]]; then unzip -q "$work/$archive" -d "$work";
    else tar -xzf "$work/$archive" -C "$work"; fi
    local binary="$work/$directory" suffix=
    [[ "$os" != windows ]] || suffix=.exe
    if ! "$stock"; then
        command -v uv >/dev/null || fail 'Missing command: uv'
        UV_CACHE_DIR="$work/setup-uv" uv run --project "$root" python "$root/demo/scenario.py" --backfill \
            --out "$work/history.om" --anchor-file "$work/anchor"
        "$binary/promtool$suffix" tsdb create-blocks-from openmetrics "$work/history.om" "$work/data"
        UV_CACHE_DIR="$work/setup-uv" uv run --project "$root" python "$root/demo/scenario.py" --serve \
            --anchor-file "$work/anchor" >"$work/scenario.log" 2>&1 &
        pids+=("$!")
        cat > "$binary/prometheus.yml" <<'YAML'
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: scenario
    static_configs:
      - targets: ['localhost:9100']
  - job_name: prometheus
    static_configs:
      - targets: ['localhost:9090']
YAML
    fi
    (cd "$binary"; exec "./prometheus$suffix" --storage.tsdb.path="$work/data") >"$work/prometheus.log" 2>&1 &
    pids+=("$!")
}

oci_started=$(date +%s)
if [[ "$path" == oci ]]; then
    compose=(docker compose -p "promcast-coldstart-$$" -f "oci://ghcr.io/intesarjawad/promcast-demo:$tag")
    "${compose[@]}" up -d </dev/null
    up_s=$(($(date +%s) - oci_started))
    "$root/scripts/demo_check.sh" --json "$work/demo.json"
    cold=$(($(date +%s) - oci_started))
    firing=$(jq '.seconds_to_firing' "$work/demo.json")
    url=http://localhost:9090
elif [[ -z "$url" ]]; then
    demo=true
    url=http://localhost:9090
    case "$platform" in
        Linux)
            if "$stock"; then
                containers+=("$(docker run -d -p 127.0.0.1:9090:9090 prom/prometheus:v3.15.1)")
            else
                compose=(docker compose -p "promcast-coldstart-$$" -f "$root/deploy/demo/compose.yaml")
                "${compose[@]}" up -d prometheus scenario </dev/null
            fi ;;
        *) native_prometheus ;;
    esac
fi
wait_ready
if "$stock"; then
    started=$SECONDS
    until request --get "$url/api/v1/query" \
        --data-urlencode 'query=count_over_time(process_open_fds[30m])' |
        jq -e '.status == "success" and any(.data.result[]?; ( .value[1] | tonumber) >= 32)' >/dev/null; do
        ((SECONDS - started < 1200)) || fail 'Stock Prometheus sample collection timed out'
        sleep 5
    done
fi

scan() {
    if [[ "$path" == uvx ]]; then
        uvx --torch-backend cpu --from "${wheel:-promcast==$version}" promcast scan "$url" --json
    else
        local network=() docker_url=$url
        if [[ "$platform" == Linux ]]; then network=(--network host);
        else
            docker_url=${url/localhost/host.docker.internal}
            docker_url=${docker_url/127.0.0.1/host.docker.internal}
        fi
        docker run --rm ${network[@]+"${network[@]}"} "$image" scan "$docker_url" --json
    fi
}
if [[ "$path" == docker ]] && "$release"; then
    # A release benchmark includes its image pull; a local image stays available for testing.
    if docker image inspect "$image" >/dev/null 2>&1; then docker image rm "$image"; fi
fi
if [[ "$path" != oci ]]; then
    started=$(date +%s)
    scan >"$work/cold.json"
    cold=$(($(date +%s) - started))
else
    scan >"$work/cold.json"
fi
started=$(date +%s)
scan >"$work/warm.json"
warm=$(($(date +%s) - started))
rows=$(jq -ce '.rows | select(type == "array" and length > 0)' "$work/cold.json")
if "$stock"; then
    jq -e 'any(.rows[]; .target == "prometheus-self")' "$work/cold.json" >/dev/null || fail 'Missing prometheus-self row'
elif "$demo" || [[ "$path" == oci ]]; then
    jq -e '.rows[0].labels.mountpoint == "/demo"' "$work/cold.json" >/dev/null || fail 'Demo is not the first row'
fi
size_kb=$(du -sk "$UV_CACHE_DIR" "$PROMCAST_CACHE_DIR" | awk '{sum += $1} END {print sum}')
jq -n --arg path "$path" --arg platform "$platform" --arg prometheus "$(if "$stock"; then echo stock; else echo demo; fi)" \
    --argjson rows "$rows" --argjson cold "$cold" --argjson warm "$warm" --argjson bytes "$((size_kb * 1024))" \
    --argjson up "${up_s:-null}" --argjson firing "${firing:-null}" \
    '{path: $path, platform: $platform, prometheus: $prometheus, rows: $rows,
      seconds_to_board_cold: $cold, seconds_to_board_warm: $warm, download_bytes: $bytes,
      seconds_to_up: $up, seconds_to_firing: $firing}' > coldstart.json
case "$path" in
    uvx) budget_ok=$((cold <= 120 && warm <= 20)) ;;
    docker) budget_ok=$((cold <= 180)) ;;
    oci) budget_ok=$((firing < 240)) ;;
esac
if ((budget_ok == 0)); then
    echo 'Cold-start budget exceeded; see coldstart.json' >&2
    exit 1
fi
echo 'Cold-start check passed; wrote coldstart.json'
