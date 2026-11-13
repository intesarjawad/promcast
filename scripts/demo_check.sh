#!/bin/sh
set -eu

usage() {
    echo 'Usage: scripts/demo_check.sh [--full] [--json FILE]'
    echo '       scripts/demo_check.sh --screenshot FILE'
}

full=false
output=demo.json
screenshot=
checking=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --full) full=true; checking=true; shift ;;
        --json)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                usage >&2
                exit 2
            fi
            output=$2
            checking=true
            shift 2
            ;;
        --screenshot)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                usage >&2
                exit 2
            fi
            screenshot=$2
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if [ -n "$screenshot" ] && "$checking"; then
    usage >&2
    exit 2
fi

grafana_url=${GRAFANA_URL:-http://localhost:3000}

if [ -n "$screenshot" ]; then
    # Capture whatever the dashboard shows now, so a shot before the first cycle shows the
    # warming-up text. The session is private to this run and is the only one closed.
    cli=$(command -v playwright-cli || true)
    if [ -z "$cli" ] && [ -x "$HOME/.local/bin/playwright-cli" ]; then
        cli=$HOME/.local/bin/playwright-cli
    fi
    if [ -z "$cli" ]; then
        echo 'Missing required command: playwright-cli' >&2
        exit 2
    fi
    session=promcast-demo-$$
    trap '"$cli" "-s=$session" close >/dev/null 2>&1 || true' EXIT
    "$cli" "-s=$session" open "$grafana_url/d/promcast-demo?orgId=1&kiosk" >/dev/null
    "$cli" "-s=$session" resize 1440 900 >/dev/null
    # Grafana draws panels after the page loads; wait inside the page for queries to settle.
    "$cli" "-s=$session" eval '() => new Promise((resolve) => setTimeout(resolve, 8000))' >/dev/null
    "$cli" "-s=$session" screenshot --filename "$screenshot" >/dev/null
    echo "Wrote $screenshot"
    exit 0
fi

for dependency in curl jq; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        echo "Missing required command: $dependency" >&2
        exit 2
    fi
done

prometheus_url=${PROMETHEUS_URL:-http://localhost:9090}
promcast_url=${PROMCAST_URL:-http://localhost:8787}
poll_s=5
# The first coverage sample needs the ten-minute lead to mature after startup.
budget_s=900
if "$full"; then
    budget_s=3600
fi
started=$(date +%s)
stack_started=null
forecast_ready=false
grafana_ready=false
audit_ready=false
promcast_firing=null
linear_firing=null
disk_full_firing=null
seconds_to_firing=null

request() {
    curl --fail --silent --show-error --connect-timeout 1 --max-time 2 "$@" 2>/dev/null
}

write_result() {
    jq -n \
        --argjson full "$full" \
        --argjson seconds_to_firing "$seconds_to_firing" \
        --argjson promcast "$promcast_firing" \
        --argjson linear "$linear_firing" \
        --argjson disk_full "$disk_full_firing" \
        --argjson forecast_ready "$forecast_ready" \
        --argjson grafana_ready "$grafana_ready" \
        --argjson audit_ready "$audit_ready" \
        --argjson timed_out "$1" '
        {seconds_to_firing: $seconds_to_firing,
         first_firing: {PromcastBreachSoon: $promcast,
                        DemoLinearBreachSoon: $linear, DemoDiskFull: $disk_full},
         q90_data: $forecast_ready, grafana_health: $grafana_ready,
         selfaudit_coverage: $audit_ready, timed_out: $timed_out}
        + if $full then {
            promcast_lead_s: (if $disk_full != null and $promcast != null
                                and $promcast < $disk_full then $disk_full - $promcast
                              else null end),
            linear_lead_s: (if $disk_full != null and $linear != null
                              and $linear < $disk_full then $disk_full - $linear
                            else null end)
          } else {} end' > "$output"
}

while :; do
    observed=$(date +%s)
    if [ "$stack_started" = null ]; then
        runtime=$(request "$prometheus_url/api/v1/status/runtimeinfo") || runtime='{}'
        stack_started=$(printf '%s' "$runtime" | jq -r '
            if .status == "success" then
                .data.startTime | sub("\\.[0-9]+Z$"; "Z") | fromdateiso8601
            else null end' 2>/dev/null) || stack_started=null
    fi
    if [ "$stack_started" != null ]; then
        # Anchor the query grid to process startup so late checks recover the same history.
        history=$(request --get "$prometheus_url/api/v1/query_range" \
            --data-urlencode 'query=ALERTS{alertstate="firing",alertname=~"PromcastBreachSoon|DemoLinearBreachSoon|DemoDiskFull"}' \
            --data-urlencode "start=$stack_started" \
            --data-urlencode "end=$observed" \
            --data-urlencode 'step=1') || history='{}'
        first_firing=$(printf '%s' "$history" | jq -r '
            if .status == "success" and .data.resultType == "matrix" then
                [.data.result[]? | .metric.alertname as $name |
                 .values[]? | select(.[1] == "1") | {name: $name, time: .[0]}] |
                group_by(.name)[] | [.[0].name, (map(.time) | min | floor)] | @tsv
            else empty end' 2>/dev/null) || first_firing=''
        while read -r name fired_at; do
            case "$name" in
                PromcastBreachSoon)
                    if [ "$promcast_firing" = null ]; then
                        promcast_firing=$fired_at
                        seconds_to_firing=$((fired_at - stack_started))
                    fi
                    ;;
                DemoLinearBreachSoon)
                    if [ "$linear_firing" = null ]; then linear_firing=$fired_at; fi
                    ;;
                DemoDiskFull)
                    if [ "$disk_full_firing" = null ]; then disk_full_firing=$fired_at; fi
                    ;;
            esac
        done <<EOF
$first_firing
EOF
    fi

    if ! "$forecast_ready"; then
        forecast=$(request --get "$promcast_url/api/v1/query_range" \
            --data-urlencode 'query=promcast_forecast_q90{target="demo_disk_free"}' \
            --data-urlencode "start=$observed" \
            --data-urlencode "end=$((observed + 7200))" \
            --data-urlencode 'step=60') || forecast='{}'
        if printf '%s' "$forecast" | jq -e '
            .status == "success" and .data.resultType == "matrix" and
            any(.data.result[]?; .metric.target == "demo_disk_free" and
                any(.values[]?; (.[1] | tonumber? // null) as $value |
                    $value != null and ($value | isfinite) and
                    ($value | isnan | not)))' >/dev/null 2>&1; then
            forecast_ready=true
        fi
    fi
    if ! "$grafana_ready"; then
        health=$(request "$grafana_url/api/datasources/uid/promcast/health") || health='{}'
        if printf '%s' "$health" | jq -e '.status == "OK"' >/dev/null 2>&1; then
            grafana_ready=true
        fi
    fi

    if ! "$audit_ready"; then
        metrics=$(request "$promcast_url/metrics") || metrics=''
        if printf '%s\n' "$metrics" | awk '
            /^promcast_selfaudit_coverage_ratio\{/ && /target="demo_disk_free"/ && /method="toto"/ {
                value = $NF
                if (value ~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ && value >= 0 && value <= 1)
                    found = 1
            }
            END { exit !found }'; then
            audit_ready=true
        fi
    fi

    elapsed=$(($(date +%s) - started))
    if [ "$elapsed" -ge "$budget_s" ]; then
        write_result true
        echo "Demo check timed out after ${budget_s}s; see $output" >&2
        exit 1
    fi
    if "$forecast_ready" && "$grafana_ready" && "$audit_ready"; then
        if { ! "$full" && [ "$promcast_firing" != null ]; } ||
            { "$full" && [ "$disk_full_firing" != null ]; }; then
            write_result false
            echo "Demo check passed; wrote $output"
            exit 0
        fi
    fi
    sleep "$poll_s"
done
