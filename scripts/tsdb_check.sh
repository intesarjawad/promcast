#!/bin/sh
set -eu

usage() {
    echo 'Usage: scripts/tsdb_check.sh vm|mimir'
}

if [ "$#" -ne 1 ]; then
    usage >&2
    exit 2
fi

backend=$1
case "$backend" in
    vm)
        base_url=${VM_URL:-http://localhost:8428}
        org_id=
        ;;
    mimir)
        base_url=${MIMIR_URL:-http://localhost:8080/prometheus}
        org_id=${MIMIR_ORG_ID:-demo}
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

for dependency in curl jq; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        echo "Missing required command: $dependency" >&2
        exit 2
    fi
done

request() {
    if [ -n "$org_id" ]; then
        curl --fail --silent --show-error --connect-timeout 1 --max-time 5 \
            -H "X-Scope-OrgID: $org_id" "$@"
    else
        curl --fail --silent --show-error --connect-timeout 1 --max-time 5 "$@"
    fi
}

timeout_s=${TSDB_CHECK_TIMEOUT_S:-120}
deadline=$(($(date +%s) + timeout_s))

until request --get "$base_url/api/v1/query" \
    --data-urlencode 'query=vector(1)' |
    jq -e '.status == "success" and .data.resultType == "vector"' >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$backend: query API not ready at $base_url" >&2
        exit 1
    fi
    sleep 2
done

# Remote write is asynchronous, so wait for the scraped series before ranging over it.
until names=$(request --get "$base_url/api/v1/label/__name__/values") &&
    printf '%s' "$names" | jq -e '.status == "success" and (.data | index("up")) != null' >/dev/null 2>&1 &&
    count=$(request --get "$base_url/api/v1/query" --data-urlencode 'query=count(up)') &&
    printf '%s' "$count" | jq -e '.status == "success" and (.data.result[0].value[1] | tonumber) > 0' >/dev/null 2>&1 &&
    end=$(date +%s) &&
    start=$((end - 60)) &&
    series=$(request --get "$base_url/api/v1/query_range" \
        --data-urlencode 'query=up' \
        --data-urlencode "start=$start" \
        --data-urlencode "end=$end" \
        --data-urlencode 'step=15') &&
    printf '%s' "$series" | jq -e '
        .status == "success" and .data.resultType == "matrix" and
        any(.data.result[]?; (.values | length) >= 2)' >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$backend: no scraped series at $base_url" >&2
        exit 1
    fi
    sleep 2
done

echo "$backend: ok"