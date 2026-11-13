"""Synthetic disk history and live metrics, using only the standard library."""

import argparse
import math
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TextIO

MINUTE_S = 60
DAY_S = 24 * 60 * MINUTE_S
CAPACITY_BYTES = 500_000_000_000
BASE_FREE_BYTES = 400_000_000_000
SEASON_BYTES = 20_000_000_000
DRIFT_BYTES_PER_DAY = 1_000_000_000
CYCLE_S = 3 * 60 * MINUTE_S
LEAK_LEAD_S = 25 * MINUTE_S
FULL_AFTER_S = 65 * MINUTE_S
CLEANUP_AFTER_S = FULL_AFTER_S + 10 * MINUTE_S
# Make acceleration visible at the anchor, inside the one-hour forecast warning.
LOGISTIC_MIDPOINT_S = 40 * MINUTE_S
LOGISTIC_SCALE_S = 10 * MINUTE_S
FILESYSTEM_LABELS = '{mountpoint="/demo",fstype="ext4",device="demo"}'
SCRAPE_LABELS = 'instance="scenario:9100",job="scenario"'
METRICS = (
    "demo_disk_free_bytes",
    "node_filesystem_avail_bytes",
    "node_filesystem_size_bytes",
)


def no_leak_bytes(t: float, anchor_ts: float) -> float:
    """Return free bytes with daily seasonality and a slow downward drift."""
    elapsed_s = t - anchor_ts
    baseline = (
        BASE_FREE_BYTES
        + SEASON_BYTES * math.sin(2 * math.pi * elapsed_s / DAY_S)
        - DRIFT_BYTES_PER_DAY * elapsed_s / DAY_S
    )
    return min(CAPACITY_BYTES, max(0.0, baseline))


def disk_free_bytes(t: float, anchor_ts: float) -> float:
    """Return a repeating logistic leak, full-disk hold and cleanup."""
    baseline = no_leak_bytes(t, anchor_ts)
    elapsed_s = t - (anchor_ts - LEAK_LEAD_S)
    if elapsed_s < 0:
        return baseline
    phase_s = elapsed_s % CYCLE_S
    if phase_s >= CLEANUP_AFTER_S:
        return baseline
    if phase_s >= FULL_AFTER_S:
        return 0.0

    def logistic(age_s: float) -> float:
        return 1 / (1 + math.exp(-(age_s - LOGISTIC_MIDPOINT_S) / LOGISTIC_SCALE_S))

    # Normalize the endpoints so the leak starts continuously and fills at 65 min.
    progress = (logistic(phase_s) - logistic(0)) / (logistic(FULL_AFTER_S) - logistic(0))
    return baseline * (1 - progress)


def metric_sample(name: str, value: float, timestamp: float | None = None) -> str:
    """Render samples; timestamped history includes labels added by live scrapes."""
    labels = [] if name == METRICS[0] else [FILESYSTEM_LABELS[1:-1]]
    if timestamp is not None:
        labels.append(SCRAPE_LABELS)
    label_set = "{" + ",".join(labels) + "}" if labels else ""
    suffix = "" if timestamp is None else f" {timestamp:.6f}"
    return f"{name}{label_set} {value:.17g}{suffix}\n"


def write_history(stream: TextIO, anchor_ts: float, hours: int) -> None:
    """Write minute samples through the anchor, grouped by metric family."""
    if hours <= 0:
        raise ValueError("hours must be a positive integer")
    for name in METRICS:
        stream.write(f"# TYPE {name} gauge\n")
        for offset in range(-hours * 60, 1):
            timestamp = anchor_ts + offset * MINUTE_S
            value = (
                float(CAPACITY_BYTES)
                if name == METRICS[2]
                else disk_free_bytes(timestamp, anchor_ts)
            )
            stream.write(metric_sample(name, value, timestamp))
    stream.write("# EOF\n")


def backfill(out: Path, anchor_file: Path, hours: int, anchor_ts: float) -> None:
    """Persist the shared anchor and its matching OpenMetrics history."""
    if hours <= 0 or not math.isfinite(anchor_ts):
        raise ValueError("hours must be positive and anchor must be finite")
    if out.exists() and anchor_file.exists():
        try:
            valid_anchor = math.isfinite(float(anchor_file.read_text(encoding="utf-8")))
        except ValueError:
            valid_anchor = False
        if valid_anchor and out.read_bytes().endswith(b"# EOF\n"):
            return
    if out.exists() or anchor_file.exists():
        raise ValueError("incomplete demo history; restore both history and anchor")
    out.parent.mkdir(parents=True, exist_ok=True)
    anchor_file.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        write_history(stream, anchor_ts, hours)
    temporary_anchor = anchor_file.with_name(anchor_file.name + ".tmp")
    temporary_anchor.write_text(repr(anchor_ts) + "\n", encoding="utf-8")
    temporary_anchor.replace(anchor_file)


def metrics(t: float, anchor_ts: float) -> bytes:
    """Render both free-space forms from one live sample."""
    free = disk_free_bytes(t, anchor_ts)
    lines: list[str] = []
    for name, value in zip(METRICS, (free, free, float(CAPACITY_BYTES)), strict=True):
        lines.append(f"# TYPE {name} gauge\n")
        lines.append(metric_sample(name, value))
    return ("".join(lines) + "# EOF\n").encode("utf-8")


def create_server(
    address: tuple[str, int], anchor_ts: float, clock: Callable[[], float] = time.time
) -> ThreadingHTTPServer:
    """Create an exporter with an injectable clock and listen address."""
    if not math.isfinite(anchor_ts):
        raise ValueError("anchor must be finite")

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/metrics":
                self.send_error(404)
                return
            payload = metrics(clock(), anchor_ts)
            self.send_response(200)
            self.send_header(
                "Content-Type", "application/openmetrics-text; version=1.0.0; charset=utf-8"
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return ThreadingHTTPServer(address, MetricsHandler)


def main() -> None:
    """Backfill history or serve the same anchored scenario on port 9100."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--backfill", action="store_true")
    modes.add_argument("--serve", action="store_true")
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--out", type=Path, default=Path("/work/history.om"))
    parser.add_argument("--anchor-file", type=Path, default=Path("/work/anchor"))
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error("--hours must be a positive integer")
    if args.backfill:
        backfill(args.out, args.anchor_file, args.hours, time.time())
    else:
        anchor_ts = float(args.anchor_file.read_text(encoding="utf-8").strip())
        with create_server(("", 9100), anchor_ts) as server:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                server.server_close()


if __name__ == "__main__":
    main()
