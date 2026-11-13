"""Check Promcast's linear breach time against Prometheus's own predict_linear.

Both fits cover the same window ending at Promcast's forecast origin; the check passes
when the two times to breach agree within 1 %. Standard library only, so it runs on a
CI runner beside the Compose stack without installing the project.
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

TOLERANCE = 0.01
# Prometheus API responses are untyped JSON; instant() validates the shape it reads.
Fetch = Callable[[str], Any]


@dataclass(frozen=True)
class Target:
    """The demo target as Promcast is configured to forecast it."""

    name: str
    query: str
    threshold: float
    window_s: int
    step_s: int = 60


def fetch_json(url: str) -> Any:  # noqa: ANN401
    """GET a Prometheus API URL and decode its JSON body."""
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
        return json.load(response)


def instant(fetch: Fetch, base: str, query: str, at: float | None = None) -> float | None:
    """Return the single value of an instant query, or None when it has no series."""
    parameters = {"query": query} | ({} if at is None else {"time": f"{at:.3f}"})
    document = fetch(f"{base}/api/v1/query?{urllib.parse.urlencode(parameters)}")
    try:
        if document["status"] != "success":
            raise ValueError(f"Prometheus rejected {query!r}: {document!r}")
        result = document["data"]["result"]
        if len(result) > 1:
            raise ValueError(f"{query!r} must return at most one series, got {len(result)}")
        return float(result[0]["value"][1]) if result else None
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"unexpected response to {query!r}: {document!r}") from error


def predict_linear_breach(target: Target) -> str:
    """Solve predict_linear for the threshold crossing, as an absolute timestamp."""
    # A subquery resamples at Promcast's step; raw scrapes are denser after the demo starts,
    # which weights recent samples more and skews a regression over a mixed-density window.
    window = f"{target.query}[{target.window_s}s:{target.step_s}s]"
    return f"time() + ({target.threshold!r} - predict_linear({window}, 0)) / deriv({window})"


def compare(fetch: Fetch, base: str, target: Target) -> tuple[float, float] | None:
    """Return Promcast's and predict_linear's times to breach from one origin, once ready."""
    origin = instant(
        fetch, base, f'promcast_forecast_origin_timestamp_seconds{{target="{target.name}"}}'
    )
    breach = instant(
        fetch, base, f'promcast_breach_timestamp_seconds{{target="{target.name}",method="linear"}}'
    )
    if origin is None or breach is None:
        return None
    expected = instant(fetch, base, predict_linear_breach(target), origin)
    if expected is None:
        raise ValueError(f"predict_linear has no breach for {target.query} at {origin:.0f}")
    return breach - origin, expected - origin


def agrees(promcast_s: float, prometheus_s: float) -> bool:
    """Accept a forecast breach within 1 % of predict_linear's own time to breach."""
    return prometheus_s > 0 and abs(promcast_s - prometheus_s) <= TOLERANCE * prometheus_s


def main(argv: list[str] | None = None, fetch: Fetch = fetch_json) -> int:
    """Wait for a linear breach forecast, then compare it once with predict_linear."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--target", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--window-s", type=int, required=True)
    parser.add_argument("--step-s", type=int, default=60)
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--poll-s", type=float, default=5)
    arguments = parser.parse_args(argv)
    target = Target(
        arguments.target, arguments.query, arguments.threshold, arguments.window_s, arguments.step_s
    )
    base = arguments.prometheus.rstrip("/")
    deadline = time.monotonic() + arguments.timeout_s
    while True:
        try:
            leads = compare(fetch, base, target)
        except (OSError, ValueError) as error:
            sys.stderr.write(f"{error}\n")
            return 1
        if leads is not None:
            break
        if time.monotonic() >= deadline:
            sys.stderr.write(
                f"no linear breach forecast for {target.name} within {arguments.timeout_s:.0f} s;"
                " check that the demo leak is running and read Promcast's logs\n"
            )
            return 1
        time.sleep(arguments.poll_s)
    promcast_s, prometheus_s = leads
    line = f"time to breach: promcast {promcast_s:.1f} s, predict_linear {prometheus_s:.1f} s"
    if not agrees(promcast_s, prometheus_s):
        sys.stderr.write(f"{line}; they differ by more than {TOLERANCE:.0%}\n")
        return 1
    sys.stdout.write(f"{line}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
