"""Capture the Compose demo's first Grafana screen once both breach times exist.

Waits until Prometheus holds a Promcast and a predict_linear breach for the demo disk (or
`--wait` runs out with only Promcast's), then
screenshots the dashboard and records the two times next to the image. Run against a stack
started from deploy/demo/compose.yaml.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PNG = ROOT / "docs/design/grafana-first-screen.png"
FACTS = ROOT / "docs/design/grafana-first-screen.json"
BRAVE = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"


def query(prometheus: str, expr: str) -> float | None:
    """The first value of an instant query, or None when it returns nothing."""
    url = f"{prometheus}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
        result = json.load(response)["data"]["result"]
    return float(result[0]["value"][1]) if result else None


def seconds_left(prometheus: str, method: str) -> float | None:
    """Seconds until the demo disk's breach by `method`, or None while there is none."""
    return query(
        prometheus,
        f'promcast_breach_timestamp_seconds{{target="demo_disk_free",method="{method}"}} - time()',
    )


def scraped_lately(prometheus: str) -> bool:
    """Whether the disk's last sample is fresh.

    While a forecast runs on a busy machine, scrapes can stall, and Prometheus then holds the
    last sample flat for its five-minute lookback: a stub on the line that the disk never drew.
    """
    age = query(prometheus, "time() - timestamp(demo_disk_free_bytes)")
    return age is not None and age < 20


def main() -> None:
    """Wait for both breaches, then capture the frame and its facts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prometheus", default="http://127.0.0.1:9090")
    parser.add_argument("--grafana", default="http://127.0.0.1:3000")
    parser.add_argument("--wait", type=int, default=3600, help="seconds to wait for both breaches")
    arguments = parser.parse_args()
    deadline = time.monotonic() + arguments.wait
    while True:
        promcast = seconds_left(arguments.prometheus, "toto")
        linear = seconds_left(arguments.prometheus, "linear")
        if (
            promcast is not None
            and promcast > 300
            and (linear is not None or time.monotonic() > deadline)
            and scraped_lately(arguments.prometheus)
        ):
            break
        if time.monotonic() > deadline + 600:
            raise SystemExit("error: no Promcast breach\nnext:  check docker compose ps")
        time.sleep(15)
    dash = "d/promcast-demo/synthetic-disk-forecast?orgId=1&kiosk&theme=dark"
    # The frame holds the fall and ends just past Promcast's breach: after it the disk is full,
    # and a band that climbs back from 0 B reads as a forecast of recovery, not of the page.
    ahead = math.ceil(promcast / 60) + 5
    url = f"{arguments.grafana}/{dash}&from=now-45m&to=now%2B{ahead}m"
    executable = os.environ.get("PROMCAST_BROWSER") or BRAVE
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable)
        view = browser.new_page(viewport={"width": 1440, "height": 660})
        # Read again as the panel loads, so the caption quotes what the stat panel shows.
        promcast = seconds_left(arguments.prometheus, "toto") or promcast
        linear = seconds_left(arguments.prometheus, "linear") or linear
        view.goto(url)
        view.wait_for_timeout(6000)
        # Only the first row: the chart and the two breach times.
        stat = view.get_by_test_id("data-testid Panel header Breach in").bounding_box()
        if stat is None:
            raise SystemExit("error: no Breach in panel\nnext:  check the dashboard loaded")
        view.screenshot(
            path=str(PNG),
            clip={"x": 0, "y": 0, "width": 1440, "height": stat["y"] + stat["height"] + 8},
        )
        browser.close()
    FACTS.write_text(
        json.dumps(
            {
                "promcast_minutes": promcast / 60,
                "linear_minutes": None if linear is None else linear / 60,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(f"promcast {promcast / 60:.0f} min, predict_linear {linear}\n")


if __name__ == "__main__":
    main()
