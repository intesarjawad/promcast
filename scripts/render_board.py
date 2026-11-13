"""Render the scan board from a `promcast scan --json` document: terminal text or SVG.

uv run python scripts/render_board.py docs/design/fleet-scan.json --width 80 --svg site/board.svg
"""

import argparse
import sys
from math import ceil
from pathlib import Path

from promcast.shell import board
from promcast.shell.scan import ScanDocument, ScanResult, redact

# The synthetic fleet scrapes every 5 minutes and its newest mount has 18 samples.
STEP_S, FRESH_SAMPLES = 300, 18
SHOW = {"disk": ["instance", "mountpoint"], "memory": ["instance"]}


def load(path: Path, *, hide: bool = False) -> ScanResult:
    """Read a scan document; label order is not in the JSON, so rebuild it from the target."""
    document = ScanDocument.model_validate_json(path.read_text(encoding="utf-8"))
    document = document.model_copy(
        update={
            "rows": [
                row.model_copy(update={"show_labels": SHOW.get(row.target, list(row.labels))})
                for row in document.rows
            ]
        }
    )
    if hide:
        document = redact(document)
    skipped = [
        item.model_copy(update={"warmup_minutes": ceil((32 - FRESH_SAMPLES) * STEP_S / 60)})
        for item in document.skipped
    ]
    return ScanResult(document.model_copy(update={"skipped": skipped}), [], 0, 0, 0)


def main() -> None:
    """Write the board to stdout, or an SVG file with --svg."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--redact", action="store_true")
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--color", choices=("truecolor", "standard"))
    arguments = parser.parse_args()
    result = load(arguments.scan, hide=arguments.redact)
    if arguments.svg:
        arguments.svg.write_text(board.scan_svg(result, width=arguments.width), encoding="utf-8")
    else:
        sys.stdout.write(board.scan_board(result, width=arguments.width, color=arguments.color))


if __name__ == "__main__":
    main()
