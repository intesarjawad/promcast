"""Print each locked NAB series' gaps on the five-minute grid the replay loader builds."""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from promcast.shell.datasets import read_lock
from promcast.shell.sources import nab, timestamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = Path.home() / ".cache" / "promcast" / "evidence" / "NAB"
STEP_S = 300


@dataclass(frozen=True)
class Gaps:
    """Grid coverage of one series as the loader sees it, plus the raw rows it drops."""

    rows: int
    missing: int
    longest_steps: int
    duplicates: int
    off_grid: int


def gaps(path: Path) -> Gaps:
    """Count missing grid points, the longest run of them, and duplicate or off-grid rows."""
    with path.open(newline="", encoding="utf-8") as stream:
        times = [timestamp(row["timestamp"]) for row in csv.DictReader(stream)]
    end_ts = max(times)
    missing = np.isnan(nab(path).values)
    longest = run = 0
    for gap in missing:
        run = run + 1 if gap else 0
        longest = max(longest, run)
    return Gaps(
        rows=len(times),
        missing=int(missing.sum()),
        longest_steps=longest,
        duplicates=len(times) - len(set(times)),
        off_grid=sum(1 for time in times if (end_ts - time) % STEP_S),
    )


def describe(name: str, found: Gaps) -> str:
    """Format one series as a single report line."""
    hours = found.longest_steps * STEP_S / 3600
    return (
        f"{name}: {found.rows} rows, {found.missing} missing, longest gap "
        f"{found.longest_steps} steps ({hours:.1f} h), {found.duplicates} duplicate, "
        f"{found.off_grid} off-grid"
    )


def main(argv: list[str] | None = None) -> int:
    """Report every CSV the lock names, read from a fetched corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--lock-file", type=Path, default=ROOT / "scripts" / "nab.lock")
    arguments = parser.parse_args(argv)
    for file in read_lock(arguments.lock_file).files:
        if not file.path.endswith(".csv"):
            continue
        path = arguments.corpus / file.path
        if not path.is_file():
            sys.stderr.write(f"{path} is missing; run scripts/fetch_nab.py\n")
            return 1
        sys.stdout.write(describe(file.path.removeprefix("data/"), gaps(path)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
