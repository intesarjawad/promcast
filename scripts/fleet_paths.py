"""Write the paths behind the timetable: each series' last day, nine quantiles and predict_linear.

The rows come from the same fleet and model as docs/design/fleet-scan.json, and the run fails
if they differ from it, so the page's paths and its breach times can never disagree.

    uv run python scripts/fleet_paths.py docs/design/fleet-paths.json
    uv run python scripts/fleet_paths.py --check docs/design/fleet-paths.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import synthetic_fleet as sf

from promcast.core.config import ModelConfig
from promcast.core.forecast import PreparationPolicy, run_cycle
from promcast.core.forecaster import Toto2Forecaster
from promcast.core.recipes import scan_target
from promcast.core.series import Series
from promcast.shell.discover import load_recipes

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "docs/design/fleet-scan.json"
HISTORY = 288  # the last day at five minutes
DIGITS = 5


def _round(values: "np.ndarray") -> list[float]:
    return [round(float(v), DIGITS) for v in values]


def paths(model_dir: Path) -> dict[str, object]:
    """Forecast the fleet once more through run_cycle and keep every quantile path."""
    committed = json.loads(SCAN.read_text(encoding="utf-8"))
    if sf.scan(model_dir).model_dump(mode="json")["rows"] != committed["rows"]:
        raise SystemExit(
            f"error: the fleet no longer scans to {SCAN.name}\n"
            "next:  run scripts/synthetic_fleet.py first"
        )
    forecaster = Toto2Forecaster(model_dir, sf.MODEL, batch_size=16)
    recipes = {recipe.name: recipe for recipe in load_recipes()}
    out: list[dict[str, object]] = []
    for name, members in sf.fleet().items():
        recipe = recipes[name]
        series = [
            Series(tuple(sorted(labels.items())), sf.STEP, float(sf.END), np.clip(v, 0, 1))
            for labels, v in members
        ]
        forecasts, _ = run_cycle(
            scan_target(recipe, sf.STEP, 50),
            series,
            forecaster,
            sf.END,
            policy=PreparationPolicy(32, True),
        )
        for item in series:
            learned = next(f for f in forecasts if f.labels == item.labels and f.model != "linear")
            linear = next(
                (f for f in forecasts if f.labels == item.labels and f.model == "linear"), None
            )
            out.append(
                {
                    "target": name,
                    "labels": dict(item.labels),
                    "threshold": recipe.threshold,
                    "history": _round(item.values[-HISTORY:]),
                    "quantiles": [_round(q) for q in learned.quantiles],
                    "linear": None if linear is None else _round(linear.quantiles[0]),
                }
            )
    return {"origin_ts": sf.END, "step_s": sf.STEP, "model": sf.MODEL, "series": out}


def _close(a: object, b: object, atol: float) -> bool:
    """Same structure and every number within atol; strings, keys and nulls must match exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        left, right = cast("dict[str, object]", a), cast("dict[str, object]", b)
        return left.keys() == right.keys() and all(_close(left[k], right[k], atol) for k in left)
    if isinstance(a, list) and isinstance(b, list):
        xs, ys = cast("list[object]", a), cast("list[object]", b)
        return len(xs) == len(ys) and all(_close(x, y, atol) for x, y in zip(xs, ys, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= atol
    return a == b


def main() -> int:
    """Write the document, or with --check fail when the file differs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--model-dir", type=Path, default=ModelConfig().cache_dir.expanduser() / sf.MODEL
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="with --check, accept numbers that differ by at most this much",
    )
    arguments = parser.parse_args()
    document = paths(arguments.model_dir)
    text = json.dumps(document, separators=(",", ":")) + "\n"
    if not arguments.check:
        arguments.out.write_text(text, encoding="utf-8")
        return 0
    committed = arguments.out.read_text(encoding="utf-8")
    if arguments.atol:
        same = _close(json.loads(committed), document, arguments.atol)
    else:
        same = committed == text
    if not same:
        sys.stderr.write(
            f"error: {arguments.out} differs from the generator\n"
            f"next:  run uv run python scripts/fleet_paths.py {arguments.out}\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
