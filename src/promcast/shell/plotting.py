from collections.abc import Sequence
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from statistics import median
from typing import Protocol, TypedDict, Unpack, runtime_checkable

import numpy as np

from promcast.core.replay import ReplayResult, detect_incidents
from promcast.core.report import report_markdown
from promcast.core.series import FloatArray
from promcast.shell import board
from promcast.shell.scan import ScanResult

_STYLES = (("-", "o"), ("--", "s"), (":", "^"), ("-.", "v"), ("-", "D"), ("--", "P"), (":", "X"))


class _LineStyle(TypedDict, total=False):
    linestyle: str
    marker: str
    label: str
    color: str


@runtime_checkable
class _ReplayAxes(Protocol):
    """Typed subset of matplotlib's keyword-based axes API used by replay."""

    def plot(
        self,
        x: Sequence[float] | FloatArray,
        y: Sequence[float] | FloatArray,
        **style: Unpack[_LineStyle],
    ) -> object: ...
    def set_xlabel(self, xlabel: str) -> object: ...
    def set_ylabel(self, ylabel: str) -> object: ...
    def legend(self) -> object: ...
    def axhline(self, y: float, *, linestyle: str, label: str) -> object: ...
    def axvline(self, x: float, *, linestyle: str, label: str) -> object: ...
    def axvspan(self, xmin: float, xmax: float, *, alpha: float, hatch: str) -> object: ...


class _Figure(Protocol):
    def add_subplot(self, rows: int, columns: int, index: int) -> _ReplayAxes: ...
    def clear(self) -> None: ...


class _Canvas(Protocol):
    def print_png(self, filename: str) -> None: ...


@runtime_checkable
class _FigureFactory(Protocol):
    def __call__(self, *, layout: str) -> _Figure: ...


@runtime_checkable
class _CanvasFactory(Protocol):
    def __call__(self, figure: _Figure) -> _Canvas: ...


def plot_replay(result: ReplayResult, values: FloatArray, path: Path) -> str:
    """Write a headless test-sweep plot and return its Markdown image alternative."""
    figure_factory: object = import_module("matplotlib.figure").Figure
    canvas_factory: object = import_module("matplotlib.backends.backend_agg").FigureCanvasAgg
    if not isinstance(figure_factory, _FigureFactory) or not isinstance(
        canvas_factory, _CanvasFactory
    ):
        raise TypeError("matplotlib must provide figure and headless canvas factories")
    figure = figure_factory(layout="constrained")
    canvas = canvas_factory(figure)
    sweep = figure.add_subplot(2, 1, 1)
    actual = figure.add_subplot(2, 1, 2)
    for index, (name, method) in enumerate(sorted(result.methods.items())):
        style, marker = _STYLES[index % len(_STYLES)]
        scores = sorted(
            method.test_sweep.items(), key=lambda item: (item[1].false_alerts_per_week, item[0])
        )
        x = [score.false_alerts_per_week for _, score in scores]
        y = [median(score.leads_s) / 60 if score.leads_s else 0 for _, score in scores]
        sweep.plot(x, y, linestyle=style, marker=marker, label=name)
        chosen = method.test_sweep[method.point]
        sweep.plot(
            [chosen.false_alerts_per_week],
            [median(chosen.leads_s) / 60 if chosen.leads_s else 0],
            marker="*",
            linestyle="none",
            color="black",
            label=f"{name} selected",
        )
    sweep.set_xlabel("Test false alerts per series-week")
    sweep.set_ylabel("Median early lead (min)")
    sweep.legend()
    times = np.arange(values.size, dtype=np.float64) * result.spec.step_s
    actual.plot(times / 3600, values, label="Actual")
    actual.axhline(result.threshold, linestyle="--", label="Threshold")
    actual.axvline(result.split * result.spec.step_s / 3600, linestyle=":", label="Test split")
    fixed = replace(result.spec, threshold=result.threshold)
    for start, stop in ((0, result.split), (result.split, values.size)):
        for incident in detect_incidents(
            times[start:stop], values[start:stop], result.threshold, fixed
        ):
            actual.axvspan(incident.start_ts / 3600, incident.end_ts / 3600, alpha=0.2, hatch="//")
    actual.set_xlabel("Hours from series start")
    actual.set_ylabel("Observed value")
    actual.legend()
    canvas.print_png(str(path))
    figure.clear()
    names = ", ".join(sorted(result.methods))
    return (
        f"Test-split median early lead versus false alerts per series-week for {names}. "
        "Every swept point is plotted; stars mark calibration-selected points. "
        "Zero lead denotes no early catch. Lower panel shows actual values, threshold, "
        "calibration/test boundary and shaded threshold incidents."
    )


def terminal_table(
    result: ReplayResult, caveats: tuple[str, ...] = (), *, foundation_models: bool = True
) -> str:
    """Print plain replay measurements without terminal styling."""
    lines = [*caveats]
    if not foundation_models:
        lines.append("Foundation models not requested; comparing baselines only.")
    lines.extend(
        [
            f"Calibration budget: {result.spec.budget_per_week:.4f} false alerts/series-week",
            "Method | lead(s) | recall | false/week | incidents | series | weeks",
        ]
    )
    for name, method in sorted(result.methods.items()):
        score = method.test
        lead = f"{median(score.leads_s):.2f}" if score.leads_s else "unavailable"
        lines.append(
            f"{name} | {lead} | {score.recall:.4f} | {score.false_alerts_per_week:.4f} | "
            f"{score.n_incidents} | {score.n_series} | {score.weeks:.6f}"
        )
        if score.false_alerts_per_week > 1.5 * result.spec.budget_per_week:
            lines.append(f"{name}: test false-alert rate above 1.5 x calibration budget")
    return "\n".join(lines) + "\n"


def markdown_report(
    result: ReplayResult,
    caveats: tuple[str, ...],
    secondary: ReplayResult | None,
    foundation_models: bool,
) -> str:
    """Compose existing score tables with replay caveats and the plot alternative."""
    text = report_markdown(result)
    winner = result.calibration_winner
    if winner != "linear" and not winner.startswith("toto-"):
        text = f"Replay-only calibration winner: {winner}\n\n" + text
    text += "\n" + "\n".join(caveats) + "\n"
    if not foundation_models:
        text += "Foundation models were not requested; this replay compares baselines only.\n"
    if secondary is not None:
        text += "\n## Secondary labelled-anomaly scores\n\n" + report_markdown(secondary)
    return text


def scan_board(result: ScanResult, *, width: int = 80, color: board.Color = None) -> str:
    """Render the countdown board within the terminal width."""
    return board.scan_board(result, width=width, color=color)


def adoption_dashboard() -> dict[str, object]:
    """Use default Grafana presentation for the provisioned capacity overview."""
    return {
        "uid": "promcast",
        "title": "Capacity forecasts",
        "schemaVersion": 39,
        "panels": [
            {
                "id": 1,
                "type": "timeseries",
                "title": "Minutes until threshold",
                "datasource": {"type": "prometheus", "uid": "promcast-tsdb"},
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
                "targets": [
                    {"refId": "A", "expr": "(promcast_breach_timestamp_seconds - time()) / 60"}
                ],
            }
        ],
    }
