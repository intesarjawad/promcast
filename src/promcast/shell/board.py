"""The `promcast scan` runway board: one shared time axis, one lane per series.

A bar is Promcast's time left, a red cross is where the threshold is reached and a grey
diamond is where predict_linear says it happens. The gap between the cross and the diamond
is the argument, so every row is drawn on the same axis.
"""

import html
import io
import os
import re
import textwrap
import unicodedata
from datetime import UTC, datetime
from math import ceil
from typing import Literal, NamedTuple, TextIO

from rich.cells import cell_len, split_graphemes
from rich.console import Console
from rich.terminal_theme import TerminalTheme
from rich.text import Text

from promcast.shell.scan import ScanResult, ScanRow, ScanSkipped

GROUND = "#111112"
INK = "#EDEDE6"
HIVIS = "#D7FF2F"
ALERT = "#FF4A3D"
GHOST = "#9A9DA3"
MUTED = "#8C9097"
RULE = "#34373C"

Color = Literal["standard", "truecolor"] | None
WIDE_FROM = 72
NARROW_LANE = 36
MAX_ROWS = 20
_STEPS = (60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400)
_THEME = TerminalTheme(
    (0x11, 0x11, 0x12),
    (0xED, 0xED, 0xE6),
    [
        (0x11, 0x11, 0x12),
        (0xFF, 0x4A, 0x3D),
        (0xD7, 0xFF, 0x2F),
        (0xFF, 0xD2, 0x3F),
        (0x6C, 0xA0, 0xFF),
        (0xC3, 0x8B, 0xFF),
        (0x5F, 0xD7, 0xD7),
        (0xED, 0xED, 0xE6),
    ],
)


def cell_at(seconds: float, span: int, size: int) -> int:
    """The one time scale: the lane cell a moment `seconds` from now falls in."""
    return round(seconds / span * size)


def clock(seconds: float) -> str:
    """Format seconds as `1h 15m`, or `5m` under an hour."""
    minutes = max(0, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


def _tick(seconds: int) -> str:
    hours, minutes = divmod(seconds // 60, 60)
    return f"{hours}h" if hours and not minutes else clock(seconds)


def _identity(item: ScanRow | ScanSkipped) -> str:
    keys = (item.show_labels if isinstance(item, ScanRow) and item.show_labels else None) or list(
        item.labels
    )
    return terminal_text(" ".join(item.labels[key] for key in keys if key in item.labels))


def terminal_text(value: str) -> str:
    """Expose control and formatting characters as literal escapes at terminal boundaries."""
    return "".join(
        char.encode("unicode_escape").decode("ascii")
        if unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"}
        else char
        for char in value
    )


def _fit(text: str, size: int) -> str:
    fitted = Text(terminal_text(text))
    fitted.truncate(size, overflow="ellipsis")
    return fitted.plain


def _threshold(row: ScanRow) -> str:
    return (
        f"{row.threshold * 100:g} % free"
        if row.unit == "ratio"
        else f"{row.threshold:g} {terminal_text(row.unit)}"
    )


def _cell(color: bool, fill: str, glyph: str | None = None) -> tuple[str, str]:
    """A filled lane cell: a background tint in colour, a full block without it."""
    if color:
        return (glyph or " ", f"bold {GROUND} on {fill}" if glyph else f"on {fill}")
    return (glyph or "█", "")


def marks(row: ScanRow, origin: int, span: int, size: int) -> tuple[int | None, int | None]:
    """Lane cells of the cross and the diamond; None when the series does not reach it."""
    reach = max(1, min(size, cell_at(row.horizon_s, span, size)))
    promcast = None if row.promcast_breach_ts is None else row.promcast_breach_ts - origin
    linear = None if row.linear_breach_ts is None else row.linear_breach_ts - origin
    cross = None if promcast is None else min(reach - 1, max(1, cell_at(promcast, span, size)))
    if promcast is not None and promcast <= 0:
        return 0, None
    index = None if linear is None else min(reach - 1, max(0, cell_at(linear, span, size)))
    return cross, None if index == cross and cross is not None else index


def _lane(row: ScanRow, origin: int, span: int, size: int, color: bool) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = [(" ", "")] * size
    promcast = None if row.promcast_breach_ts is None else row.promcast_breach_ts - origin
    if promcast is not None and promcast <= 0:
        label = f" past {_threshold(row)} · page now "
        label = ("█" + label + "█" if not color else label)[:size]
        fill = _cell(color, ALERT)
        return [
            (label[i], f"bold {GROUND} on {ALERT}" if color else "") if i < len(label) else fill
            for i in range(size)
        ]
    reach = max(1, min(size, cell_at(row.horizon_s, span, size)))
    cross, diamond = marks(row, origin, span, size)
    cells[:reach] = [("─", RULE)] * reach
    if cross is not None:
        cells[:cross] = [_cell(color, HIVIS)] * cross
        cells[cross] = ("✕", f"bold {ALERT}")
    if diamond is not None:
        bar = cross is not None and diamond < cross
        cells[diamond] = _cell(color, HIVIS, "◆") if bar else ("◆", f"bold {GHOST}")
    return cells


def _warming(item: ScanSkipped, block: int, size: int) -> list[tuple[str, str]]:
    if item.reason != "too_short":
        text = f"not forecast: {terminal_text(item.reason)}"
    elif item.warmup_minutes is None:
        text = "warming up"
    else:
        text = f"warming up, {clock(item.warmup_minutes * 60)} to first forecast"
    return [("░", RULE)] * block + [(char, MUTED) for char in f"  {text}"[:size]]


def _axis(span: int, size: int, pad: str) -> tuple[Text, Text]:
    """Tick marks over the lane with each label centred on its tick."""
    step = next(s for s in _STEPS if span / s <= 6)
    base = len(pad)
    labels, ticks = [" "] * (base + size + 8), [" "] * base + ["┈"] * (size + 1)
    end = -1
    for at in range(0, span + 1, step):
        position = base + cell_at(at, span, size)
        text = "now" if at == 0 else _tick(at)
        ticks[position] = "┼"
        start = max(0, position - len(text) // 2)
        if start > end:
            labels[start : start + len(text)] = text
            end = start + len(text)
    return Text("".join(labels).rstrip(), style=MUTED), Text("".join(ticks), style=RULE)


def _lane_text(cells: list[tuple[str, str]]) -> Text:
    text = Text()
    for char, style in cells:
        text.append(char, style=style)
    return text


def _headline(result: ScanResult, span: int, margin: str, width: int, wide: bool) -> list[Text]:
    document = result.document
    first = document.rows[0] if document.rows else None
    if first is None:
        return [Text(margin + "Warming up", style=f"bold {INK}")]
    name = Text(_fit(_identity(first), width - 30 if wide else width), style=f"bold {INK}")
    left = ""
    if first.promcast_breach_ts is None:
        return [
            Text.assemble(
                (margin + "Nothing fills in the next ", MUTED), (clock(span), f"bold {HIVIS}")
            )
        ]
    if first.promcast_breach_ts <= document.origin_ts:
        free = f"  {first.value * 100:.1f} % free" if first.unit == "ratio" else ""
        label = Text(margin + "Past threshold", style=f"bold {ALERT}")
        detail = Text.assemble(name, (free, MUTED))
    else:
        left = clock(first.promcast_breach_ts - document.origin_ts)
        label = Text.assemble((margin + "First to fill", MUTED))
        detail = Text.assemble(name, ("  in ", MUTED), (left, f"bold {HIVIS}"))
    if wide:
        return [Text.assemble(label, "  ", detail)]
    if first.promcast_breach_ts > document.origin_ts:
        return [Text.assemble(label, " in ", (left, f"bold {HIVIS}")), name]
    return [label, detail]


def _legend(color: bool, margin: str, wide: bool) -> Text:
    bar = _cell(color, HIVIS)
    parts = (
        (bar[0] * 3, bar[1], " Promcast lead time   " if wide else " lead  "),
        ("✕", f"bold {ALERT}", " threshold reached   " if wide else " limit  "),
        ("◆", f"bold {GHOST}", " predict_linear" if wide else " linear"),
    )
    legend = Text(margin)
    for glyph, style, label in parts:
        legend.append(glyph, style=style)
        legend.append(label, style=MUTED)
    return legend


def _footer(result: ScanResult, console: Console, color: bool, margin: str, wide: bool) -> None:
    document = result.document
    notes_shown = False
    for row in result.document.rows[:MAX_ROWS]:
        notes: list[str] = []
        if row.short_history_min is not None:
            notes.append(f"short history: {row.short_history_min:g} min")
        if row.missing_fraction:
            notes.append(f"{row.missing_fraction:.1%} of samples missing")
        if notes:
            text = f"{margin}{_identity(row)}: " + ", ".join(notes)
            console.print(Text(text, style=MUTED), overflow="fold")
            notes_shown = True
    footnotes: list[str] = []
    if len(document.rows) > MAX_ROWS:
        footnotes.append(f"{len(document.rows) - MAX_ROWS} more rows; --json lists them all")
    if result.dropped:
        footnotes.append(
            f"{result.dropped} more series not forecast; "
            f"rerun with --max-series {min(result.total, 500)}"
        )
    for note in footnotes:
        console.print(Text(margin + note, style=MUTED), overflow="fold")
    if notes_shown or footnotes:
        console.print()
    console.print(_legend(color, margin, wide), overflow="fold")
    console.print()
    eta = (
        ceil(result.replay_forecasts_needed / result.forecasts_per_s / 60)
        if result.forecasts_per_s
        else None
    )
    tail = f" (ETA up to {eta} min for 30d)" if eta is not None else ""
    for command, extra in (
        (f"promcast init {document.url}", ""),
        ("promcast replay --auto", tail if wide else ""),
    ):
        console.print(
            Text.assemble(
                (margin + "next ", f"bold {HIVIS}"), (terminal_text(command), INK), (extra, MUTED)
            ),
            overflow="fold",
        )


def _dims(width: int) -> tuple[bool, int, int]:
    """Wide or lane-only, the name column width and the lane width in cells."""
    if width >= WIDE_FROM:
        name_width = 26 if width < 120 else 28
        return True, name_width, max(36, min((width - name_width - 11) // 6 * 6, 96))
    return False, 0, max(30, (width - 8) // 6 * 6)


class Geometry(NamedTuple):
    """Where each lane sits on the character grid, for overlays drawn over the SVG."""

    span: int
    size: int
    lane_col: int
    first_lane_line: int
    line_stride: int


def geometry(result: ScanResult, width: int) -> Geometry:
    """The grid position of the first lane, its width and the line stride between lanes."""
    rows = result.document.rows[:MAX_ROWS]
    span = max((row.horizon_s for row in rows), default=21600)
    wide, name_width, size = _dims(width)
    head = len(_headline(result, span, "", width, wide))
    first = 2 + head + 1 + 2
    return Geometry(
        span, size, 2 + name_width + 1 if wide else 1, first + (0 if wide else 1), 1 if wide else 2
    )


def _draw(result: ScanResult, console: Console, width: int, color: bool) -> None:
    document = result.document
    rows = document.rows[:MAX_ROWS]
    span = max((row.horizon_s for row in rows), default=21600)
    wide, name_width, size = _dims(width)
    margin = "  " if wide else ""
    pad = "  " + " " * name_width + " " if wide else " "
    host = terminal_text(document.url.split("://")[-1])
    stamp = f"{host} · {terminal_text(document.model)} · " if wide else f"{host} · "
    stamp += f"{datetime.fromtimestamp(document.origin_ts, UTC):%H:%M} UTC"
    header = Text.assemble(
        (margin or " ", ""), (" PROMCAST ", f"bold {GROUND} on {HIVIS}"), ("  " + stamp, MUTED)
    )
    console.print(header, overflow="crop", no_wrap=True)
    console.print()
    for line in _headline(result, span, margin, width, wide):
        console.print(line, overflow="fold")
    console.print()
    for line in _axis(span, size, pad):
        console.print(line, overflow="crop", no_wrap=True)

    def label(name: str, style: str) -> Text:
        if wide:
            fitted = _fit(name, name_width)
            return Text(margin + fitted + " " * (name_width - cell_len(fitted) + 1), style=style)
        console.print(Text(" " + _fit(name, width - 1), style=style), overflow="crop", no_wrap=True)
        return Text(" ")

    for row in rows:
        promcast = row.promcast_breach_ts
        late = promcast is not None and promcast <= document.origin_ts
        left = (
            "clear" if promcast is None else "now" if late else clock(promcast - document.origin_ts)
        )
        line = label(_identity(row), MUTED if promcast is None else INK)
        line.append_text(_lane_text(_lane(row, document.origin_ts, span, size, color)))
        line.append(
            f" {left:>{7 if wide else 6}}",
            style=MUTED if promcast is None else f"bold {ALERT if late else HIVIS}",
        )
        console.print(line, overflow="crop", no_wrap=True)
    for item in document.skipped:
        line = label(_identity(item), MUTED)
        line.append_text(_lane_text(_warming(item, 9 if wide else 3, size + 8)))
        console.print(line, overflow="crop", no_wrap=True)
    console.print()
    _footer(result, console, color, margin, wide)


def detect_color(stream: TextIO) -> Color:
    """Pick truecolor, ANSI-16 or none from the stream and the NO_COLOR convention."""
    if not stream.isatty() or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return None
    return "truecolor" if os.environ.get("COLORTERM") in {"truecolor", "24bit"} else "standard"


def _console(buffer: TextIO, width: int, color: Color) -> Console:
    return Console(
        file=buffer,
        width=width,
        color_system=color,
        force_terminal=color is not None,
        legacy_windows=False,
        highlight=False,
    )


def scan_empty(url: str, names: list[str], *, width: int = 80, color: Color = None) -> str:
    """No recipe metric matched: a placeholder lane, what was tried and the next steps."""
    buffer = io.StringIO()
    console = _console(buffer, width, color)
    wide = width >= WIDE_FROM
    margin = "  " if wide else ""
    host = terminal_text(url.rsplit("://", maxsplit=1)[-1])
    console.print(
        Text.assemble(
            (" PROMCAST ", f"bold {GROUND} on {HIVIS}"), (f"  {host} · no series", MUTED)
        ),
        overflow="crop",
        no_wrap=True,
    )
    console.print()
    console.print(Text(margin + "Nothing to rank yet", style=f"bold {INK}"))
    for line in textwrap.wrap(
        "No recipe metrics matched: " + ", ".join(terminal_text(name) for name in names),
        width - len(margin),
        break_long_words=True,
    ):
        console.print(Text(margin + line, style=MUTED), overflow="crop", no_wrap=True)
    console.print()
    size = max(30, (width - 4) // 6 * 6) if wide else max(30, (width - 2) // 6 * 6)
    for line in _axis(21600, size, margin):
        console.print(line, overflow="crop", no_wrap=True)
    lane = [("░", RULE)] * 9 + [(c, MUTED) for c in "  no series to forecast"[: size - 9]]
    console.print(Text(margin) + _lane_text(lane), overflow="crop", no_wrap=True)
    console.print()
    for command in (
        "docker compose -f oci://ghcr.io/intesarjawad/promcast-demo:0.1.0 up",
        "promcast scan --query '<PromQL>' --below <threshold>",
    ):
        first, *rest = textwrap.wrap(
            "next " + command,
            width - len(margin),
            break_long_words=True,
            break_on_hyphens=False,
        )
        console.print(
            Text.assemble((margin + "next", f"bold {HIVIS}"), (first[4:], INK)),
            overflow="crop",
            no_wrap=True,
        )
        for line in rest:
            console.print(Text(margin + line, style=INK), overflow="crop", no_wrap=True)
    return buffer.getvalue()


def scan_progress(
    stage: str, done: int = 0, total: int = 0, *, width: int = 80, color: Color = None
) -> str:
    """One status line for stderr: a lane filling as series are forecast."""
    text = Text.assemble((" PROMCAST ", f"bold {GROUND} on {HIVIS}"), "  ")
    if stage == "discovery":
        text.append("finding Prometheus and recipe metrics", style=MUTED)
    else:
        text.append(f"forecasting {done} of {total}", style=INK)
        cells = max(0, min(20, width - 36))
        filled = round(cells * done / total) if total else 0
        lane = [_cell(color is not None, HIVIS)] * filled + [("░", RULE)] * (cells - filled)
        text.append("  ")
        text.append_text(_lane_text(lane))
    buffer = io.StringIO()
    _console(buffer, width, color).print(text, overflow="crop", no_wrap=True, end="")
    return buffer.getvalue()


def error_report(what: str, next_step: str, *, color: Color = None) -> str:
    """The two error lines, never wrapped: what happened and the exact next step."""
    buffer = io.StringIO()
    console = _console(buffer, 10_000, color)
    console.print(
        Text.assemble(("error:", f"bold {ALERT}"), f" {terminal_text(what)}"),
        overflow="ignore",
        no_wrap=True,
    )
    console.print(
        Text.assemble(("next: ", f"bold {HIVIS}"), f" {terminal_text(next_step)}"),
        overflow="ignore",
        no_wrap=True,
    )
    return buffer.getvalue()


def scan_board(result: ScanResult, *, width: int = 80, color: Color = None) -> str:
    """Render the board; `color` is None (plain), "standard" (ANSI-16) or "truecolor"."""
    buffer = io.StringIO()
    console = _console(buffer, width, color)
    _draw(result, console, width, color is not None)
    return buffer.getvalue()


_TITLE_BAND = 24


def scan_svg(result: ScanResult, *, width: int = 80, title: str = "promcast scan") -> str:
    """Export the truecolor board as an SVG that asks for no network font."""
    console = Console(
        record=True,
        file=io.StringIO(),
        width=width,
        color_system="truecolor",
        legacy_windows=False,
        highlight=False,
    )
    _draw(result, console, width, True)
    svg = console.export_svg(title=title, theme=_THEME, font_aspect_ratio=0.61)
    svg = re.sub(r"@font-face\s*\{.*?\}", "", svg, flags=re.S)
    svg = re.sub(r'<g transform="translate\(26,22\)">.*?</g>', "", svg, flags=re.S)
    svg = re.sub(r'<text class="[^"]*-title"[^>]*>.*?</text>', "", svg, flags=re.S)
    # The title bar is gone, so crop the band it occupied; the user space stays as Rich drew it.
    svg = re.sub(
        r'viewBox="0 0 ([\d.]+) ([\d.]+)"',
        lambda m: f'viewBox="0 {_TITLE_BAND} {m[1]} {float(m[2]) - _TITLE_BAND:.1f}"',
        svg,
        count=1,
    )
    svg = RUN.sub(_per_cell, svg)
    return svg.replace("Fira Code, monospace", '"Martian Mono", ui-monospace, monospace')


RUN = re.compile(
    r'<text class="([^"]*)" x="([\d.]+)" y="([\d.]+)" textLength="([\d.]+)" '
    r'clip-path="([^"]*)">([^<]*)</text>'
)


def _per_cell(match: "re.Match[str]") -> str:
    """Place every glyph of a ruler or lane run on its own cell.

    A run of box-drawing glyphs falls back to another font with other advance widths, which
    would shift the ticks against the bars; one text element per cell cannot drift.
    """
    name, x, y, length, clip, body = match.groups()
    chars = html.unescape(body)
    cell = float(length) / len(chars)
    spans, cells = split_graphemes(chars)
    if not any(char in chars for char in "┼┈─░"):
        if cells == len(chars):
            return match.group()
        # Rich's SVG widths count code points, unlike its terminal cell offsets.
        return match.group().replace(f'textLength="{length}"', f'textLength="{cell * cells:.1f}"')
    position = float(x)
    runs: list[str] = []
    for start, end, width in spans:
        glyph = chars[start:end]
        if glyph not in " \u00a0":
            runs.append(
                f'<text class="{name}" x="{position:.1f}" y="{y}" textLength="{cell * width:.1f}" '
                f'clip-path="{clip}">{html.escape(glyph)}</text>'
            )
        position += cell * width
    return "".join(runs)
