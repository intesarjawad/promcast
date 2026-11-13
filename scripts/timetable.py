"""Draw the fleet as a graphical timetable, after Marey's Paris to Lyon chart of 1885.

Time runs across, free space runs up the sheet on a log scale, and each series is a train.
Promcast's nine quantile paths leave "now" as a bundle; predict_linear is one straight dashed
line. A train stops where it meets the threshold rule, at the first sample the scan itself
counts as a breach, so a thread's end and the scan's breach time are the same instant.

Arrivals are stamped above the rule like stations: Promcast's in ink, predict_linear's in blue,
the earlier one to the left of its arrival and the later one to the right. Between the two, a
solid red bar lies across the rule as long as the gap, with what the gap means set over it, and
from Promcast's arrival on, the rule runs thick: the disk is past the line.
"""

import html
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from promcast.shell import board
from promcast.shell.scan import ScanDocument, ScanRow

LEVELS = (0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07)
LOW, HIGH = 0.022, 0.68  # the top and bottom of the sheet; the top leaves room for stamps
PAGE_Q = 0  # the disk and memory recipes page on q10 of free space
DIGITS = 4
AHEAD_S = 13500  # forecast shown: through 17:45, the fleet's last arrival is 17:35
CLEAR = 12  # paper kept around every stamp


@dataclass(frozen=True)
class Sheet:
    """One layout of the timetable, in SVG user units."""

    width: int
    height: int
    left: float
    right: float
    top: float
    bottom: float
    before_s: int  # history shown before now
    hour_every: int
    font: float
    wide: bool  # a narrow sheet breaks the bar's label over two lines
    after_s: int = 6 * 3600  # forecast shown after now
    stamp: float = 0  # station stamp type size; the sheet's own when 0

    def x(self, seconds: float) -> float:
        """Horizontal position of a moment, in seconds from now."""
        span = self.before_s + self.after_s
        return self.left + (seconds + self.before_s) / span * (self.right - self.left)

    def y(self, free: float) -> float:
        """Vertical position of a free-space level; equal room per halving of free space."""
        free = min(max(free, LOW), HIGH)
        share = (math.log(free) - math.log(LOW)) / (math.log(HIGH) - math.log(LOW))
        return self.top + share * (self.bottom - self.top)


# One hour of history: the forecast, and the gap between the stamps, get the width.
WIDE = Sheet(1368, 520, 58, 1350, 40, 476, 3600, 1, 13, wide=True, after_s=AHEAD_S, stamp=17)
NARROW = Sheet(380, 470, 40, 372, 46, 430, 3600, 1, 11, wide=False, after_s=AHEAD_S)
# The card is read at thumbnail size, so it shows only the hours around the two stamps.
SOCIAL = Sheet(830, 430, 48, 812, 66, 382, 3600, 1, 15, wide=False, after_s=3 * 3600)


def hm(ts: float) -> str:
    """A UTC clock time, `15:15`."""
    return datetime.fromtimestamp(ts, UTC).strftime("%H:%M")


def name(labels: dict[str, str]) -> str:
    """The train's name on the sheet: its instance."""
    return labels.get("instance") or " ".join(labels.values())


def place(labels: dict[str, str], target: str) -> str:
    """What the series measures: its mount, or the target for memory."""
    return labels.get("mountpoint") or target


def arrive(values: list[float], threshold: float) -> tuple[list[float], int | None]:
    """Cut a forecast path at its first step at or under the threshold, ending on the rule.

    Returns the kept values and the index of the arrival step, or None when it never arrives.
    """
    for index, value in enumerate(values):
        if value <= threshold:
            return [*values[:index], threshold], index
    return values, None


def level_at(start: float, path: list[float], step_s: int, seconds: float) -> float:
    """A path's value `seconds` after now, linear between steps; `start` is the value at now."""
    if seconds <= 0:
        return start
    position = seconds / step_s
    index = min(int(position), len(path) - 1)
    before = start if index == 0 else path[index - 1]
    after = path[index] if position < len(path) else path[-1]
    return before + (after - before) * (min(position, len(path)) - index)


def _d(points: list[tuple[float, float]]) -> str:
    return "".join(f"{'L' if i else 'M'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))


def _text(x: float, y: float, body: str, cls: str, **attrs: str | float) -> str:
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}"{extra}>{html.escape(body)}</text>'


@dataclass(frozen=True)
class Train:
    """One series: its scan row and its paths."""

    row: ScanRow
    history: list[float]
    quantiles: list[list[float]]
    linear: list[float] | None


def trains(doc: ScanDocument, paths: dict[str, Any]) -> list[Train]:
    """Pair each scan row with its paths, in the scan's order."""
    by_labels = {json.dumps(p["labels"], sort_keys=True): p for p in paths["series"]}
    out: list[Train] = []
    for row in doc.rows:
        found = by_labels[json.dumps(row.labels, sort_keys=True)]
        out.append(Train(row, found["history"], found["quantiles"], found["linear"]))
    return out


def minutes(seconds: float) -> str:
    """A gap in words: `45 min`, or `1h 05m` past the hour."""
    whole = round(seconds / 60)
    return f"{whole} min" if whole < 60 else board.clock(seconds)


def verdict(row: ScanRow) -> str | None:
    """What the stretch of rule between the two arrivals means, in the scan's own numbers."""
    a, b = row.promcast_breach_ts, row.linear_breach_ts
    if a is not None and b is not None and a != b:
        if b > a:
            return f"{minutes(b - a)} unpaged"
        return f"{minutes(a - b)} where predict_linear paged first"
    if b is not None and a is None:
        return "a page Promcast never sends"
    return None


def reading(train: Train, step_s: int, origin: float, seconds: float) -> dict[str, str]:
    """The reading over the sheet with the hand `seconds` after now; site.js says the same.

    A path that has reached the rule reads as the time it paged, as its thread ends there on
    the sheet; once Promcast has paged, the last item is the gap the bar draws.
    """
    row = train.row
    a, b, t = row.promcast_breach_ts, row.linear_breach_ts, origin + seconds

    def at(path: list[float], arrived: float | None) -> str:
        if arrived is not None and t >= arrived:
            return hm(arrived)
        return f"{level_at(train.history[-1], path, step_s, seconds) * 100:.1f} % free"

    if a is None:
        say, left = "Pages", "not in this forecast"
    elif t < a:
        say, left = "Pages in", board.clock(a - t)
    elif b is not None and b < a:
        say, left = "", f"predict_linear {minutes(a - b)} early"
    else:
        say, left = "", f"{minutes(min(t, b or t) - a)} unpaged"
    return {
        "p": at(train.quantiles[PAGE_Q], a),
        "d": "no forecast" if train.linear is None else at(train.linear, b),
        "say": say,
        "left": left,
    }


def page_data(doc: ScanDocument, paths: dict[str, Any]) -> str:
    """What the page script needs to move the hand and read the sheet, rounded for size."""
    origin = doc.origin_ts
    series = [
        {
            "name": name(t.row.labels),
            "place": place(t.row.labels, t.row.target),
            "threshold": t.row.threshold,
            "now": round(t.history[-1], DIGITS),
            "q": [[round(v, DIGITS) for v in q] for q in t.quantiles],
            "lin": None if t.linear is None else [round(v, DIGITS) for v in t.linear],
            "p": None if t.row.promcast_breach_ts is None else t.row.promcast_breach_ts - origin,
            "d": None if t.row.linear_breach_ts is None else t.row.linear_breach_ts - origin,
        }
        for t in trains(doc, paths)
    ]
    return json.dumps(
        {"origin": origin, "step": paths["step_s"], "low": LOW, "high": HIGH, "series": series},
        separators=(",", ":"),
    )


def spread(items: list[tuple[str, float]], gap_y: float) -> list[tuple[str, float]]:
    """Labels nudged down until none overlaps the one above it."""
    placed: list[tuple[str, float]] = []
    for label, at in sorted(items, key=lambda item: item[1]):
        placed.append((label, max(at, placed[-1][1] + gap_y) if placed else at))
    return placed


@dataclass(frozen=True)
class _Frame:
    """What every part of one drawing shares."""

    doc: ScanDocument
    sheet: Sheet
    step: int
    key: str
    page_y: float  # the threshold rule
    still_s: float  # where the hand rests in the still frame


def _ink(key: str, seed: int, shift: tuple[float, float], region: str) -> str:
    """One stamp's impression: its own edge, paper tooth, uneven pressure and misregistration."""
    dx, dy = shift
    return (
        f'<filter id="{key}" {region}>'
        f'<feTurbulence type="fractalNoise" baseFrequency="0.7" numOctaves="2" seed="{seed}"/>'
        '<feDisplacementMap in="SourceGraphic" scale="1.8" xChannelSelector="R" '
        'yChannelSelector="G" result="edge"/>'
        f'<feTurbulence type="fractalNoise" baseFrequency="1.9" seed="{seed + 1}"/>'
        '<feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -3 2.6" '
        'result="tooth"/>'
        f'<feTurbulence type="fractalNoise" baseFrequency="0.035" seed="{seed + 2}"/>'
        '<feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.4 0 0 0 0.15" '
        'result="press"/>'
        '<feComposite in="tooth" in2="press" operator="arithmetic" k1="1" result="mask"/>'
        '<feComposite in="edge" in2="mask" operator="in" result="print"/>'
        f'<feOffset in="print" dx="{dx}" dy="{dy}"/>'
        '<feComponentTransfer><feFuncA type="linear" slope="0.3"/></feComponentTransfer>'
        '<feMerge><feMergeNode/><feMergeNode in="print"/></feMerge></filter>'
    )


def _defs(f: _Frame) -> str:
    """Patterns, the sweep that clips everything ahead of the hand, and the ink filters.

    Each stamp prints through its own filter, so no two impressions share a speckle; `bleed`
    lets the thick rule spread a little into the paper around it.
    """
    s, k = f.sheet, f.key
    region = f'filterUnits="userSpaceOnUse" x="0" y="0" width="{s.width}" height="{s.height}"'
    return (
        "<defs>"
        f'<pattern id="{k}-hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><line class="hatch" x1="0" y1="0" x2="0" y2="6"/></pattern>'
        f'<clipPath id="{k}-plot"><rect x="{s.left}" y="0" width="{s.right - s.left}" '
        f'height="{s.height}"/></clipPath>'
        f'<clipPath id="{k}-sweep"><rect class="sweep" x="{s.x(0) - 1:.1f}" y="0" '
        f'width="{s.x(f.still_s) - s.x(0) + 1:.1f}" height="{s.height}"/></clipPath>'
        + _ink(f"{k}-ink-pc", 7, (0.9, -0.7), region)
        + _ink(f"{k}-ink-ln", 23, (-0.8, 0.9), region)
        + f'<filter id="{k}-bleed" {region}>'
        '<feTurbulence type="fractalNoise" baseFrequency="0.04 0.8" numOctaves="2" seed="4"/>'
        '<feDisplacementMap in="SourceGraphic" scale="2.6" xChannelSelector="R" '
        'yChannelSelector="G" result="edge"/>'
        '<feGaussianBlur in="edge" stdDeviation="1.1" result="spread"/>'
        '<feMerge><feMergeNode in="spread"/><feMergeNode in="edge"/></feMerge></filter>'
        "</defs>"
    )


def _grid(f: _Frame) -> str:
    """The ruling: ten-minute and hour columns, the level rules and the % axis."""
    s, x, y, fs = f.sheet, f.sheet.x, f.sheet.y, f.sheet.font
    out: list[str] = []
    for t in range(-s.before_s, s.after_s + 1, 600):
        hour = t % 3600 == 0
        out.append(
            f'<line class="{"hr" if hour else "fine"}" x1="{x(t):.1f}" x2="{x(t):.1f}" '
            f'y1="{s.top:.1f}" y2="{s.bottom:.1f}"/>'
        )
        if hour and (t // 3600) % s.hour_every == 0:
            edge = x(t) + fs * 1.6 > s.width  # the last hour would run off the sheet
            out.append(
                _text(
                    x(t) + (fs * 0.4 if edge else 0),
                    s.bottom + fs + 8,
                    hm(f.doc.origin_ts + t),
                    "tick",
                    text_anchor="end" if edge else "middle",
                )
            )
    out.extend(
        f'<line class="lvl" x1="{s.left:.1f}" x2="{s.right:.1f}" '
        f'y1="{y(level):.1f}" y2="{y(level):.1f}"/>'
        + _text(s.left - 6, y(level) + fs * 0.35, f"{level * 100:g} %", "lvl-l")
        for level in LEVELS
    )
    return (
        f'<g class="grid">{"".join(out)}</g>'
        f'<rect x="{s.left:.1f}" y="{s.top:.1f}" width="{s.right - s.left:.1f}" '
        f'height="{f.page_y - s.top:.1f}" fill="url(#{f.key}-hatch)"/>'
    )


def _trains(f: _Frame, paths: dict[str, Any]) -> str:
    """Each series as one group: measured history, then its forecast under the sweep clip.

    Under every group, the whole forecast is printed faintly to the sheet's edge, so the hours
    past the hand show what is coming. A group fades as a whole, so a background series'
    stacked quantiles never print darker than one line, and the lead series is drawn last, over
    a paper knock-out that keeps other threads from reading through its bundle. Each name sits
    where history ends, clear of its own line.
    """
    s, x, y, step, fs = f.sheet, f.sheet.x, f.sheet.y, f.step, f.sheet.font
    groups: list[str] = []
    ghosts: list[str] = []
    names: list[tuple[str, float]] = []
    for i, train in enumerate(trains(f.doc, paths)):
        hist = train.history[-(s.before_s // step) - 1 :]
        pts = [(x(-(len(hist) - 1 - k) * step), y(v)) for k, v in enumerate(hist)]
        start = (x(0), y(hist[-1]))
        lines = [("q50" if k == 4 else "q", q) for k, q in enumerate(train.quantiles)]
        if train.linear is not None:
            lines.append(("lin", train.linear))
        ahead: list[str] = []
        knock: list[str] = []
        for cls, path in lines:
            kept, _ = arrive(path, train.row.threshold)
            d = _d([start, *((x((j + 1) * step), y(v)) for j, v in enumerate(kept))])
            ahead.append(f'<path class="{cls}" d="{d}"/>')
            knock.append(f'<path d="{d}"/>')
            if cls != "lin":
                ghosts.append(f'<path class="{cls}" d="{d}"/>')
        groups.append(
            f'<g class="trn{" dim" if i else ""}" data-i="{i}">'
            f'<path class="hist" d="{_d(pts)}"/>'
            f'<g clip-path="url(#{f.key}-sweep)"><g class="knock">{"".join(knock)}</g>'
            f"{''.join(ahead)}</g></g>"
        )
        label = name(train.row.labels)
        reach = x(0) - 6 - len(label) * fs * 0.62
        names.append((label, min(py for px, py in pts if px >= reach) - 5))
    labels = "".join(
        _text(x(0) - 6, at, label, "train", text_anchor="end")
        for label, at in spread(names, fs + 2)
    )
    return (
        f'<g class="trains"><g class="ghosts" clip-path="url(#{f.key}-plot)">{"".join(ghosts)}</g>'
        f"{''.join(reversed(groups))}{labels}</g>"
    )


def _rule_and_hand(f: _Frame) -> str:
    """The page rule, the line at now, and the clock hand."""
    s, x, fs, origin = f.sheet, f.sheet.x, f.sheet.font, f.doc.origin_ts
    threshold = f"{f.doc.rows[0].threshold * 100:g} %"
    return (
        f'<line class="page" x1="{s.left - (44 if s.wide else 30):.1f}" x2="{s.right:.1f}" '
        f'y1="{f.page_y:.1f}" y2="{f.page_y:.1f}"/>'
        + _text(s.left - 6, f.page_y - 7, threshold, "page-l")
        + _text(s.left - 6, f.page_y + fs + 5, "page", "page-l")
        + f'<line class="now" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{s.top - 14:.1f}" '
        f'y2="{s.bottom:.1f}"/>'
        + _text(x(0) - 6, s.top - 18, f"now {hm(origin)}", "now-l")
        + f'<g class="hand" style="transform:translateX({x(f.still_s) - x(0):.1f}px)">'
        f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{s.top - 14:.1f}" y2="{s.bottom:.1f}"/>'
        f'<path class="hand-bg" d="M{x(0) - 5:.1f},{s.bottom + 4:.1f}L{x(0):.1f},'
        f'{s.bottom - 2:.1f}L{x(0) + 5:.1f},{s.bottom + 4:.1f}Z"/>'
        f'<rect class="hand-bg" x="{x(0) - fs * 2.9:.1f}" y="{s.bottom + 3:.1f}" '
        f'width="{fs * 5.8:.1f}" height="{fs + 10:.1f}" rx="3"/>'
        + _text(x(0), s.bottom + fs + 8, hm(origin + f.still_s), "hand-l", text_anchor="middle")
        + "</g>"
    )


def _shown(f: _Frame, at: float) -> str:
    """The still frame shows what the hand has already passed."""
    return " in" if at <= f.still_s else ""


def _station(f: _Frame, offset: float, who: str, cls: str, left: bool) -> str:
    """A station stamp above the rule, beside its arrival, with a short leader down to it.

    The earlier of two arrivals stamps to its left and the later to its right, so the stretch
    of rule between them stays clear for the bar and its remark.
    """
    s, fs = f.sheet, f.sheet.stamp or f.sheet.font
    at = s.x(offset)
    w, h = fs * 5.6, fs * 3.4
    x0 = min(max(at - CLEAR - w if left else at + CLEAR, 2), s.width - w - 2)
    y0 = f.page_y - CLEAR - h
    foot = min(max(at, x0 + 4), x0 + w - 4)
    tilt = -2.5 if cls == "pc" else 2.5
    return (
        f'<g class="station {cls}{_shown(f, offset)}" data-t="{offset}" '
        f'style="font-size:{fs}px">'
        f'<line x1="{foot:.1f}" x2="{at:.1f}" y1="{y0 + h:.1f}" y2="{f.page_y - 5:.1f}"/>'
        f'<g class="impress" filter="url(#{f.key}-ink-{cls})">'
        f'<g transform="rotate({tilt} {x0 + w / 2:.1f} {y0 + h / 2:.1f})">'
        f'<rect class="box" x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="2"/>'
        f'<rect class="rim" x="{x0 + 2.5:.1f}" y="{y0 + 2.5:.1f}" width="{w - 5:.1f}" '
        f'height="{h - 5:.1f}" rx="1"/>'
        + _text(x0 + w / 2, y0 + fs * 1.15, who, "who", text_anchor="middle")
        + _text(
            x0 + w / 2,
            y0 + h - fs * 0.6,
            hm(f.doc.origin_ts + offset),
            "when",
            text_anchor="middle",
        )
        + "</g></g></g>"
    )


def _remark(f: _Frame, row: ScanRow, p: float | None, d: float | None) -> str:
    """What the gap means, knocked out of the paper over the bar, or over the stamps if short.

    With both arrivals, the minutes are the largest numeral on the sheet and count up as the
    hand crosses the gap (`data-from`, `data-to`), with the word after them small, on one line.
    """
    said = verdict(row)
    if not said:
        return ""
    s, x, y = f.sheet, f.sheet.x, f.page_y
    big, small = s.font * 2.15, s.font * 1.08
    stamp_h = (s.stamp or s.font) * 3.4
    if p is not None and d is not None:
        a, b = sorted((p, d))
        number = minutes(b - a)
        rest = said[len(number) :].strip()
        lines = [[(number, "n", big), (rest, "w", small)]]
        centre = (x(a) + x(b)) / 2
        room = x(b) - x(a) + 8
        count = f' data-from="{a}" data-to="{b}"'
    else:
        a = next(t for t in (p, d) if t is not None)
        lines = [[(w, "w", small)] for w in _two_lines(said)]
        centre = x(a) + CLEAR + (s.stamp or s.font) * 2.8
        room, count = 0.0, ""
    width = max(sum((len(body) + 0.6) * size * 0.5 for body, _, size in line) for line in lines)
    last = y - 16 if width <= room else y - CLEAR - stamp_h - 10
    rows: list[str] = []
    for k, line in enumerate(lines):
        at = last - (len(lines) - 1 - k) * small * 1.2
        rows.append(
            f'<tspan x="{centre:.1f}" y="{at:.1f}">'
            + "".join(
                f'<tspan class="{cls}"{f' dx="{size * 0.3:.1f}"' if j else ""} '
                f'style="font-size:{size:.1f}px">{html.escape(body)}</tspan>'
                for j, (body, cls, size) in enumerate(line)
            )
            + "</tspan>"
        )
    return (
        f'<g class="verdict{_shown(f, a)}" data-t="{a}"{count}>'
        f'<text x="{centre:.1f}" y="{last:.1f}">{"".join(rows)}</text></g>'
    )


def _arrivals(f: _Frame) -> str:
    """Every arrival is a dot on the rule; the chosen series also gets stamps and a bar.

    Its stamps, the bar across the rule between them, the thick rule from Promcast's arrival
    on, and the bar's meaning in words. The bar sits under the sweep, so it grows as the hand
    crosses it; the thick rule lands whole when Promcast pages. Every Promcast arrival also has
    a faint dot from the start, where its faint thread meets the rule.
    """
    s, x, y, origin = f.sheet, f.sheet.x, f.page_y, f.doc.origin_ts
    out: list[str] = []
    for i, row in enumerate(f.doc.rows):
        a, b = row.promcast_breach_ts, row.linear_breach_ts
        if a is None and b is None:
            continue
        p = None if a is None else a - origin
        d = None if b is None else b - origin
        dots = "".join(
            f'<circle class="{cls}{_shown(f, t)}" data-t="{t}" cx="{x(t):.1f}" cy="{y:.1f}" '
            f'r="{r}"/>'
            for cls, t, r in (("pc-d", p, 4.5), ("lin-d", d, 4))
            if t is not None
        )
        swept: list[str] = []
        full = (
            ""
            if p is None
            else f'<line class="full{_shown(f, p)}" data-t="{p}" x1="{x(p):.1f}" '
            f'x2="{s.right:.1f}" y1="{y:.1f}" y2="{y:.1f}" filter="url(#{f.key}-bleed)"/>'
        )
        if p is not None and d is not None:
            x0, x1 = sorted((x(p), x(d)))
            box = f'x="{x0:.1f}" y="{y - 8:.1f}" width="{x1 - x0:.1f}" height="16"'
            swept.append(
                f'<rect class="bar-bg" {box}/>'
                f'<rect class="bar" {box} filter="url(#{f.key}-bleed)"/>'
            )
        first = p if d is None or (p is not None and p <= d) else d
        stamps: list[str] = []
        if p is not None:
            stamps.append(_station(f, p, "PROMCAST", "pc", left=p == first and d is not None))
        if d is not None:
            stamps.append(_station(f, d, "LINEAR", "ln", left=d == first and p is not None))
        stamps.append(_remark(f, row, p, d))
        # Under the series' own dot, over the lead's thick rule drawn before it.
        ghost = "" if p is None else f'<circle class="ghost-d" cx="{x(p):.1f}" cy="{y:.1f}" r="4"/>'
        state = " on" if i == 0 else " dim"
        out.append(
            f'<g class="arr{state}" data-i="{i}"><g class="swept">{full}'
            f'<g clip-path="url(#{f.key}-sweep)">{"".join(swept)}</g></g>{ghost}{dots}'
            f'<g class="lab">{"".join(stamps)}</g></g>'
        )
    return "".join(out)


def _two_lines(text: str) -> list[str]:
    """Break a sentence at the space nearest its middle."""
    words = text.split(" ")
    best = min(
        range(1, len(words)),
        key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))),
    )
    return [" ".join(words[:best]), " ".join(words[best:])]


def svg(doc: ScanDocument, paths: dict[str, Any], sheet: Sheet, still_s: float, key: str) -> str:
    """The sheet as SVG: grid, trains, the rule and its stamps, the hand at `still_s`."""
    origin, s = doc.origin_ts, sheet
    lead = trains(doc, paths)[0]
    frame = _Frame(doc, sheet, int(paths["step_s"]), key, s.y(lead.row.threshold), still_s)
    summary = " ".join(
        f"{name(r.labels)}: Promcast "
        + (hm(r.promcast_breach_ts) if r.promcast_breach_ts else "clear")
        + ", predict_linear "
        + (hm(r.linear_breach_ts) if r.linear_breach_ts else "clear")
        + "."
        for r in doc.rows
    )
    return (
        f'<svg class="sheet-svg" viewBox="0 0 {s.width} {s.height}" '
        f'style="font-size:{s.font}px" role="img" aria-labelledby="{key}-title" '
        f'data-left="{s.left}" data-right="{s.right}" data-top="{s.top}" '
        f'data-bottom="{s.bottom}" data-before="{s.before_s}" data-after="{s.after_s}" '
        f'data-w="{s.width}" data-h="{s.height}">'
        f'<title id="{key}-title">Timetable of {len(doc.rows)} series from '
        f"{hm(origin - s.before_s)} to {hm(origin + s.after_s)} UTC. {summary}</title>"
        + _defs(frame)
        + _grid(frame)
        + _trains(frame, paths)
        + _rule_and_hand(frame)
        + _arrivals(frame)
        + "</svg>"
    )


def load_paths(path: Path) -> dict[str, Any]:
    """Read the committed paths document."""
    return json.loads(path.read_text(encoding="utf-8"))
