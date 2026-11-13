import html
import importlib
import re
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from rich.cells import cell_len

from promcast.shell import board

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
load: Callable[[Path], board.ScanResult] = importlib.import_module("render_board").load

SCAN = Path(__file__).resolve().parents[2] / "docs/design/fleet-scan.json"
TEXT = re.compile(
    r'<text class="[^"]*" x="([\d.]+)" y="([\d.]+)" textLength="([\d.]+)"[^>]*>(.)</text>'
)
CELL = 12.2
LINE = 24.4


def glyph_x(svg: str, glyph: str) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y, _, g in TEXT.findall(svg) if g == glyph]


@pytest.mark.parametrize("width", [44, 80, 160])
def test_marks_and_ticks_sit_on_the_one_time_scale(width: int) -> None:
    result = load(SCAN)
    doc = result.document
    geo = board.geometry(result, width)
    svg = board.scan_svg(result, width=width)

    def line_y(line: int) -> float:
        return 20 + LINE * line

    crosses = glyph_x(svg, "✕")
    diamonds = glyph_x(svg, "◆")
    for i, row in enumerate(doc.rows):
        y = line_y(geo.first_lane_line + i * geo.line_stride)
        cross, diamond = board.marks(row, doc.origin_ts, geo.span, geo.size)
        for glyph, found, cell in (("✕", crosses, cross), ("◆", diamonds, diamond)):
            hits = [x for x, gy in found if abs(gy - y) < 1]
            if cell is None:
                continue
            assert any(abs(x - (geo.lane_col + cell) * CELL) < 0.2 for x in hits), (glyph, i)
        if row.promcast_breach_ts is not None and cross:
            seconds = row.promcast_breach_ts - doc.origin_ts
            expected = (geo.lane_col + board.cell_at(seconds, geo.span, geo.size)) * CELL
            assert any(abs(x - expected) < 0.2 for x, gy in crosses if abs(gy - y) < 1)

    ticks = {round(x, 1) for x, _ in glyph_x(svg, "┼")}
    for seconds in range(0, geo.span + 1, 3600):
        expected = (geo.lane_col + board.cell_at(seconds, geo.span, geo.size)) * CELL
        if geo.span % 3600 == 0:
            assert round(expected, 1) in ticks, seconds


@pytest.mark.parametrize("label", ["東京" * 30, "e\u0301" * 60])
def test_svg_label_lengths_match_terminal_display_cells(label: str):
    result = load(SCAN)
    first = result.document.rows[0].model_copy(
        update={"labels": {"instance": label}, "show_labels": ["instance"]}
    )
    result = replace(result, document=result.document.model_copy(update={"rows": [first]}))
    svg = board.scan_svg(result, width=80)
    lengths = [
        (float(length), html.unescape(body))
        for length, body in re.findall(r'<text[^>]*textLength="([\d.]+)"[^>]*>([^<]*)</text>', svg)
        if label[:2] in html.unescape(body)
    ]
    assert lengths
    for length, body in lengths:
        assert length == pytest.approx(CELL * cell_len(body), abs=0.1)


def test_svg_rulers_preserve_graphemes_and_their_cell_offsets():
    result = load(SCAN)
    first = result.document.rows[0].model_copy(
        update={"labels": {"instance": "東京é─"}, "show_labels": ["instance"]}
    )
    result = replace(result, document=result.document.model_copy(update={"rows": [first]}))
    geo = board.geometry(result, 80)
    svg = board.scan_svg(result, width=80)
    y = 20 + LINE * geo.first_lane_line
    assert f'x="73.2" y="{y:.1f}" textLength="12.2"' in svg
    assert ">é</text>" in svg
    assert f'x="85.4" y="{y:.1f}" textLength="12.2"' in svg
