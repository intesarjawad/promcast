import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from rich.cells import cell_len

from promcast.shell import board, render
from promcast.shell.errors import NoPrometheusFoundError, TsdbUnavailableError
from promcast.shell.scan import ScanDocument, ScanResult, ScanRow, ScanSkipped

ORIGIN = 1700000000
URL = "http://localhost:9090"
SNAPSHOTS = Path(__file__).parent / "__snapshots__"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FALLING = [float(15 - i) for i in range(16)]


def row(instance: str = "host-000", **changes: object) -> ScanRow:
    values: dict[str, object] = {
        "target": "disk",
        "labels": {"instance": instance, "mountpoint": "/data"},
        "value": 0.057,
        "unit": "ratio",
        "threshold": 0.05,
        "promcast_breach_ts": ORIGIN + 5400,
        "linear_breach_ts": ORIGIN + 720,
        "missing_fraction": 0.0,
        "short_history_min": None,
        "horizon_s": 7680,
        "show_labels": ["instance", "mountpoint"],
        "forecast_values": FALLING,
    }
    return ScanRow.model_validate(values | changes)


def result(
    rows: list[ScanRow], skipped: list[ScanSkipped] | None = None, dropped: int = 0, total: int = 0
) -> ScanResult:
    document = ScanDocument(
        url=URL, model="toto-2.0-22m", origin_ts=ORIGIN, rows=rows, skipped=skipped or []
    )
    return ScanResult(document, [], dropped, total, 1000, 600000)


def warming(instance: str, minutes: int | None) -> ScanSkipped:
    return ScanSkipped(
        target="disk", labels={"instance": instance}, reason="too_short", warmup_minutes=minutes
    )


FLAT = [0.5] * 16
STATES: dict[str, ScanResult] = {
    "breach": result([row()]),
    "past-threshold": result(
        [
            row(
                value=0.031,
                promcast_breach_ts=ORIGIN,
                linear_breach_ts=ORIGIN - 60,
                forecast_values=FLAT,
            )
        ]
    ),
    "clear": result(
        [
            row(
                value=0.42,
                promcast_breach_ts=None,
                linear_breach_ts=None,
                missing_fraction=0.125,
                forecast_values=FLAT,
            )
        ]
    ),
    "warming": result([], [warming("host-000", 3), warming("host-001", None)]),
    "truncated": result([row()], dropped=70, total=120),
    "long-labels": result(
        [
            row(
                "ip-10-0-123-45.eu-west-1.compute.internal:9100",
                value=0.041,
                labels={
                    "instance": "ip-10-0-123-45.eu-west-1.compute.internal:9100",
                    "mountpoint": "/var/lib/docker/containers",
                },
            )
        ]
    ),
    "many-rows": result(
        [
            row(
                f"host-{i:03}",
                value=0.05,
                promcast_breach_ts=None,
                linear_breach_ts=None,
                show_labels=["instance"],
                forecast_values=[],
            )
            for i in range(21)
        ]
    ),
}


@pytest.mark.parametrize("width", [44, 80, 160])
@pytest.mark.parametrize("state", STATES)
def test_scan_states_match_their_snapshots(state: str, width: int) -> None:
    board = render.scan_board(STATES[state], width=width)
    assert board == (SNAPSHOTS / f"scan-{state}-{width}.txt").read_text()
    assert max(map(len, board.splitlines())) <= width


@pytest.mark.parametrize("width", [44, 80, 160])
def test_colour_modes_keep_the_plain_layout(width: int) -> None:
    plain = render.scan_board(STATES["breach"], width=width)
    coloured = render.scan_board(STATES["breach"], width=width, color="truecolor")
    assert "38;2;215;255;47" in coloured
    assert ANSI.sub("", coloured).replace("█", " ") == plain.replace("█", " ")
    assert "\x1b" not in plain


def test_narrow_board_is_lane_only() -> None:
    board = render.scan_board(STATES["breach"], width=44).splitlines()
    assert " host-000 /data" in board
    lane = board[len(board) - 1 - board[::-1].index(" host-000 /data") + 1]
    assert "✕" in lane
    assert lane.endswith("1h 30m")


def test_past_threshold_lane_is_a_full_row_that_says_to_page() -> None:
    board = render.scan_board(STATES["past-threshold"], width=80)
    assert "past 5 % free · page now" in board
    assert "Past threshold" in board


EMPTY_NAMES = ["disk", "memory", "prometheus-self", "prometheus-tsdb", "pvc", "windows-disk"]
PROGRESS = {
    "discovery": ("discovery", 0, 0),
    "forecast-start": ("forecast", 0, 40),
    "forecast-mid": ("forecast", 12, 40),
    "forecast-done": ("forecast", 40, 40),
}
ERRORS = {
    "unreachable": TsdbUnavailableError("cannot reach http://prometheus:9090: connection refused"),
    "none-found": NoPrometheusFoundError(["http://localhost:9090", "http://prometheus:9090"]),
}


@pytest.mark.parametrize("width", [44, 80, 160])
def test_empty_scan_matches_its_snapshot(width: int) -> None:
    text = render.scan_empty(URL, EMPTY_NAMES, width=width)
    assert text == (SNAPSHOTS / f"scan-empty-{width}.txt").read_text()
    assert max(map(len, text.splitlines())) <= width
    coloured = render.scan_empty(URL, EMPTY_NAMES, width=width, color="truecolor")
    assert ANSI.sub("", coloured) == text


@pytest.mark.parametrize("width", [44, 80])
@pytest.mark.parametrize("state", PROGRESS)
def test_progress_line_matches_its_snapshot(state: str, width: int) -> None:
    stage, done, total = PROGRESS[state]
    line = render.scan_progress(stage, done, total, width=width)
    assert line + "\n" == (SNAPSHOTS / f"scan-progress-{state}-{width}.txt").read_text()
    assert "\n" not in line
    assert len(line) <= width
    coloured = render.scan_progress(stage, done, total, width=width, color="truecolor")
    assert ANSI.sub("", coloured).replace("█", " ") == line.replace("█", " ")


@pytest.mark.parametrize("state", ERRORS)
def test_error_report_keeps_two_unwrapped_lines(
    state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    text = render.error_report(ERRORS[state])
    assert text == (SNAPSHOTS / f"error-{state}.txt").read_text()
    assert len(text.splitlines()) == 2
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.delenv("NO_COLOR", raising=False)
    coloured = render.error_report(ERRORS[state])
    assert "\x1b[1;38;2;255;74;61merror:" in coloured
    assert ANSI.sub("", coloured) == text


@pytest.mark.parametrize("color", [None, "standard", "truecolor"])
@pytest.mark.parametrize(
    "payload",
    [
        "\x1b[2J",
        "\x1b[31m",
        "\x1b]52;c;c2VjcmV0\x1b\\",
        "\x9b2J",
        "\nnext: forged\r\b\t",
        "\u202e\u2028",
    ],
)
def test_untrusted_board_text_cannot_control_the_terminal(color: board.Color, payload: str):
    baseline = result([row("host")], [warming("host", None)])
    document = baseline.document.model_copy(
        update={
            "url": "http://host/" + payload,
            "model": "model" + payload,
            "rows": [row("host" + payload, unit="bytes" + payload, promcast_breach_ts=ORIGIN)],
            "skipped": [
                warming("host" + payload, None).model_copy(update={"reason": "reason" + payload})
            ],
        }
    )
    outputs = [
        board.scan_board(replace(baseline, document=document), color=color),
        board.scan_empty("http://host/" + payload, ["recipe" + payload], color=color),
        board.error_report("detail" + payload, "edit file" + payload, color=color),
    ]
    for output in outputs:
        plain = ANSI.sub("", output)
        assert all(
            ord(char) >= 32 and not 127 <= ord(char) <= 159 for char in plain if char != "\n"
        )
        assert "\u202e" not in plain
        assert "\u2028" not in plain
        assert "\nnext: forged" not in plain
    assert baseline.document.rows[0].labels["instance"] == "host"
    assert document.rows[0].labels["instance"] == "host" + payload
    assert board.terminal_text(payload) in ANSI.sub("", outputs[-1])


@pytest.mark.parametrize("label", ["東京" * 30, "e\u0301" * 60, "界a" * 30, "👩‍💻" * 30])
@pytest.mark.parametrize("color", [None, "standard", "truecolor"])
def test_wide_labels_keep_time_left_visible(label: str, color: board.Color):
    output = ANSI.sub("", render.scan_board(result([row(label)]), width=80, color=color))
    lane = next(
        line for line in output.splitlines() if "✕" in line and "◆" in line and "1h 30m" in line
    )
    assert lane.endswith("1h 30m")
    assert cell_len(lane) <= 80
