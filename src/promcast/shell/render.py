"""Terminal output for scans and command failures."""

import sys

from promcast.core.diagnostics import safe_urls
from promcast.shell.board import detect_color, scan_empty, scan_progress, terminal_text
from promcast.shell.board import error_report as _styled_error
from promcast.shell.errors import PromcastError
from promcast.shell.plotting import scan_board

BUG_REPORT_URL = "https://github.com/intesarjawad/promcast/issues/new?template=bug.yml"

__all__ = [
    "BUG_REPORT_URL",
    "bug_report",
    "error_report",
    "scan_board",
    "scan_empty",
    "scan_progress",
]


def error_report(error: PromcastError) -> str:
    """Render the two stderr lines every expected failure ends with, never wrapped."""
    return _styled_error(error.what, error.next_step, color=detect_color(sys.stderr))


def bug_report(error: Exception) -> str:
    """Render an unexpected failure as two lines that point to the bug tracker."""
    detail = safe_urls(" ".join(f"{type(error).__name__}: {error}".split()))
    return f"error: unexpected {terminal_text(detail)}\nnext:  report it at {BUG_REPORT_URL}\n"
