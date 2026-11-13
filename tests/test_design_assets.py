import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from promcast.core.config import ModelConfig
from promcast.shell import board

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "docs/design/fleet-scan.json"
PAGES = ("index.html", "social.html", "board.svg", "board-narrow.svg")


def render(*arguments: str) -> str:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "scripts/render_board.py"), str(SCAN), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_ansi16_board_uses_the_nearest_standard_colours():
    board = render("--color", "standard")
    assert "\x1b[103m" in board
    assert "38;2" not in board


def test_redacted_board_hides_every_label():
    board = render("--redact")
    assert "db-01" not in board
    assert "host-" in board


def test_svg_asks_for_no_network_font(tmp_path: Path):
    target = tmp_path / "board.svg"
    render("--svg", str(target))
    svg = target.read_text(encoding="utf-8")
    assert "Martian Mono" in svg
    assert "@font-face" not in svg
    assert "url(http" not in svg


def build(tmp_path: Path) -> dict[str, str]:
    site = tmp_path / "site"
    shutil.copytree(ROOT / "site/_templates", site / "_templates", dirs_exist_ok=True)
    script = (
        "import sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); import build_site; "
        "build_site.SITE = Path(sys.argv[2]); "
        "sys.argv = ['build_site.py', '--no-png']; build_site.main()"
    )
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(ROOT / "scripts"), str(site)], check=True, cwd=ROOT
    )
    return {name: (site / name).read_text(encoding="utf-8") for name in PAGES}


def test_site_numbers_come_from_the_scan(tmp_path: Path):
    # durations and percentages never break across lines
    page = build(tmp_path)["index.html"].replace("\u00a0", " ").replace("&nbsp;", " ")
    assert "1h 15m" in page
    assert "db-01: predicted crossing at 15:15." in page
    assert "predict_linear: 16:00, 45 min later." in page
    assert "predict_linear says 2h 00m" in page
    assert "45m later" in page
    assert "difference between two predicted crossing times from the same forecast origin" in page
    assert "unpaged" not in page.lower()
    assert "{{" not in page


def test_site_build_is_deterministic_and_self_contained(tmp_path: Path):
    first = build(tmp_path)
    assert build(tmp_path) == first
    for name, text in first.items():
        assert "cdn." not in text, name
        assert 'src="http' not in text, name
    assert 'class="sheet-svg"' in first["index.html"]


def test_social_preview_is_the_documented_size():
    png = (ROOT / "docs/design/social-preview.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert int.from_bytes(png[16:20]) == 1280
    assert int.from_bytes(png[20:24]) == 640
    assert (ROOT / "site/og.png").read_bytes() == png


@pytest.mark.slow
def test_committed_fleet_scan_is_the_generators_output():
    model = Path(
        os.environ.get("PROMCAST_TOTO_DIR", ModelConfig().cache_dir.expanduser() / "toto-2.0-22m")
    )
    if not (model / "config.json").is_file():
        pytest.skip(f"needs the toto-2.0-22m files in {model}")
    check = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "scripts/synthetic_fleet.py"),
            "--check",
            str(SCAN),
            "--model-dir",
            str(model),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr


# The file was written on Apple Silicon. Torch's CPU kernels round differently on x86-64, so
# elsewhere each value may move by the last digit the file keeps (DIGITS = 5, so 1e-5). Measured
# on linux/amd64 against the committed file: 45 of 5040 numbers differ, all by exactly 1e-5, none
# in the history, same series, labels and structure. The tolerance is two units of that digit.
GENERATED_HERE = platform.system() == "Darwin" and platform.machine() == "arm64"
FLEET_PATHS_ATOL = 2e-5


@pytest.mark.slow
def test_committed_fleet_paths_are_the_generators_output():
    model = Path(
        os.environ.get("PROMCAST_TOTO_DIR", ModelConfig().cache_dir.expanduser() / "toto-2.0-22m")
    )
    if not (model / "config.json").is_file():
        pytest.skip(f"needs the toto-2.0-22m files in {model}")
    check = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "scripts/fleet_paths.py"),
            "--check",
            str(ROOT / "docs/design/fleet-paths.json"),
            "--model-dir",
            str(model),
            *([] if GENERATED_HERE else ["--atol", str(FLEET_PATHS_ATOL)]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr


@pytest.mark.parametrize(
    ("tty", "env", "expected"),
    [
        (False, {}, None),
        (True, {"NO_COLOR": "1"}, None),
        (True, {"TERM": "dumb"}, None),
        (True, {"COLORTERM": "truecolor"}, "truecolor"),
        (True, {}, "standard"),
    ],
)
def test_colour_system_follows_the_stream_and_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tty: bool,
    env: dict[str, str],
    expected: str | None,
):
    for name in ("NO_COLOR", "TERM", "COLORTERM"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: tty)
    assert board.detect_color(sys.stdout) == expected


def test_committed_site_matches_the_builder(tmp_path: Path):
    for name, expected in build(tmp_path).items():
        assert (ROOT / "site" / name).read_text(encoding="utf-8") == expected, name
