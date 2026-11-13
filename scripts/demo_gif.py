"""Record a running Compose demo and the seeded timetable: make demo-gif."""

import mimetypes
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TextIO

from playwright.sync_api import Route, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PUBLIC = "https://promcast.intesarjawad.com/"
BRAVE = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
FPS = 12


def _run(
    *args: str,
    stdout: TextIO | int | None = None,
    stderr: int | None = None,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(args, cwd=ROOT, check=True, stdout=stdout, stderr=stderr, env=env)  # noqa: S603


def main() -> None:  # noqa: PLR0915
    """Capture real product output and encode it with a GIF palette."""
    with tempfile.TemporaryDirectory(prefix="promcast-demo-") as tmp:
        work = Path(tmp)
        frames = work / "frames"
        frames.mkdir()
        environment = os.environ.copy()
        environment.pop("NO_COLOR", None)
        environment["TERM"] = "xterm-256color"
        scan = [
            "docker",
            "compose",
            "exec",
            "-T",
            "-e",
            "COLUMNS=44",
            "-e",
            "PROMETHEUS_URL=http://prometheus:9090",
            "-e",
            "PROMCAST_CACHE_DIR=/opt/promcast/models",
            "promcast",
            "promcast",
            "scan",
            "--redact",
            "--svg",
            "/tmp/demo-board.svg",  # noqa: S108
        ]
        # The full check runs throughout capture and supplies the measured demo.json.
        with (work / "check.log").open("w") as check_log:
            check = subprocess.Popen(  # noqa: S603
                [str(ROOT / "scripts/demo_check.sh"), "--full", "--json", str(ROOT / "demo.json")],
                cwd=ROOT,
                stdout=check_log,
                stderr=subprocess.STDOUT,
            )
            try:
                with (work / "scan.log").open("w") as log:
                    _run(*scan, stdout=log, stderr=subprocess.STDOUT, env=environment)
                _run(
                    "docker",
                    "compose",
                    "cp",
                    "promcast:/tmp/demo-board.svg",
                    str(work / "board.svg"),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                shim = work / "uvx"
                shim.write_text(
                    "#!/bin/sh\nshift 3\nexec docker compose exec -T "
                    "-e COLUMNS=44 -e PYTHONWARNINGS=ignore "
                    "-e PROMETHEUS_URL=http://prometheus:9090 "
                    '-e PROMCAST_CACHE_DIR=/opt/promcast/models promcast promcast "$@"\n'
                )
                shim.chmod(0o755)
                tape = work / "scan.tape"
                tape.write_text(
                    f'Output "{work / "scan.mp4"}"\nSet Width 1280\nSet Height 720\n'
                    'Set FontSize 25\nSet Shell "bash"\nHide\n'
                    f'Type "export PATH={work}:$PATH; export PYTHONWARNINGS=ignore; clear" Enter\n'
                    'Sleep 1s\nShow\nType@45ms "uvx --torch-backend cpu promcast scan --redact"\n'
                    "Enter\nSleep 35s\n"
                )
                with (work / "vhs.log").open("w") as log:
                    _run("vhs", str(tape), stdout=log, stderr=subprocess.STDOUT, env=environment)
                svg = (work / "board.svg").read_text()
                board = (
                    '<!doctype html><meta charset="utf-8"><style>'
                    "body{margin:0;background:#171717;color:#f2f0e9;font:32px monospace}"
                    "header{padding:24px 40px 0}small{float:right;color:#bcb8aa;font-size:23px}"
                    "svg{display:block;width:100%;height:590px}"
                    "</style><header>uvx --torch-backend cpu promcast scan --redact"
                    "<small>Synthetic Compose demo</small></header>" + svg
                )
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(
                        executable_path=os.environ.get("PROMCAST_BROWSER", BRAVE)
                    )
                    page = browser.new_page(viewport={"width": 1280, "height": 720})

                    def route(request: Route) -> None:
                        path = request.request.url.removeprefix(PUBLIC).split("?", 1)[0]
                        if path == "capture-board":
                            request.fulfill(body=board, content_type="text/html")
                            return
                        if path == "capture-caption":
                            request.fulfill(
                                body="<style>body{margin:0;color:#bcb8aa;font:23px monospace}"
                                "span{position:fixed;right:40px;bottom:35px}</style>"
                                "<span>Synthetic Compose demo</span>",
                                content_type="text/html",
                            )
                            return
                        file = (SITE / (path or "index.html")).resolve()
                        if file.is_relative_to(SITE) and file.is_file():
                            request.fulfill(
                                body=file.read_bytes(),
                                content_type=mimetypes.guess_type(file.name)[0]
                                or "application/octet-stream",
                            )
                        else:
                            request.abort()

                    page.route("**/*", route)
                    page.goto(PUBLIC + "capture-caption")
                    page.screenshot(path=str(work / "caption.png"), omit_background=True)
                    page.goto(PUBLIC + "capture-board")
                    page.evaluate("document.fonts.ready")
                    page.screenshot(path=str(work / "board.png"))
                    page.clock.install()
                    page.goto(PUBLIC)
                    page.evaluate("document.fonts.ready")
                    page.evaluate("""() => {
                      const caption = document.createElement('div');
                      caption.textContent = 'Synthetic seeded fleet';
                      Object.assign(caption.style, {position:'fixed', right:'20px', bottom:'16px',
                        background:'#efece4', color:'#1a1916', padding:'6px 10px',
                        font:'bold 18px sans-serif', zIndex:'9999'});
                      document.body.appendChild(caption);
                    }""")
                    for index in range(72):
                        page.clock.run_for(1000 // FPS)
                        page.screenshot(path=str(frames / f"{index:03}.png"))
                    browser.close()
                # First frame is the scan board; the typed command and the timetable follow.
                _run(
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-loop",
                    "1",
                    "-framerate",
                    str(FPS),
                    "-t",
                    "2",
                    "-i",
                    str(work / "board.png"),
                    "-i",
                    str(work / "scan.mp4"),
                    "-framerate",
                    str(FPS),
                    "-i",
                    str(frames / "%03d.png"),
                    "-loop",
                    "1",
                    "-i",
                    str(work / "caption.png"),
                    "-sseof",
                    "-2",
                    "-i",
                    str(work / "scan.mp4"),
                    "-filter_complex",
                    f"[0:v]fps={FPS},setsar=1[b];[3:v]split[c1][c2];[1:v]trim=duration=3,setpts=PTS-STARTPTS,fps={FPS},setsar=1[v];[v][c1]overlay=shortest=1[t];[4:v]setpts=PTS-STARTPTS,fps={FPS},setsar=1[o];[o][c2]overlay=shortest=1[r];[2:v]fps={FPS},setsar=1[s];[b][t][r][s]concat=n=4:v=1:a=0,split[x][y];[x]palettegen=max_colors=128[p];[y][p]paletteuse=dither=sierra2_4a",
                    "-loop",
                    "0",
                    str(ROOT / "docs/demo.gif"),
                )
                if check.wait():
                    raise RuntimeError("demo-check-full failed; inspect demo.json")
            finally:
                if check.poll() is None:
                    check.terminate()
                    check.wait()


if __name__ == "__main__":
    main()
