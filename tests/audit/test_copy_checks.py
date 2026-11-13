import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEAN = "# Promcast\n\nForecast alerts for the Prometheus you already run.\n"
# Split so this file does not trip the same checks on the real repository.
PERSONAL_EMAIL = "someone@" + "gmail.com"


def _run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> int:
    executable = shutil.which(command[0])
    assert executable is not None
    # Fixed test commands only; nothing here comes from outside the test.
    return subprocess.run(  # noqa: S603
        [executable, *command[1:]], cwd=cwd, env=env, capture_output=True, text=True, check=False
    ).returncode


def _repo(path: Path, files: dict[str, str]) -> Path:
    shutil.copy(ROOT / "Makefile", path / "Makefile")
    for name, text in {"README.md": CLEAN, **files}.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    for command in (
        ("git", "init", "-q"),
        ("git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "add", "."),
        (
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "-q",
            "-m",
            "Add docs",
        ),
    ):
        assert _run(*command, cwd=path) == 0
    return path


def _check_copy(path: Path) -> int:
    commands = path / "bin"
    commands.mkdir()
    (commands / "uv").write_text("#!/bin/sh\nexit 0\n")
    (commands / "uv").chmod(0o755)
    env = {**os.environ, "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}"}
    return _run("make", "--no-print-directory", "check-copy", cwd=path, env=env)


def test_clean_copy_passes(tmp_path: Path):
    assert _check_copy(_repo(tmp_path, {})) == 0


def test_personal_email_fails_the_target(tmp_path: Path):
    files = {"src/contact.py": f'EMAIL = "{PERSONAL_EMAIL}"\n'}
    assert _check_copy(_repo(tmp_path, files)) != 0
