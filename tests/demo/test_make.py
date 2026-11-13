import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMAND = """
import json
import os
import sys
from pathlib import Path

with Path("commands.jsonl").open("a") as stream:
    stream.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + "\\n")
sys.exit(int(os.environ["COMMAND_EXIT"]))
"""


@pytest.mark.parametrize("exit_code", [0, 17])
@pytest.mark.parametrize(
    ("target", "command"),
    [
        ("coldstart", ["coldstart.sh"]),
        ("evidence", ["uv", "run", "python", "scripts/evidence.py"]),
        ("oci-compose", ["uv", "run", "python", "scripts/oci_compose.py"]),
        ("demo", ["docker", "compose", "up", "-d", "--build"]),
        ("demo-check", ["demo_check.sh", "--json", "demo.json"]),
        ("demo-check-full", ["demo_check.sh", "--full", "--json", "demo.json"]),
    ],
)
def test_demo_targets_forward_commands_and_failures(
    tmp_path: Path, target: str, command: list[str], exit_code: int
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for path in (
        tmp_path / "docker",
        tmp_path / "uv",
        scripts / "demo_check.sh",
        scripts / "coldstart.sh",
    ):
        path.write_text(f"#!{sys.executable}\n" + COMMAND)
        path.chmod(0o755)

    async def invoke():
        process = await asyncio.create_subprocess_exec(
            "make",
            "-f",
            str(ROOT / "Makefile"),
            target,
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                "COMMAND_EXIT": str(exit_code),
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return process.returncode, (stdout + stderr).decode()

    code, output = asyncio.run(invoke())
    assert (code == 0) == (exit_code == 0), output
    assert [
        json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()
    ] == [command]
