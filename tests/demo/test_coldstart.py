import asyncio
import hashlib
import io
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FAKE_COMMAND = r"""
import json
import os
import shutil
import sys
from pathlib import Path

root = Path(os.environ["COLDSTART_TEST_ROOT"])
command = Path(sys.argv[0]).name
args = sys.argv[1:]
with (root / "calls").open("a") as stream:
    stream.write(json.dumps([command, *args]) + "\n")
clock = root / "clock"
if command == "date":
    print(clock.read_text())
elif command == "uname":
    print(os.environ.get("TEST_PLATFORM", "Linux") if args == ["-s"] else "arm64")
elif command == "sleep":
    clock.write_text(str(int(clock.read_text()) + int(args[0])))
elif command == "curl":
    if "-o" in args:
        target = Path(args[args.index("-o") + 1])
        source = root / "release" / target.name
        if source.exists():
            shutil.copyfile(source, target)
        else:
            target.write_text("0" * 64 + "  prometheus-3.15.1.darwin-arm64.tar.gz\n"
                              if target.name == "sha256sums.txt" else "bad archive")
    elif any("/api/v1/status/runtimeinfo" in arg for arg in args):
        print(json.dumps({"status": "success", "data": {"startTime": "1970-01-01T00:16:40Z"}}))
    elif any(arg.startswith("query=ALERTS{") for arg in args):
        print(json.dumps({"status": "success", "data": {"resultType": "matrix", "result": [
            {"metric": {"alertname": "PromcastBreachSoon"}, "values": [[1008, "1"]]}]}}))
    elif any("/api/v1/query_range" in arg for arg in args):
        print(json.dumps({"status": "success", "data": {"resultType": "matrix", "result": [
            {"metric": {"target": "demo_disk_free"}, "values": [[1000, "-1"]]}]}}))
    elif any("/api/datasources/" in arg for arg in args):
        print('{"status":"OK"}')
    elif any(arg.endswith(":8787/metrics") for arg in args):
        print('promcast_selfaudit_coverage_ratio{'
              'target="demo_disk_free",lead="10m",method="toto"} 0.8')
    elif any("/api/v1/query" in arg for arg in args):
        state = root / "polls"
        count = int(state.read_text()) if state.exists() else 0
        state.write_text(str(count + 1))
        print(json.dumps({"status": "success", "data": {"result": [
            {"value": [1000, "32" if count else "31"]}]}}))
    else:
        print("ready")
elif command == "uv":
    if "--backfill" in args:
        Path(args[args.index("--out") + 1]).write_text("# EOF\n")
        Path(args[args.index("--anchor-file") + 1]).write_text("1000")
elif command in ("uvx", "docker"):
    assert sys.stdin.read() == ""
    if command == "docker" and "compose" in args:
        if "up" in args:
            clock.write_text(str(int(clock.read_text()) + 8))
        sys.exit(0)
    if command == "docker" and "inspect" in args:
        sys.exit(0)
    if command == "docker" and "rm" in args:
        sys.exit(0)
    if command == "docker" and "scan" not in args:
        print("test-container")
        sys.exit(0)
    if "TEST_SCAN_EXIT" in os.environ:
        sys.exit(int(os.environ["TEST_SCAN_EXIT"]))
    counter = root / "scans"
    count = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(count + 1))
    if command == "uvx":
        uv_cache = Path(os.environ["UV_CACHE_DIR"])
        model_cache = Path(os.environ["PROMCAST_CACHE_DIR"])
        if count == 0:
            assert not list(uv_cache.iterdir())
            assert not list(model_cache.iterdir())
            (uv_cache / "download").write_bytes(b"x" * 4096)
            (model_cache / "weights").write_bytes(b"x" * 4096)
        else:
            assert (uv_cache / "download").is_file()
            assert (model_cache / "weights").is_file()
    duration = int(os.environ.get("TEST_COLD_S", "12") if count == 0 else
                   os.environ.get("TEST_WARM_S", "3"))
    clock.write_text(str(int(clock.read_text()) + duration))
    print(json.dumps({"rows": json.loads(os.environ.get("TEST_ROWS",
        '[{"target":"prometheus-self", "labels":{"mountpoint":"/demo"}}]'))}))
"""


def run_coldstart(tmp_path: Path, *args: str, settings: dict[str, str] | None = None):
    commands = tmp_path / "bin"
    commands.mkdir()
    (tmp_path / "clock").write_text("1000")
    for command in ("uvx", "uv", "docker", "curl", "date", "uname", "sleep"):
        executable = commands / command
        executable.write_text(f"#!{sys.executable}\n" + FAKE_COMMAND)
        executable.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "COLDSTART_TEST_ROOT": str(tmp_path),
        **(settings or {}),
    }

    async def invoke():
        process = await asyncio.create_subprocess_exec(
            "bash",
            str(ROOT / "scripts/coldstart.sh"),
            *args,
            cwd=tmp_path,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
        return process.returncode, stdout.decode(), stderr.decode()

    code, stdout, stderr = asyncio.run(invoke())
    report = tmp_path / "coldstart.json"
    calls = tmp_path / "calls"
    return (
        code,
        json.loads(report.read_text()) if report.exists() else None,
        [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else [],
        stdout,
        stderr,
    )


def test_uvx_times_cold_and_warm_with_fresh_caches_and_closed_stdin(tmp_path: Path):
    wheel = tmp_path / "package with spaces.whl"
    wheel.touch()
    code, report, calls, _, stderr = run_coldstart(
        tmp_path, "--path", "uvx", "--wheel", str(wheel), "--prometheus", "http://metrics:9090"
    )
    assert code == 0, stderr
    assert report is not None
    assert report["seconds_to_board_cold"] == 12
    assert report["seconds_to_board_warm"] == 3
    assert report["download_bytes"] >= 8192
    scans = [call for call in calls if call[0] == "uvx"]
    assert (
        scans
        == [
            [
                "uvx",
                "--torch-backend",
                "cpu",
                "--from",
                str(wheel),
                "promcast",
                "scan",
                "http://metrics:9090",
                "--json",
            ]
        ]
        * 2
    )


@pytest.mark.parametrize("settings", [{"TEST_COLD_S": "121"}, {"TEST_WARM_S": "21"}])
def test_budget_failure_keeps_measurements(tmp_path: Path, settings: dict[str, str]):
    code, report, _, _, stderr = run_coldstart(
        tmp_path, "--prometheus", "http://metrics:9090", settings=settings
    )
    assert code == 1
    assert report is not None
    assert "budget exceeded" in stderr


def test_stock_polls_for_samples_and_cleans_up_container(tmp_path: Path):
    code, report, calls, _, stderr = run_coldstart(tmp_path, "--stock", "--pypi", "1.2.3")
    assert code == 0, stderr
    assert report is not None
    assert report["prometheus"] == "stock"
    assert (tmp_path / "polls").read_text() == "2"
    assert ["docker", "rm", "-f", "test-container"] in calls
    assert any("promcast==1.2.3" in call for call in calls)


def test_stock_requires_self_recipe(tmp_path: Path):
    code, _, calls, _, stderr = run_coldstart(
        tmp_path, "--stock", settings={"TEST_ROWS": '[{"target":"other"}]'}
    )
    assert code == 2
    assert "Missing prometheus-self" in stderr
    assert ["docker", "rm", "-f", "test-container"] in calls


def test_docker_release_removes_exact_tag_and_routes_to_host(tmp_path: Path):
    code, report, calls, _, stderr = run_coldstart(
        tmp_path,
        "--path",
        "docker",
        "--pypi",
        "1.2.3",
        "--prometheus",
        "http://localhost:9090",
        settings={"TEST_PLATFORM": "Darwin"},
    )
    assert code == 0, stderr
    assert report is not None
    assert report["path"] == "docker"
    assert ["docker", "image", "rm", "ghcr.io/intesarjawad/promcast:1.2.3"] in calls
    assert any(arg == "http://host.docker.internal:9090" for call in calls for arg in call)


def test_docker_local_image_is_kept(tmp_path: Path):
    code, _, calls, _, stderr = run_coldstart(
        tmp_path, "--path", "docker", "--prometheus", "http://metrics:9090"
    )
    assert code == 0, stderr
    assert not any(call[:3] == ["docker", "image", "rm"] for call in calls)
    assert any("--network" in call and "host" in call for call in calls)


def test_native_download_rejects_bad_checksum_before_extraction(tmp_path: Path):
    code, _, calls, _, stderr = run_coldstart(
        tmp_path, "--stock", settings={"TEST_PLATFORM": "Darwin"}
    )
    assert code == 2
    assert "checksum mismatch" in stderr
    assert any(any("darwin-arm64.tar.gz" in arg for arg in call) for call in calls)
    assert not any(call[0] == "uvx" for call in calls)


@pytest.mark.parametrize(
    "args",
    [
        ("--path", "invalid"),
        ("--wheel",),
        ("--unknown",),
        ("--stock", "--prometheus", "http://metrics:9090"),
        ("--path", "docker", "--wheel", "file.whl"),
        ("--path", "oci", "--stock"),
        ("--wheel", "a", "--pypi", "1.2.3"),
    ],
)
def test_invalid_arguments_fail_before_setup(tmp_path: Path, args: tuple[str, ...]):
    code, report, calls, _, _ = run_coldstart(tmp_path, *args)
    assert code == 2
    assert report is None
    assert not calls


def test_oci_starts_with_closed_stdin_checks_demo_and_cleans_up(tmp_path: Path):
    code, report, calls, _, stderr = run_coldstart(tmp_path, "--path", "oci", "--pypi", "1.2.3")
    assert code == 0, stderr
    assert report is not None
    assert report["path"] == "oci"
    assert report["seconds_to_up"] == 8
    assert report["seconds_to_firing"] == 8
    assert report["seconds_to_board_cold"] == 8
    compose = [call for call in calls if call[:2] == ["docker", "compose"]]
    assert compose[0][-2:] == ["up", "-d"]
    assert compose[1][-2:] == ["down", "-v"]
    assert "oci://ghcr.io/intesarjawad/promcast-demo:1.2.3" in compose[0]


def test_release_candidate_names_the_image_by_its_semver_tag(tmp_path: Path):
    code, _, calls, _, stderr = run_coldstart(tmp_path, "--path", "oci", "--pypi", "1.2.3rc1")
    assert code == 0, stderr
    assert any("oci://ghcr.io/intesarjawad/promcast-demo:1.2.3-rc.1" in call for call in calls)


def test_docker_release_candidate_pulls_the_semver_tag(tmp_path: Path):
    code, _, calls, _, stderr = run_coldstart(
        tmp_path, "--path", "docker", "--pypi", "1.2.3rc1", "--prometheus", "http://metrics:9090"
    )
    assert code == 0, stderr
    assert ["docker", "image", "rm", "ghcr.io/intesarjawad/promcast:1.2.3-rc.1"] in calls


def test_version_without_an_image_tag_fails_before_setup(tmp_path: Path):
    code, report, calls, _, stderr = run_coldstart(
        tmp_path, "--path", "docker", "--pypi", "1.2.3.dev1"
    )
    assert code == 2
    assert report is None
    assert not calls
    assert "has no image tag" in stderr


@pytest.mark.parametrize("platform", ["Darwin", "MINGW64_NT-10.0"])
@pytest.mark.parametrize("stock", [True, False])
def test_native_archive_is_verified_and_configured(tmp_path: Path, platform: str, stock: bool):
    windows = platform.startswith("MINGW")
    directory = "prometheus-3.15.1." + ("windows-amd64" if windows else "darwin-arm64")
    suffix = ".exe" if windows else ""
    executable = (
        b'#!/bin/sh\ncase "$0" in\n'
        b'  *promtool*) test -f "$4" ;;\n'
        b'  *) cat prometheus.yml > "$COLDSTART_TEST_ROOT/native-config" ;;\nesac\n'
    )
    files = {
        f"prometheus{suffix}": executable,
        f"promtool{suffix}": executable,
        "prometheus.yml": b"bundled: true\n",
    }
    release = tmp_path / "release"
    release.mkdir()
    archive = release / (directory + (".zip" if windows else ".tar.gz"))
    if windows:
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, content in files.items():
                entry = zipfile.ZipInfo(f"{directory}/{name}")
                entry.external_attr = 0o100755 << 16
                bundle.writestr(entry, content)
    else:
        with tarfile.open(archive, "w:gz") as bundle:
            for name, content in files.items():
                entry = tarfile.TarInfo(f"{directory}/{name}")
                entry.size = len(content)
                entry.mode = 0o755
                bundle.addfile(entry, io.BytesIO(content))
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (release / "sha256sums.txt").write_text(f"{checksum}  {archive.name}\n")
    args = ("--stock",) if stock else ()
    code, report, calls, _, stderr = run_coldstart(
        tmp_path, *args, settings={"TEST_PLATFORM": platform}
    )
    assert code == 0, stderr
    assert report is not None
    config = (tmp_path / "native-config").read_text()
    if stock:
        assert config == "bundled: true\n"
    else:
        assert "localhost:9100" in config
        assert any("--backfill" in call for call in calls)
        assert any("--serve" in call for call in calls)
    assert not any(call[0] == "docker" for call in calls)


def test_scan_failure_preserves_exit_code_and_cleans_up_demo(tmp_path: Path):
    code, report, calls, _, _ = run_coldstart(tmp_path, settings={"TEST_SCAN_EXIT": "7"})
    assert code == 7
    assert report is None
    compose = [call for call in calls if call[:2] == ["docker", "compose"]]
    assert compose[0][-4:] == ["up", "-d", "prometheus", "scenario"]
    assert compose[-1][-2:] == ["down", "-v"]
