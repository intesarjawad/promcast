import hashlib
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from promcast.core.evidence import estimate_lines
from promcast.shell.datasets import (
    FileLock,
    LockedFile,
    lock_sums,
    read_lock,
    read_sums,
    write_sums,
)
from promcast.shell.evidence import series_items, verify_nab_lock

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import fetch_nab
else:
    spec = importlib.util.spec_from_file_location("fetch_nab", ROOT / "scripts/fetch_nab.py")
    assert spec is not None
    assert spec.loader is not None
    fetch_nab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_nab)

CLOUDWATCH = "data/realAWSCloudwatch/a.csv"
LABELS = "labels/combined_labels.json"
SERIES = "timestamp,value\n" + "".join(
    f"2014-01-01 00:{minute:02d}:00,{minute}\n" for minute in range(0, 60, 5)
)
CONTENTS = {CLOUDWATCH: SERIES.encode(), LABELS: b"{}"}


def _locked(path: str, data: bytes) -> LockedFile:
    return LockedFile(
        path=path,
        commit=fetch_nab.NAB_COMMIT,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _lock() -> FileLock:
    return FileLock(
        repository=fetch_nab.NAB_REPOSITORY,
        files=tuple(_locked(path, data) for path, data in sorted(CONTENTS.items())),
    )


def _git_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def _raw_handler(contents: dict[str, bytes]) -> httpx.MockTransport:
    prefix = f"/{fetch_nab.NAB_REPOSITORY}/{fetch_nab.NAB_COMMIT}/"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "raw.githubusercontent.com"
        assert request.url.path.startswith(prefix)
        return httpx.Response(200, content=contents[request.url.path.removeprefix(prefix)])

    return httpx.MockTransport(handler)


def test_committed_lock_pins_every_protocol_file_at_the_commit():
    lock = read_lock(fetch_nab.LOCK_PATH)
    paths = [file.path for file in lock.files]
    assert lock.repository == fetch_nab.NAB_REPOSITORY
    assert paths == sorted(set(paths))
    assert {file.commit for file in lock.files} == {fetch_nab.NAB_COMMIT}
    assert all(fetch_nab.protocol_file(path) for path in paths)
    assert set(paths) >= fetch_nab.NAMED_FILES
    series = [path for path in paths if path.endswith(".csv")]
    assert any(path.startswith(fetch_nab.CLOUDWATCH_DIRECTORY) for path in series)
    assert estimate_lines(1, 0, 1.0).splitlines()[2].startswith(f"e3: {len(series)} series")


@pytest.mark.parametrize(
    ("path", "accepted"),
    [
        ("data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv", True),
        ("data/realKnownCause/machine_temperature_system_failure.csv", True),
        ("labels/combined_labels.json", True),
        ("data/realKnownCause/nyc_taxi.csv", False),
        ("data/realAWSCloudwatch/nested/a.csv", False),
        ("data/realAWSCloudwatch/../../../escape.csv", False),
        ("data/realAWSCloudwatch/README.md", False),
        ("labels/combined_windows.json", False),
    ],
)
def test_protocol_file_accepts_only_the_e3_corpus(path: str, *, accepted: bool):
    assert fetch_nab.protocol_file(path) is accepted


def test_install_nab_downloads_by_commit_and_manifests_the_lock(tmp_path: Path):
    lock = _lock()
    with httpx.Client(transport=_raw_handler(CONTENTS)) as client:
        fetch_nab.install_nab(lock, tmp_path, client)
    assert read_sums(tmp_path) == lock_sums(lock)
    assert (tmp_path / CLOUDWATCH).read_bytes() == CONTENTS[CLOUDWATCH]


def test_install_nab_rejects_bytes_that_differ_from_the_lock(tmp_path: Path):
    changed = {**CONTENTS, CLOUDWATCH: SERIES.replace(",55", ",56").encode()}
    with (
        httpx.Client(transport=_raw_handler(changed)) as client,
        pytest.raises(ValueError, match="sha256"),
    ):
        fetch_nab.install_nab(_lock(), tmp_path, client)
    assert not (tmp_path / CLOUDWATCH).exists()


def test_install_nab_rejects_a_truncated_download(tmp_path: Path):
    short = {**CONTENTS, LABELS: b"{"}
    with (
        httpx.Client(transport=_raw_handler(short)) as client,
        pytest.raises(ValueError, match="1 bytes, expected 2"),
    ):
        fetch_nab.install_nab(_lock(), tmp_path, client)


def test_install_nab_refuses_a_lock_entry_outside_the_protocol(tmp_path: Path):
    lock = FileLock(repository=fetch_nab.NAB_REPOSITORY, files=(_locked("../x.csv", b"x"),))
    with (
        httpx.Client(transport=_raw_handler({})) as client,
        pytest.raises(ValueError, match="outside the E3 protocol"),
    ):
        fetch_nab.install_nab(lock, tmp_path, client)


def _tree_handler(
    tree: dict[str, bytes], contents: dict[str, bytes], *, truncated: bool = False
) -> httpx.MockTransport:
    document = {
        "truncated": truncated,
        "tree": [
            {"path": "data", "type": "tree", "sha": "0" * 40},
            *(
                {"path": path, "type": "blob", "sha": _git_sha1(data)}
                for path, data in tree.items()
            ),
        ],
    }
    raw = _raw_handler(contents)

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == fetch_nab.TREE_URL:
            return httpx.Response(200, content=json.dumps(document).encode())
        return raw.handle_request(request)

    return httpx.MockTransport(handler)


def _upstream() -> dict[str, bytes]:
    named = {path: path.encode() for path in fetch_nab.NAMED_FILES}
    return {**named, CLOUDWATCH: SERIES.encode(), "data/realKnownCause/nyc_taxi.csv": b"skip"}


def test_build_lock_pins_only_protocol_files_from_verified_bytes(tmp_path: Path):
    upstream = _upstream()
    with httpx.Client(transport=_tree_handler(upstream, upstream)) as client:
        lock = fetch_nab.build_lock(client, tmp_path)
    expected = {path: data for path, data in upstream.items() if fetch_nab.protocol_file(path)}
    assert lock == FileLock(
        repository=fetch_nab.NAB_REPOSITORY,
        files=tuple(_locked(path, data) for path, data in sorted(expected.items())),
    )
    assert not (tmp_path / "data/realKnownCause/nyc_taxi.csv").exists()


def test_build_lock_rejects_bytes_that_differ_from_the_commit(tmp_path: Path):
    upstream = _upstream()
    served = {**upstream, CLOUDWATCH: b"tampered"}
    with (
        httpx.Client(transport=_tree_handler(upstream, served)) as client,
        pytest.raises(ValueError, match="git blob"),
    ):
        fetch_nab.build_lock(client, tmp_path)


@pytest.mark.parametrize(
    ("tree", "truncated", "message"),
    [
        (_upstream(), True, "truncated"),
        ({CLOUDWATCH: b"x"}, False, "lacks protocol files: data/realKnownCause"),
        (dict.fromkeys(fetch_nab.NAMED_FILES, b"x"), False, "realAWSCloudwatch/\\*.csv"),
    ],
)
def test_build_lock_refuses_an_incomplete_tree(
    tmp_path: Path, tree: dict[str, bytes], *, truncated: bool, message: str
):
    with (
        httpx.Client(transport=_tree_handler(tree, tree, truncated=truncated)) as client,
        pytest.raises(ValueError, match=message),
    ):
        fetch_nab.build_lock(client, tmp_path)


def test_main_names_a_malformed_lock_file(tmp_path: Path):
    lock_file = tmp_path / "nab.lock"
    lock_file.write_text('{"repository": "numenta/NAB", "files": []}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"nab\.lock is not a valid lock file"):
        fetch_nab.main(["--out", str(tmp_path / "out"), "--lock-file", str(lock_file)])


def _corpus(data_directory: Path) -> Path:
    root = data_directory / "NAB"
    for path, data in CONTENTS.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    write_sums(root, [root / path for path in CONTENTS])
    return root


def test_e3_loads_only_manifest_files_and_verifies_them(tmp_path: Path):
    root = _corpus(tmp_path)
    extra = root / "data/realKnownCause/nyc_taxi.csv"
    extra.parent.mkdir()
    extra.write_text(SERIES, encoding="utf-8")
    assert [item.series_id for item in series_items("e3", tmp_path, 0)] == [
        "realAWSCloudwatch/a.csv"
    ]
    (root / CLOUDWATCH).write_text(SERIES + "2014-01-01 01:00:00,60\n", encoding="utf-8")
    with pytest.raises(ValueError, match="remove it and refetch"):
        series_items("e3", tmp_path, 0)


def test_e3_without_a_manifest_names_the_fetch_script(tmp_path: Path):
    (_corpus(tmp_path) / "SHA256SUMS").unlink()
    with pytest.raises(
        FileNotFoundError, match=r"SHA256SUMS is missing; run scripts/fetch_nab\.py"
    ):
        series_items("e3", tmp_path, 0)


def test_e3_with_a_listed_file_deleted_names_the_fetch_script(tmp_path: Path):
    (_corpus(tmp_path) / LABELS).unlink()
    with pytest.raises(FileNotFoundError, match=r"is missing; run scripts/fetch_nab\.py"):
        series_items("e3", tmp_path, 0)


def test_verify_nab_lock_refuses_a_corpus_fetched_from_another_lock(tmp_path: Path):
    repository = tmp_path / "repo"
    (repository / "scripts").mkdir(parents=True)
    (repository / "scripts/nab.lock").write_text(_lock().model_dump_json(), encoding="utf-8")
    root = _corpus(tmp_path / "evidence")
    verify_nab_lock(repository, tmp_path / "evidence")
    write_sums(root, [root / CLOUDWATCH])
    with pytest.raises(ValueError, match=r"differ from scripts/nab\.lock; rerun"):
        verify_nab_lock(repository, tmp_path / "evidence")
    with pytest.raises(FileNotFoundError, match=r"run scripts/fetch_nab\.py first"):
        verify_nab_lock(repository, tmp_path / "missing")
