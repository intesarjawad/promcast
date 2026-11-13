import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from promcast.shell.datasets import read_sums

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import fetch_boom
else:
    spec = importlib.util.spec_from_file_location("fetch_boom", ROOT / "scripts/fetch_boom.py")
    assert spec is not None
    assert spec.loader is not None
    fetch_boom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_boom)

PAYLOAD_SHA256 = "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5"
PAYLOAD_GIT_SHA1 = "47d05ff6403c8e6c3cf635ea6eb9263738432773"


def test_remote_files_prefers_lfs_digest_and_keeps_git_oid():
    document = [
        {
            "type": "file",
            "path": "dataset_taxonomy.json",
            "oid": PAYLOAD_GIT_SHA1,
            "size": 7,
        },
        {
            "type": "file",
            "path": "data/a.arrow",
            "oid": "ignored",
            "size": 7,
            "lfs": {"oid": PAYLOAD_SHA256, "size": 7},
        },
        {"type": "directory", "path": "data"},
    ]
    files = fetch_boom.remote_files(document)
    assert [file.path for file in files] == ["data/a.arrow", "dataset_taxonomy.json"]
    assert files[0].sha256 == PAYLOAD_SHA256
    assert files[0].git_sha1 == "ignored"
    assert files[1].sha256 is None
    assert files[1].git_sha1 == PAYLOAD_GIT_SHA1


def test_remote_files_refuses_an_entry_without_a_digest():
    document = [{"type": "file", "path": "data/a.arrow", "size": 7}]
    with pytest.raises(ValueError, match="no digest"):
        fetch_boom.remote_files(document)


def test_manifest_reads_the_pinned_tree():
    document = [{"type": "file", "path": "a", "oid": PAYLOAD_GIT_SHA1, "size": 7}]

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == fetch_boom.TREE_URL
        return httpx.Response(200, content=json.dumps(document))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        files = fetch_boom.manifest(client)
    assert [file.path for file in files] == ["a"]


def test_manifest_follows_the_next_link():
    first = [{"type": "file", "path": "a", "oid": PAYLOAD_GIT_SHA1, "size": 7}]
    second = [{"type": "file", "path": "b", "oid": PAYLOAD_GIT_SHA1, "size": 7}]
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if str(request.url) == fetch_boom.TREE_URL:
            return httpx.Response(
                200,
                content=json.dumps(first),
                headers={"Link": '<https://example.test/page2>; rel="next"'},
            )
        return httpx.Response(200, content=json.dumps(second))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        files = fetch_boom.manifest(client)
    assert [file.path for file in files] == ["a", "b"]
    assert seen == [fetch_boom.TREE_URL, "https://example.test/page2"]


def test_install_boom_verifies_each_digest_and_writes_the_manifest(tmp_path: Path):
    files = (
        fetch_boom.RemoteFile("data/a.bin", PAYLOAD_SHA256, "ignored"),
        fetch_boom.RemoteFile("dataset_taxonomy.json", None, PAYLOAD_GIT_SHA1),
    )
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"payload")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        fetch_boom.install_boom(files, tmp_path, client, "Datadog/BOOM", "revision")
    assert (tmp_path / "data/a.bin").read_bytes() == b"payload"
    assert (tmp_path / "dataset_taxonomy.json").read_bytes() == b"payload"
    assert sorted(seen) == [
        "https://huggingface.co/datasets/Datadog/BOOM/resolve/revision/data/a.bin",
        "https://huggingface.co/datasets/Datadog/BOOM/resolve/revision/dataset_taxonomy.json",
    ]
    assert read_sums(tmp_path) == (
        f"{PAYLOAD_SHA256}  data/a.bin",
        f"{PAYLOAD_SHA256}  dataset_taxonomy.json",
    )


def test_install_boom_refuses_a_path_outside_the_destination(tmp_path: Path):
    files = (fetch_boom.RemoteFile("../escape", PAYLOAD_SHA256, None),)
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client,
        pytest.raises(ValueError, match="escapes"),
    ):
        fetch_boom.install_boom(files, tmp_path, client, "repo", "revision")
