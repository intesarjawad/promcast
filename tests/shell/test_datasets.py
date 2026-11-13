from pathlib import Path

import httpx
import pytest

from promcast.shell.datasets import (
    SUMS_FILENAME,
    download_checked,
    git_blob_sha1,
    read_sums,
    sha256_file,
    verify_git_blob,
    verify_sha256,
    write_sums,
)

PAYLOAD_SHA256 = "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5"
OTHER_SHA256 = "d9298a10d1b0735837dc4bd85dac641b0f3cef27a47e5d53a54f2f3f5b2fcffa"
PAYLOAD_GIT_SHA1 = "47d05ff6403c8e6c3cf635ea6eb9263738432773"


def test_verify_sha256_accepts_the_pinned_bytes_and_rejects_a_change(tmp_path: Path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")
    verify_sha256(path, PAYLOAD_SHA256)
    assert sha256_file(path) == PAYLOAD_SHA256
    path.write_bytes(b"other")
    with pytest.raises(ValueError, match="sha256"):
        verify_sha256(path, PAYLOAD_SHA256)


def test_git_blob_digest_matches_the_hub_object_id(tmp_path: Path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")
    assert git_blob_sha1(path) == PAYLOAD_GIT_SHA1
    verify_git_blob(path, PAYLOAD_GIT_SHA1)
    with pytest.raises(ValueError, match="git blob"):
        verify_git_blob(path, "0" * 40)


def test_download_streams_once_then_reuses_the_verified_file(tmp_path: Path):
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=b"payload")

    destination = tmp_path / "nested" / "payload.bin"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        download_checked(
            client,
            "https://example.test/payload",
            destination,
            lambda path: verify_sha256(path, PAYLOAD_SHA256),
        )
        download_checked(
            client,
            "https://example.test/payload",
            destination,
            lambda path: verify_sha256(path, PAYLOAD_SHA256),
        )
    assert requests == ["https://example.test/payload"]
    assert destination.read_bytes() == b"payload"
    assert not destination.with_name("payload.bin.part").exists()


def test_failed_download_removes_the_partial_file(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"other")

    destination = tmp_path / "payload.bin"
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError, match="sha256"),
    ):
        download_checked(
            client,
            "https://example.test/payload",
            destination,
            lambda path: verify_sha256(path, PAYLOAD_SHA256),
        )
    assert not destination.exists()
    assert not destination.with_name("payload.bin.part").exists()


def test_manifest_round_trips_relative_paths_and_known_digests(tmp_path: Path):
    (tmp_path / "a.txt").write_bytes(b"payload")
    (tmp_path / "b.txt").write_bytes(b"other")
    write_sums(tmp_path, [tmp_path / "b.txt", tmp_path / "a.txt"])
    assert read_sums(tmp_path) == (
        f"{PAYLOAD_SHA256}  a.txt",
        f"{OTHER_SHA256}  b.txt",
    )
    assert (tmp_path / SUMS_FILENAME).read_text().endswith("\n")
