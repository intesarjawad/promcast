"""Fetch the pinned BOOM dataset and verify every file's digest."""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx
from pydantic import BaseModel, TypeAdapter

from promcast.shell.datasets import (
    download_checked,
    verify_git_blob,
    verify_sha256,
    write_sums,
)

LOGGER = logging.getLogger(__name__)
BOOM_REPOSITORY = "Datadog/BOOM"
BOOM_REVISION = "69325b544c45ff0d6c43c7a99c49a6601a01725b"
TREE_URL = (
    f"https://huggingface.co/api/datasets/{BOOM_REPOSITORY}/tree/"
    f"{BOOM_REVISION}?recursive=true&limit=1000"
)
DEFAULT_DIRECTORY = Path.home() / ".cache" / "promcast" / "evidence" / "BOOM"


class _Lfs(BaseModel):
    oid: str
    size: int


class _TreeEntry(BaseModel):
    type: str
    path: str
    oid: str | None = None
    size: int | None = None
    lfs: _Lfs | None = None


@dataclass(frozen=True)
class RemoteFile:
    """One Hub tree entry with the digest that verifies its bytes."""

    path: str
    sha256: str | None
    git_sha1: str | None


def remote_files(document: object) -> tuple[RemoteFile, ...]:
    """Keep file entries and prefer the LFS SHA-256 over the git blob id."""
    entries = TypeAdapter(list[_TreeEntry]).validate_python(document)
    files: list[RemoteFile] = []
    for entry in entries:
        if entry.type != "file":
            continue
        sha256 = None if entry.lfs is None else entry.lfs.oid
        if sha256 is None and entry.oid is None:
            raise ValueError(f"Hub tree entry has no digest: {entry.path}")
        files.append(RemoteFile(entry.path, sha256, entry.oid))
    return tuple(sorted(files, key=lambda item: item.path))


def manifest(client: httpx.Client) -> tuple[RemoteFile, ...]:
    """Follow every tree page without downloading series bytes."""
    url: str | None = TREE_URL
    files: list[RemoteFile] = []
    while url is not None:
        response = client.get(url)
        response.raise_for_status()
        document: object = response.json()
        files.extend(remote_files(document))
        next_link = response.links.get("next")
        url = None if next_link is None else next_link["url"]
    if not files:
        raise ValueError("Hub tree contains no files")
    return tuple(sorted(files, key=lambda item: item.path))


def _target(destination: Path, relative: str) -> Path:
    root = destination.resolve()
    target = (destination / relative).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"dataset path escapes {destination}: {relative}")
    return target


def _checker(file: RemoteFile) -> Callable[[Path], None]:
    if file.sha256 is not None:
        expected = file.sha256
        return lambda path: verify_sha256(path, expected)
    if file.git_sha1 is not None:
        expected = file.git_sha1
        return lambda path: verify_git_blob(path, expected)
    raise ValueError(f"file has no digest: {file.path}")


def install_boom(
    files: tuple[RemoteFile, ...],
    destination: Path,
    client: httpx.Client,
    repository: str,
    revision: str,
) -> None:
    """Download each missing file to its tree path and write one checksum manifest."""
    base = f"https://huggingface.co/datasets/{repository}/resolve/{revision}"
    targets: list[Path] = []
    for file in files:
        target = _target(destination, file.path)
        download_checked(client, f"{base}/{quote(file.path)}", target, _checker(file))
        targets.append(target)
    write_sums(destination, targets)


def main() -> int:
    """Fetch BOOM at the pinned revision into the evidence cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DIRECTORY)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        files = manifest(client)
        install_boom(files, arguments.out, client, BOOM_REPOSITORY, BOOM_REVISION)
    LOGGER.info("Verified %d BOOM files in %s", len(files), arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
