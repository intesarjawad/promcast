"""Fetch the NAB files the replay protocol scores, each verified against scripts/nab.lock.

With --lock, list the pinned commit's tree, check every protocol file against its git
blob id, and rewrite the lock from the verified bytes.
"""

import argparse
import logging
from pathlib import Path

import httpx
from pydantic import BaseModel

from promcast.shell.datasets import (
    FileLock,
    LockedFile,
    download_checked,
    read_lock,
    sha256_file,
    verify_git_blob,
    verify_locked,
    write_sums,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "scripts" / "nab.lock"
NAB_REPOSITORY = "numenta/NAB"
NAB_COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"
TREE_URL = f"https://api.github.com/repos/{NAB_REPOSITORY}/git/trees/{NAB_COMMIT}?recursive=1"
DEFAULT_DIRECTORY = Path.home() / ".cache" / "promcast" / "evidence" / "NAB"
# The E3 corpus in docs/protocol.md: every AWS CloudWatch series, three machine metrics,
# and the point labels for the secondary table.
CLOUDWATCH_DIRECTORY = "data/realAWSCloudwatch/"
NAMED_FILES = frozenset(
    {
        "data/realKnownCause/machine_temperature_system_failure.csv",
        "data/realKnownCause/cpu_utilization_asg_misconfiguration.csv",
        "data/realKnownCause/ec2_request_latency_system_failure.csv",
        "labels/combined_labels.json",
    }
)


class _TreeEntry(BaseModel):
    path: str
    type: str
    sha: str


class _Tree(BaseModel):
    truncated: bool
    tree: list[_TreeEntry]


def protocol_file(path: str) -> bool:
    """Accept only the files the protocol names, which also keeps every path inside the corpus."""
    name = path.removeprefix(CLOUDWATCH_DIRECTORY)
    in_cloudwatch = name != path and "/" not in name and name.endswith(".csv")
    return in_cloudwatch or path in NAMED_FILES


def raw_url(repository: str, file: LockedFile) -> str:
    """Address a file by commit so the bytes cannot move under the lock."""
    return f"https://raw.githubusercontent.com/{repository}/{file.commit}/{file.path}"


def install_nab(lock: FileLock, destination: Path, client: httpx.Client) -> None:
    """Download each locked file that is missing, verify every one, and manifest them."""
    targets: list[Path] = []
    for file in lock.files:
        if not protocol_file(file.path):
            raise ValueError(f"lock names a file outside the E3 protocol: {file.path}")
        target = destination / file.path
        download_checked(
            client,
            raw_url(lock.repository, file),
            target,
            lambda path, file=file: verify_locked(path, file),
        )
        targets.append(target)
    write_sums(destination, targets)


def build_lock(client: httpx.Client, destination: Path) -> FileLock:
    """Pin every protocol file at the commit by the size and SHA-256 of its verified bytes."""
    response = client.get(TREE_URL)
    response.raise_for_status()
    tree = _Tree.model_validate(response.json())
    if tree.truncated:
        raise ValueError(f"GitHub truncated the tree listing at {TREE_URL}; lock not written")
    entries = sorted(
        (entry for entry in tree.tree if entry.type == "blob" and protocol_file(entry.path)),
        key=lambda entry: entry.path,
    )
    missing = NAMED_FILES - {entry.path for entry in entries}
    if missing or len(entries) == len(NAMED_FILES):
        absent = ", ".join(sorted(missing)) or f"{CLOUDWATCH_DIRECTORY}*.csv"
        raise ValueError(f"NAB {NAB_COMMIT} lacks protocol files: {absent}")
    files: list[LockedFile] = []
    for entry in entries:
        target = destination / entry.path
        url = f"https://raw.githubusercontent.com/{NAB_REPOSITORY}/{NAB_COMMIT}/{entry.path}"
        download_checked(
            client, url, target, lambda path, sha=entry.sha: verify_git_blob(path, sha)
        )
        files.append(
            LockedFile(
                path=entry.path,
                commit=NAB_COMMIT,
                size=target.stat().st_size,
                sha256=sha256_file(target),
            )
        )
    return FileLock(repository=NAB_REPOSITORY, files=tuple(files))


def main(argv: list[str] | None = None) -> int:
    """Fetch the locked files, or regenerate the lock first with --lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--lock-file", type=Path, default=LOCK_PATH)
    parser.add_argument(
        "--lock", action="store_true", help="rebuild the lock from the pinned commit first"
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        if arguments.lock:
            lock = build_lock(client, arguments.out)
            arguments.lock_file.write_text(lock.model_dump_json(indent=2) + "\n", encoding="utf-8")
            LOGGER.info("Wrote %d entries to %s", len(lock.files), arguments.lock_file)
        else:
            lock = read_lock(arguments.lock_file)
        install_nab(lock, arguments.out, client)
    LOGGER.info("Verified %d NAB files in %s", len(lock.files), arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
