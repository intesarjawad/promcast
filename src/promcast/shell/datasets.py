"""Verify downloaded corpus files before they reach an evidence run."""

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

SUMS_FILENAME = "SHA256SUMS"
_CHUNK_BYTES = 1 << 20


class LockedFile(BaseModel):
    """One upstream file pinned by commit, size and digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    path: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FileLock(BaseModel):
    """Every upstream file one corpus reads, each pinned on its own."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    repository: str
    files: tuple[LockedFile, ...] = Field(min_length=1)


def read_lock(path: Path) -> FileLock:
    """Parse a lock file, naming the file when it is malformed."""
    try:
        return FileLock.model_validate_json(path.read_bytes())
    except ValidationError as error:
        raise ValueError(f"{path} is not a valid lock file: {error}") from error


def lock_sums(lock: FileLock) -> tuple[str, ...]:
    """Return the manifest lines a corpus fetched from this lock must have."""
    return tuple(f"{file.sha256}  {file.path}" for file in sorted(lock.files, key=lambda f: f.path))


def sha256_file(path: Path) -> str:
    """Hash a file with the digest the manifests record."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    """Reject a file whose bytes differ from the pinned digest."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path} has sha256 {actual}, expected {expected}; remove it and refetch")


def verify_locked(path: Path, file: LockedFile) -> None:
    """Reject a file whose size or digest differs from its lock entry."""
    size = path.stat().st_size
    if size != file.size:
        raise ValueError(f"{path} has {size} bytes, expected {file.size}; remove it and refetch")
    verify_sha256(path, file.sha256)


def git_blob_sha1(path: Path) -> str:
    """Compute the object id the Hub API reports for a non-LFS file."""
    data = path.read_bytes()
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


def verify_git_blob(path: Path, expected: str) -> None:
    """Reject a file whose git object id differs from the Hub tree entry."""
    actual = git_blob_sha1(path)
    if actual != expected:
        raise ValueError(
            f"{path} has git blob {actual}, expected {expected}; remove it and refetch"
        )


def download_checked(
    client: httpx.Client, url: str, destination: Path, check: Callable[[Path], None]
) -> None:
    """Download through a partial file, verify it, then replace the destination."""
    if destination.exists():
        check(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    stream.write(chunk)
        check(partial)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def write_sums(directory: Path, files: Iterable[Path]) -> None:
    """Record every corpus file's digest relative to the directory evidence reads."""
    lines = [
        f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}" for path in sorted(files)
    ]
    (directory / SUMS_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_sums(directory: Path) -> tuple[str, ...]:
    """Return the manifest lines verbatim so reports can quote them."""
    path = directory / SUMS_FILENAME
    if not path.is_file():
        return ()
    return tuple(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
