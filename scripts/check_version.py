"""Fail unless a release tag names the version in pyproject.toml."""

import argparse
import sys
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]


def mismatch(tag: str, pyproject: Path) -> str | None:
    """Return why the tag disagrees with pyproject, or None when they agree."""
    with pyproject.open("rb") as stream:
        declared = tomllib.load(stream)["project"]["version"]
    if not tag.startswith("v"):
        return f"tag {tag!r} must start with 'v', for example v{declared}"
    try:
        tagged = Version(tag.removeprefix("v"))
    except InvalidVersion:
        return f"tag {tag!r} is not a PEP 440 version; tag v{declared} instead"
    if tagged != Version(declared):
        return (
            f"tag {tag!r} is version {tagged} but {pyproject.name} says {declared}; "
            f"set the pyproject version to {tagged} or tag v{declared}"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    """Print the mismatch and exit 1, or exit 0 when the tag is the release version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="the pushed git tag, such as v0.1.0-rc.1")
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    arguments = parser.parse_args(argv)
    problem = mismatch(arguments.tag, arguments.pyproject)
    if problem is not None:
        sys.stderr.write(f"{problem}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
