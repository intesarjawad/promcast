"""Print the container image tag for the package version in pyproject.toml.

The image, the Helm chart and the Compose artifact share one SemVer tag, derived here from
the PEP 440 package version: 0.1.0rc1 becomes 0.1.0-rc.1 and 0.1.0 stays 0.1.0. Standard
library only, so shell scripts and workflows can run it with any Python 3.11 or later.
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_RELEASE = re.compile(r"(?P<release>\d+\.\d+\.\d+)(?:(?P<label>a|b|rc)(?P<number>\d+))?")


def image_tag(version: str) -> str:
    """Map a normalised X.Y.Z or X.Y.Z{a,b,rc}N package version to its SemVer image tag."""
    match = _RELEASE.fullmatch(version)
    if match is None:
        raise ValueError(
            f"package version {version!r} has no image tag; use X.Y.Z or X.Y.Z{{a,b,rc}}N "
            "in pyproject.toml"
        )
    if match["label"] is None:
        return match["release"]
    return f"{match['release']}-{match['label']}.{match['number']}"


def package_version(pyproject: Path) -> str:
    """Read the declared package version."""
    with pyproject.open("rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def main(argv: list[str] | None = None) -> int:
    """Print the tag, or explain why the version has none."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="a package version; default: pyproject.toml")
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    arguments = parser.parse_args(argv)
    try:
        tag = image_tag(arguments.version or package_version(arguments.pyproject))
    except ValueError as error:
        sys.stderr.write(f"{error}\n")
        return 1
    sys.stdout.write(f"{tag}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
