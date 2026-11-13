"""Fail unless every installed package uses a licence compatible with Apache-2.0 distribution.

Reads `pip-licenses --format=json` on standard input. Any package outside the allowed
families is printed and needs an ADR before launch.
"""

import re
import sys

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

ALLOWED_FAMILIES = {
    "MIT": re.compile(r"\bMIT\b"),
    "BSD": re.compile(r"\b0?BSD\b"),
    "Apache": re.compile(r"\bApache\b"),
    "PSF": re.compile(r"\bPSF\b|Python Software Foundation"),
    "ISC": re.compile(r"\bISC\b"),
    "MPL-2.0": re.compile(r"\bMPL[- ]?2\.0\b|Mozilla Public License 2\.0"),
    "LGPL": re.compile(r"\bLGPL|Lesser General Public License"),
    "Unlicense": re.compile(r"\bUnlicense\b"),
}
# Packages whose metadata falls outside the families above, each accepted at one version by
# the ADR named; a new version needs the review again.
_LICENCE_ADR = "docs/adr/0018-dependency-licences.md"
ACCEPTED = {
    ("augurs", "0.10.3"): _LICENCE_ADR,
    ("dd-unit-scaling", "0.1.0"): _LICENCE_ADR,
    ("numpy", "2.5.4"): _LICENCE_ADR,
    ("regex", "2026.9.29"): _LICENCE_ADR,
    ("text-unidecode", "1.3"): _LICENCE_ADR,
    ("torch", "2.15.0"): _LICENCE_ADR,
    ("torch", "2.15.0+cpu"): _LICENCE_ADR,
}


class Package(BaseModel):
    """One row of pip-licenses JSON output."""

    model_config = ConfigDict(populate_by_name=True)
    name: str = Field(alias="Name")
    version: str = Field(alias="Version")
    licence: str = Field(alias="License")


def _allowed_term(term: str) -> bool:
    return any(pattern.search(term) for pattern in ALLOWED_FAMILIES.values())


def allowed(licence: str) -> bool:
    """Accept a licence when every required part is allowed and each choice has an allowed side.

    A semicolon separates trove classifiers, which may cover different parts of a package, so
    each must pass; SPDX `OR` offers a choice and SPDX `AND` combines terms.
    """
    # shortcut: parentheses are dropped, so AND binds loosest and "(A AND B) OR C" is judged
    # stricter than it is; parse SPDX properly if a real package needs that form.
    expression = licence.replace("(", " ").replace(")", " ").strip()
    if not expression:
        return False
    required = re.split(r"\s+AND\s+|\s*;\s*", expression)
    return all(
        any(_allowed_term(choice) for choice in re.split(r"\s+OR\s+", part))
        for part in required
        if part
    )


def disallowed(packages: list[Package]) -> list[Package]:
    """Return the packages whose licence falls outside the allowed families and no ADR accepts."""
    return [
        package
        for package in packages
        if not allowed(package.licence) and (package.name, package.version) not in ACCEPTED
    ]


def main() -> int:
    """Print each package that needs an ADR and exit 1, or exit 0 when all are allowed."""
    try:
        packages = TypeAdapter(list[Package]).validate_json(sys.stdin.read())
    except ValidationError as error:
        sys.stderr.write(f"expected pip-licenses --format=json on standard input: {error}\n")
        return 1
    rejected = disallowed(packages)
    for package in rejected:
        sys.stdout.write(f"{package.name} {package.version}: {package.licence}\n")
    if rejected:
        sys.stderr.write(
            f"{len(rejected)} of {len(packages)} packages are outside the allowed licence "
            f"families ({', '.join(ALLOWED_FAMILIES)}); record each in an ADR before launch\n"
        )
        return 1
    sys.stdout.write(f"Every package uses an allowed licence ({len(packages)} checked)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
