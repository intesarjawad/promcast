"""Verify public claims against their specific evidence fields.

Public prose rejects numbers outside a field-bound claim. Evidence
rows bind every cell to its corpus, method and field. Generated site content is
checked against its builder by make check-copy.
"""

import argparse
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from number_claims import bound_text, write_readme
from pydantic import BaseModel, TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "docs/evidence.md",
    "bench.json",
    "demo.json",
    "coldstart.json",
    "docs/sources/prices.json",
)
README_SECTIONS = ("## Try it", "## Evidence", "## Performance")
ALLOWLIST = ROOT / "scripts/number_allowlist.txt"
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_COPY_NUMBER = re.compile(rf"(?<![A-Za-z_\d.,])(?:{_NUMBER})")
_SOURCE_NUMBER = re.compile(_NUMBER)
_INLINE_CODE = re.compile(r"`[^`]*`")
_URL = re.compile(r"\]\([^)]*\)|<?https?://[^\s)>]+>?")


def _fence(line: str) -> bool:
    return line.lstrip().startswith(("```", "~~~"))


def prose_lines(lines: list[str], selected: Iterable[int]) -> Iterator[tuple[int, str]]:
    """Yield selected lines outside fenced code, without inline code or URLs."""
    wanted = set(selected)
    fenced = False
    for index, line in enumerate(lines):
        if _fence(line):
            fenced = not fenced
            continue
        if fenced or index not in wanted:
            continue
        yield index + 1, _URL.sub(" ", _INLINE_CODE.sub(" ", line))


def readme_lines(lines: list[str]) -> tuple[list[int], list[str]]:
    """Pick the hero line and the checked sections; also name the sections that are absent."""
    selected: list[int] = []
    title = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    if title is not None:
        hero = next(
            (
                i
                for i in range(title + 1, len(lines))
                if lines[i].strip() and not lines[i].lstrip().startswith(("#", "!", "[", "<", "`"))
            ),
            None,
        )
        if hero is not None:
            selected.append(hero)
    missing: list[str] = []
    for heading in README_SECTIONS:
        start = next((i for i, line in enumerate(lines) if line.strip() == heading), None)
        if start is None:
            missing.append(heading)
            continue
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines)
        )
        selected.extend(range(start + 1, end))
    return selected, missing


def strip_names(line: str, names: Iterable[str]) -> str:
    """Blank out allowlisted names so the numbers inside them are not checked."""
    for name in sorted(names, key=len, reverse=True):
        line = re.sub(rf"(?<![\w.]){re.escape(name)}(?!\.?\d)", " ", line, flags=re.IGNORECASE)
    return line


def source_numbers(root: Path, sources: Iterable[str] = SOURCES) -> set[str]:
    """Collect every number the measurement files and the dated sources contain."""
    numbers: set[str] = set()
    for name in sources:
        path = root / name
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            numbers.update(match.replace(",", "") for match in _SOURCE_NUMBER.findall(text))
    return numbers


def is_sourced(token: str, known: set[str]) -> bool:
    """Accept a number printed by a source, exactly or rounded to the decimals shown."""
    plain = token.replace(",", "")
    if plain in known:
        return True
    _, _, decimals = plain.partition(".")
    return bool(decimals) and any(f"{float(value):.{len(decimals)}f}" == plain for value in known)


def unsourced(
    path: Path, lines: list[str], selected: Iterable[int], names: list[str], known: set[str]
) -> list[str]:
    """Describe every number in the selected prose that no source contains."""
    return [
        f"{path}:{number}: {token} is in none of {', '.join(SOURCES)}; print it from a script, "
        f"quote its source in docs/sources/prices.json, or add the name that contains it to "
        f"{ALLOWLIST.relative_to(ROOT)}"
        for number, line in prose_lines(lines, selected)
        for token in _COPY_NUMBER.findall(strip_names(line, names))
        if not is_sourced(token, known)
    ]


class _Method(BaseModel):
    n: int
    incidents: int
    recall: float
    precision: float
    false_alerts_per_week: float
    median_lead_s: float | None
    recall_ci: tuple[float, float] | None
    lead_ci: tuple[float, float] | None


class _Corpus(BaseModel):
    methods: dict[str, _Method]


class _Evidence(BaseModel):
    corpora: dict[str, _Corpus]


def named_claims(root: Path, path: Path, text: str) -> list[str]:
    """Bind E1 headline counts and evidence-table cells to their structured fields."""
    evidence = _Evidence.model_validate_json((root / "site/data/evidence.json").read_text())
    problems: list[str] = []
    if path.name == "README.md":
        pattern = re.compile(
            r"(toto-2\.0-\d+m) caught (\d+) of (\d+) incidents "
            r"and linear extrapolation caught (\d+)",
            re.I,
        )
        for claim in pattern.finditer(text):
            model, caught, incidents, linear = claim.groups()
            for name, count in ((model.lower(), caught), ("linear", linear)):
                score = evidence.corpora["e1"].methods[name]
                if int(incidents) != score.incidents or int(count) != round(
                    score.recall * score.incidents
                ):
                    problems.append(
                        f"{path}: E1 {name} count disagrees with "
                        f"site/data/evidence.json:corpora.e1.methods.{name}.incidents/recall"
                    )
                if (
                    "no false alerts from either" in text[claim.end() :].split(".", 1)[0]
                    and score.false_alerts_per_week != 0
                ):
                    problems.append(f"{path}: E1 {name}.false_alerts_per_week is not zero")
    if path.name == "evidence.md":
        problems.extend(evidence_rows(evidence, path, text))
    return problems


def evidence_rows(evidence: _Evidence, path: Path, text: str) -> list[str]:
    """Check each method table column against its corresponding evidence field."""
    problems: list[str] = []
    corpus = ""
    fields = tuple(_Method.model_fields)
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            corpus = line[3:].strip().lower()
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not line.startswith("| ") or not cells or cells[0] in {"Method", "---"}:
            continue
        if corpus not in evidence.corpora or cells[0] not in evidence.corpora[corpus].methods:
            problems.append(f"{path}:{number}: unknown evidence row {corpus}.{cells[0]}")
            continue
        score = evidence.corpora[corpus].methods[cells[0]]
        if len(cells) != len(fields) + 1:
            problems.append(f"{path}:{number}: expected {len(fields)} evidence fields")
            continue
        for field, cell in zip(fields, cells[1:], strict=True):
            expected = getattr(score, field)
            actual = (
                None
                if cell == "unavailable"
                else TypeAdapter(float | tuple[float, float]).validate_json(cell)
            )
            if expected is not None and isinstance(expected, float):
                matches = isinstance(actual, int | float) and f"{expected:.4f}" == f"{actual:.4f}"
            else:
                matches = actual == expected
            if not matches:
                problems.append(
                    f"{path}:{number}: {corpus}.{cells[0]}.{field} "
                    "disagrees with site/data/evidence.json"
                )
    return problems


def field_bound_numbers(root: Path, path: Path, text: str, names: list[str]) -> list[str]:
    """Reject quantitative prose without a binding for that exact claim."""
    remaining = bound_text(root, path, text)
    identifiers = [name for name in names if not name.isdigit()] + ["SHA-256", "UTF-8"]
    return [
        f"{path}:{number}: {token} is in none of the field-bound claims; "
        "bind this claim to its specific data field and print it from a repo script"
        for number, line in prose_lines(remaining.splitlines(), range(len(remaining.splitlines())))
        for token in _COPY_NUMBER.findall(strip_names(line, identifiers))
    ]


def main(argv: list[str] | None = None) -> int:
    """Check complete public documents, or the supplied file, with field-qualified claims."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="check every number in this file")
    parser.add_argument(
        "--write", action="store_true", help="regenerate README evidence from its fields"
    )
    arguments = parser.parse_args(argv)
    problems: list[str] = []
    try:
        if arguments.write:
            write_readme(ROOT)
        paths = (
            [arguments.path]
            if arguments.path
            else [
                ROOT / "README.md",
                ROOT / "docs/evidence.md",
                *sorted((ROOT / "site/_templates").glob("*.html")),
            ]
        )
        names = ALLOWLIST.read_text(encoding="utf-8").splitlines()
        for path in paths:
            text = path.read_text(encoding="utf-8")
            problems.extend(named_claims(ROOT, path, text))
            problems.extend(field_bound_numbers(ROOT, path, text, names))
    except (OSError, ValueError, KeyError) as error:
        problems.append(f"claim verification failed: {error}")
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
