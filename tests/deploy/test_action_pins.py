import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github/workflows").glob("*.yml"))
USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<reference>\S+)(?P<rest>.*)$")
PINNED = re.compile(r"(?P<action>[\w.-]+/[\w./-]+)@(?P<sha>[0-9a-f]{40})")
VERSION_COMMENT = re.compile(r"\s+# (?P<version>v\d+(?:\.\d+){0,2})\s*")


def pins(text: str) -> list[tuple[int, str, str]]:
    """Return (line, reference, trailing text) for every action a workflow uses."""
    return [
        (number, match["reference"], match["rest"])
        for number, line in enumerate(text.splitlines(), start=1)
        if (match := USES.match(line))
    ]


def violations(text: str) -> list[str]:
    """Name each action not pinned to a full commit SHA with its version in a comment."""
    found: list[str] = []
    for number, reference, rest in pins(text):
        if reference.startswith(("./", "docker://")):
            continue
        if not PINNED.fullmatch(reference) or not VERSION_COMMENT.fullmatch(rest):
            found.append(f"line {number}: {reference}{rest}")
    return found


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_every_action_is_pinned_to_a_commit_with_its_version(workflow: Path):
    assert violations(workflow.read_text(encoding="utf-8")) == []


def test_each_action_resolves_to_one_commit_and_version_everywhere():
    seen: dict[str, set[tuple[str, str]]] = {}
    for workflow in WORKFLOWS:
        for _, reference, rest in pins(workflow.read_text(encoding="utf-8")):
            pinned = PINNED.fullmatch(reference)
            comment = VERSION_COMMENT.fullmatch(rest)
            if pinned and comment:
                seen.setdefault(pinned["action"], set()).add((pinned["sha"], comment["version"]))
    assert seen
    assert {action: refs for action, refs in seen.items() if len(refs) > 1} == {}


@pytest.mark.parametrize(
    "line",
    [
        "      - uses: actions/checkout@v7",
        "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "      - uses: actions/checkout@3d3c42e # v7.0.1",
        "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # latest",
    ],
)
def test_tags_short_shas_and_missing_versions_are_refused(line: str):
    assert violations(f"steps:\n{line}\n") == [f"line 2: {line.split('uses: ')[1]}"]


def test_local_actions_and_a_pinned_reference_pass():
    text = (
        "steps:\n"
        "  - uses: ./.github/actions/setup\n"
        "  - uses: github/codeql-action/init@33d7b85da1a04171e211befe2c739303b9a4c066 # v4.39.3\n"
    )
    assert violations(text) == []
