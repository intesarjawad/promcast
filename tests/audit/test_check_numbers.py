import importlib.util
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import check_numbers
else:
    spec = importlib.util.spec_from_file_location(
        "check_numbers", ROOT / "scripts/check_numbers.py"
    )
    assert spec is not None
    assert spec.loader is not None
    check_numbers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_numbers)

NAMES = ["Toto-2.0-22m", "Toto 2.0", "Chronos-2", "Python 3.12", "9090"]
README = """# Promcast

Know what your Prometheus runs out of next, 41 minutes early.

```sh
uvx promcast scan --port 1234
```

## Try it

The board appears in 87 s. See [the run](https://example.com/runs/555).
Use `promcast scan --max 77` against localhost:9090 with Python 3.12.

## Limits

Untested above 5000 series.

## Evidence

On E3, Toto-2.0-22m and chronos-2-small caught 0.4167 of incidents, 2,000 resamples.

## Performance

A cycle takes 0.16 s and 12 s cold.
"""


def _problems(text: str, known: set[str], *, readme: bool = True) -> list[str]:
    lines = text.splitlines()
    selected = check_numbers.readme_lines(lines)[0] if readme else range(len(lines))
    return check_numbers.unsourced(Path("README.md"), lines, selected, NAMES, known)


def _tokens(problems: list[str]) -> list[str]:
    return [problem.split(": ", 2)[1].split(" ", 1)[0] for problem in problems]


def test_readme_mode_checks_hero_and_sections_outside_code_and_urls():
    problems = _problems(README, set())
    assert _tokens(problems) == ["41", "87", "0.4167", "2,000", "0.16", "12"]
    assert problems[0].startswith("README.md:3: 41 is in none of docs/evidence.md")


def test_sourced_numbers_pass_exactly_or_rounded():
    known = {"41", "87", "0.41666667", "2000", "0.16451283311471343", "12"}
    assert _problems(README, known) == []


def test_integers_are_never_matched_by_rounding():
    assert not check_numbers.is_sourced("41", {"40.6"})
    assert check_numbers.is_sourced("40.6", {"40.6"})
    assert check_numbers.is_sourced("0.2", {"0.16"})
    assert not check_numbers.is_sourced("0.2", {"0.26"})


def test_path_mode_checks_every_prose_line():
    assert _tokens(_problems(README, set(), readme=False)) == [
        "41",
        "87",
        "5000",
        "0.4167",
        "2,000",
        "0.16",
        "12",
    ]


@pytest.mark.parametrize(
    "line",
    [
        "Toto 2.0. Done.",
        "toto-2.0-22m beats chronos-2-small",
        "E3 and E4 on v3 at x86_64",
        "run `make check` at https://example.com/1/2",
    ],
)
def test_names_and_code_are_not_numbers(line: str):
    assert _problems(f"# T\n\n{line}\n", set()) == []


def test_source_numbers_read_every_existing_source(tmp_path: Path):
    (tmp_path / "bench.json").write_text(json.dumps({"median": 0.5, "rows": 1200}))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/evidence.md").write_text("| linear | 2,000 | 0.4167 |\n")
    assert check_numbers.source_numbers(tmp_path) == {"0.5", "1200", "2000", "0.4167"}


def test_missing_sections_are_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    readme = tmp_path / "README.md"
    readme.write_text("# Promcast\n\nA hero line.\n\n## Try it\n\nNothing numeric.\n")
    assert check_numbers.readme_lines(readme.read_text().splitlines())[1] == [
        "## Evidence",
        "## Performance",
    ]
    assert check_numbers.main([str(readme)]) == 0
    capsys.readouterr()


def test_main_fails_on_an_unsourced_number(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    page = tmp_path / "writeup.md"
    page.write_text("Promcast warned 987654 minutes early.\n")
    assert check_numbers.main([str(page)]) == 1
    assert "987654 is in none of" in capsys.readouterr().err


def test_committed_readme_passes(capsys: pytest.CaptureFixture[str]):
    assert check_numbers.main([]) == 0
    capsys.readouterr()


def test_named_readme_claim_cannot_borrow_an_unrelated_number():
    text = (
        "In E1, toto-2.0-4m caught 2 of 5 incidents and linear extrapolation caught 4, "
        "with no false alerts from either."
    )
    problems = check_numbers.named_claims(ROOT, Path("README.md"), text)
    assert any("toto-2.0-4m" in problem and "recall" in problem for problem in problems)
    assert any("linear" in problem and "recall" in problem for problem in problems)


def test_evidence_row_is_bound_to_its_method_and_field():
    original = (ROOT / "docs/evidence.md").read_text()
    changed = original.replace("| linear | 7 | 5 | 0.4000", "| linear | 7 | 5 | 0.6000")
    problems = check_numbers.named_claims(ROOT, Path("docs/evidence.md"), changed)
    assert any("linear.recall" in problem for problem in problems)
    assert check_numbers.named_claims(ROOT, Path("docs/evidence.md"), original) == []


def test_structured_field_changes_invalidate_named_claim(tmp_path: Path):
    path = tmp_path / "site/data"
    path.mkdir(parents=True)
    document = json.loads((ROOT / "site/data/evidence.json").read_text())
    document["corpora"]["e1"]["methods"]["toto-2.0-4m"]["incidents"] = 7
    (path / "evidence.json").write_text(json.dumps(document))
    text = (
        "In E1, toto-2.0-4m caught 4 of 5 incidents and linear extrapolation caught 2, "
        "with no false alerts from either."
    )
    assert check_numbers.named_claims(tmp_path, Path("README.md"), text)


@pytest.mark.parametrize(
    "claim",
    [
        "Every forecast is 100 percent accurate.",
        "Scan takes 99 seconds on every machine.",
        "It handles 3000 series.",
    ],
)
def test_matching_source_tokens_cannot_qualify_a_new_claim(tmp_path: Path, claim: str):
    page = tmp_path / "writeup.md"
    page.write_text(claim)
    assert check_numbers.main([str(page)]) == 1


def test_default_coverage_rejects_an_unbound_claim_outside_the_readme_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    shutil.copytree(ROOT / "site/data", tmp_path / "site/data")
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        (ROOT / "README.md").read_text() + "\n## Limits\nEvery forecast is 100 percent accurate.\n"
    )
    (tmp_path / "docs/evidence.md").write_text((ROOT / "docs/evidence.md").read_text())
    monkeypatch.setattr(check_numbers, "ROOT", tmp_path)
    assert check_numbers.main([]) == 1


def test_readme_interval_is_bound_to_each_named_method(tmp_path: Path):
    data = tmp_path / "site/data"
    data.mkdir(parents=True)
    document = json.loads((ROOT / "site/data/evidence.json").read_text())
    document["corpora"]["e1"]["methods"]["linear"]["recall_ci"] = [0.2, 1.0]
    (data / "evidence.json").write_text(json.dumps(document))
    text = (ROOT / "README.md").read_text()
    assert check_numbers.field_bound_numbers(tmp_path, Path("README.md"), text, NAMES)


def test_corpus_metadata_cannot_borrow_another_corpus_field(tmp_path: Path):
    data = tmp_path / "site/data"
    data.mkdir(parents=True)
    document = json.loads((ROOT / "site/data/evidence.json").read_text())
    document["corpora"]["e2"] = {**document["corpora"]["e1"], "n": 99}
    (data / "evidence.json").write_text(json.dumps(document))
    text = (ROOT / "docs/evidence.md").read_text().replace("- Series: 7", "- Series: 99")
    assert check_numbers.field_bound_numbers(tmp_path, Path("evidence.md"), text, NAMES)


def test_template_copy_requires_a_field_instead_of_a_numeric_literal(tmp_path: Path):
    path = tmp_path / "index.html"
    path.write_text(
        '<p>Every forecast is 100 percent accurate.</p><svg viewBox="0 0 100 100"></svg>'
    )
    assert check_numbers.main([str(path)]) == 1


def test_default_coverage_checks_site_template_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    shutil.copytree(ROOT / "site/data", tmp_path / "site/data")
    (tmp_path / "docs").mkdir(exist_ok=True)
    for name in ("README.md", "docs/evidence.md"):
        (tmp_path / name).write_text((ROOT / name).read_text())
    templates = tmp_path / "site/_templates"
    templates.mkdir()
    (templates / "index.html").write_text("<p>Every forecast is 100 percent accurate.</p>")
    monkeypatch.setattr(check_numbers, "ROOT", tmp_path)
    assert check_numbers.main([]) == 1


def test_template_geometry_does_not_count_as_a_public_claim(tmp_path: Path):
    path = tmp_path / "index.html"
    path.write_text(
        '<p>Read the board.</p><svg viewBox="0 0 100 100"><text x="20">123</text></svg>'
    )
    assert check_numbers.main([str(path)]) == 0
