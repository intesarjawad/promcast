"""Format public claims from their named evidence fields."""

import re
from html.parser import HTMLParser
from pathlib import Path

from pydantic import BaseModel


class _Method(BaseModel):
    incidents: int
    recall: float
    false_alerts_per_week: float
    recall_ci: tuple[float, float] | None


class _Corpus(BaseModel):
    n: int
    wall_time_s: float
    caveats: list[str]
    methods: dict[str, _Method]


class _Evidence(BaseModel):
    budget_per_week: float
    corpora: dict[str, _Corpus]


def readme_evidence(evidence: _Evidence) -> str:
    """Keep each headline count and interval tied to its corpus and method."""
    model = "toto-2.0-4m"
    toto = evidence.corpora["e1"].methods[model]
    linear = evidence.corpora["e1"].methods["linear"]
    if toto.incidents != linear.incidents:
        raise ValueError("E1 headline methods need the same incident population")
    alerts = (
        "with no false alerts from either"
        if toto.false_alerts_per_week == linear.false_alerts_per_week == 0
        else f"with {model} at {toto.false_alerts_per_week:g} and linear at "
        f"{linear.false_alerts_per_week:g} false alerts per series-week"
    )
    intervals = "; ".join(
        f"{name} {score.recall_ci[0]:g} to {score.recall_ci[1]:g}"
        if score.recall_ci
        else f"{name} unavailable"
        for name, score in ((model, toto), ("linear", linear))
    )
    return (
        f"In the synthetic E1 run, {model} caught {round(toto.recall * toto.incidents)} "
        f"of {toto.incidents} incidents and linear extrapolation caught "
        f"{round(linear.recall * linear.incidents)}, {alerts}. "
        f"{toto.incidents} incidents is a smoke test, not an accuracy result. "
        f"Recall intervals: {intervals}. The full table is in [evidence](docs/evidence.md)."
    )


class _PublicHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ignored = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"head", "svg", "script", "style", "code"}:
            self.ignored += 1
        elif not self.ignored:
            self.parts.extend(
                value
                for name, value in attrs
                if name in {"alt", "title", "aria-label"} and value is not None
            )

    def handle_endtag(self, tag: str) -> None:
        if tag in {"head", "svg", "script", "style", "code"}:
            self.ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored:
            self.parts.append(data)


def template_prose(text: str) -> str:
    """Check visible copy separately from SVG geometry, scripts and commands."""
    parser = _PublicHTML()
    parser.feed(text)
    return " ".join(parser.parts).replace("SIL Open Font License 1.1", " ")


def bound_text(root: Path, path: Path, text: str) -> str:
    """Remove only full, field-qualified claims before searching for unbound numbers."""
    if path.suffix == ".html":
        return template_prose(text)
    evidence = _Evidence.model_validate_json((root / "site/data/evidence.json").read_text())
    phrases: list[str] = []
    if path.name == "README.md":
        phrases.append(readme_evidence(evidence))
    if path.name == "evidence.md":
        phrases.append(
            f"Decision rule budget: {evidence.budget_per_week:.4f} false alerts per series-week."
        )
        sections = re.split(r"(?m)(^## [^\n]+$)", text)
        for index in range(1, len(sections), 2):
            key = sections[index][3:].strip().lower()
            if key not in evidence.corpora:
                continue
            corpus = evidence.corpora[key]
            for phrase in [
                f"- Series: {corpus.n}",
                f"- Wall time: {corpus.wall_time_s:.3f} s",
                *("  - " + caveat for caveat in corpus.caveats),
            ]:
                sections[index + 1] = re.sub(
                    r"(?m)^" + re.escape(phrase) + r"$", "", sections[index + 1]
                )
        text = "".join(sections)
        text = re.sub(r"^\| .*\|$", "", text, flags=re.M)
        text = re.sub(r"^CPU model: [^\n]+$", "", text, flags=re.M)
    for phrase in phrases:
        text = re.sub(r"\s+".join(re.escape(part) for part in phrase.split()), " ", text)
    return text


def write_readme(root: Path) -> None:
    """Regenerate the evidence paragraph without running inference."""
    evidence = _Evidence.model_validate_json((root / "site/data/evidence.json").read_text())
    path = root / "README.md"
    text, count = re.subn(
        r"In the synthetic E1 run,.*?(?=\n\n)",
        readme_evidence(evidence),
        path.read_text(),
        flags=re.S,
    )
    if count != 1:
        raise ValueError("README needs one E1 evidence paragraph")
    path.write_text(text, encoding="utf-8")
