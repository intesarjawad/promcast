"""Compare a new evidence report's headline cells with the previously published ones."""

from pydantic import BaseModel

from promcast.core.evidence import MAX_HEADLINE_DRIFT, EvidenceReport, MethodSummary

HEADLINE_FIELDS = ("recall", "precision", "false_alerts_per_week", "median_lead_s")


class _PublishedMethod(BaseModel):
    recall: float
    precision: float
    false_alerts_per_week: float
    median_lead_s: float | None


class _PublishedCorpus(BaseModel):
    methods: dict[str, _PublishedMethod]


class _PublishedReport(BaseModel):
    corpora: dict[str, _PublishedCorpus]


def _moved(previous: float | None, current: float | None, tolerance: float) -> bool:
    if previous is None or current is None:
        return previous != current
    if previous == 0:
        return current != 0
    return abs(current - previous) / abs(previous) > tolerance


def _cell(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _change(previous: float | None, current: float | None) -> str:
    if previous is None or current is None or previous == 0:
        return "new value"
    return f"{(current - previous) / abs(previous):+.1%}"


def _describe(corpus: str, summary: MethodSummary, previous: _PublishedMethod, field: str) -> str:
    before: float | None = getattr(previous, field)
    after: float | None = getattr(summary, field)
    return (
        f"{corpus} {summary.method} {field}: {_cell(before)} -> {_cell(after)} "
        f"({_change(before, after)})"
    )


def headline_drift(
    published_json: str, report: EvidenceReport, tolerance: float = MAX_HEADLINE_DRIFT
) -> tuple[str, ...]:
    """Describe every threshold-incident cell that moved more than `tolerance` relative.

    Only cells present in both reports are compared, so adding a corpus or method is not drift.
    """
    published = _PublishedReport.model_validate_json(published_json)
    moved: list[str] = []
    for corpus in report.corpora:
        before = published.corpora.get(corpus.corpus)
        if before is None:
            continue
        for name, summary in corpus.methods.items():
            previous_method = before.methods.get(name)
            if previous_method is None:
                continue
            moved.extend(
                _describe(corpus.corpus, summary, previous_method, field)
                for field in HEADLINE_FIELDS
                if _moved(getattr(previous_method, field), getattr(summary, field), tolerance)
            )
    return tuple(moved)


__all__ = ["HEADLINE_FIELDS", "headline_drift"]
