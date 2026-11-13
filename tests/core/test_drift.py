import pytest
from pydantic import ValidationError

from promcast.core.drift import headline_drift
from promcast.core.evidence import CorpusEvidence, EvidenceReport, MethodSummary, site_document


def _summary(
    name: str = "linear",
    recall: float = 0.5,
    precision: float = 0.5,
    false_alerts: float = 1.0,
    lead_s: float | None = 600.0,
) -> MethodSummary:
    return MethodSummary(name, 4, 8, recall, precision, false_alerts, lead_s, None, None)


def _report(*summaries: MethodSummary, corpus: str = "e3") -> EvidenceReport:
    methods = {summary.method: summary for summary in summaries}
    return EvidenceReport("ties", 1.0, (CorpusEvidence(corpus, 4, 1.0, (), (), methods, None),))


def _published(*summaries: MethodSummary, corpus: str = "e3") -> str:
    return site_document(_report(*summaries, corpus=corpus))


def test_identical_report_has_no_drift():
    assert headline_drift(_published(_summary()), _report(_summary())) == ()


def test_moves_within_ten_percent_are_not_drift():
    current = _summary(recall=0.54, precision=0.46, false_alerts=1.09, lead_s=660.0)
    assert headline_drift(_published(_summary()), _report(current)) == ()


def test_each_cell_past_ten_percent_is_described():
    current = _summary(recall=0.4, lead_s=900.0)
    assert headline_drift(_published(_summary()), _report(current)) == (
        "e3 linear recall: 0.5000 -> 0.4000 (-20.0%)",
        "e3 linear median_lead_s: 600.0000 -> 900.0000 (+50.0%)",
    )


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        (0.0, 0.0, ()),
        (0.0, 0.1, ("e3 linear false_alerts_per_week: 0.0000 -> 0.1000 (new value)",)),
    ],
)
def test_change_from_zero_is_drift(previous: float, current: float, expected: tuple[str, ...]):
    published = _published(_summary(false_alerts=previous))
    assert headline_drift(published, _report(_summary(false_alerts=current))) == expected


def test_lead_appearing_or_disappearing_is_drift():
    published = _published(_summary(lead_s=None))
    assert headline_drift(published, _report(_summary(lead_s=None))) == ()
    assert headline_drift(published, _report(_summary(lead_s=60.0))) == (
        "e3 linear median_lead_s: unavailable -> 60.0000 (new value)",
    )
    assert headline_drift(_published(_summary()), _report(_summary(lead_s=None))) == (
        "e3 linear median_lead_s: 600.0000 -> unavailable (new value)",
    )


def test_new_corpus_or_method_is_not_drift():
    published = _published(_summary())
    assert headline_drift(published, _report(_summary(recall=0.1), corpus="e4")) == ()
    current = _report(_summary(), _summary("toto-2.0-22m", recall=0.9))
    assert headline_drift(published, current) == ()


def test_custom_tolerance_applies():
    assert headline_drift(_published(_summary()), _report(_summary(recall=0.52)), 0.01) == (
        "e3 linear recall: 0.5000 -> 0.5200 (+4.0%)",
    )


def test_malformed_published_report_is_rejected():
    with pytest.raises(ValidationError):
        headline_drift('{"corpora": {"e3": {}}}', _report(_summary()))
