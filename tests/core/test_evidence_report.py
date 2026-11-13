import json

import numpy as np
import pytest

from promcast.core.evidence import (
    CorpusEvidence,
    EvidenceReport,
    MethodSummary,
    SeriesItem,
    SeriesRecord,
    SweepRecord,
    corpus_evidence,
    decide_verdict,
    estimate_lines,
    render_markdown,
    resolve_corpus,
    select_shard,
    site_document,
    summarize,
)
from promcast.core.operating import MethodScore, OperatingPoint
from promcast.core.replay import PointScores, ReplaySpec

MODEL = "toto-2.0-22m"
E2_CAVEAT = "Datadog data; Toto may have seen related series"
E3_CAVEAT = "Labelled boundaries follow NAB point labels; Wu and Keogh document benchmark flaws."


def _score(warning_s: float, weeks: float = 1.0) -> MethodScore:
    caught = warning_s > 0
    return MethodScore(
        n_incidents=1,
        caught_early=1 if caught else 0,
        late=0 if caught else 1,
        n_episodes=1,
        matched=1,
        weeks=weeks,
        leads_s=[warning_s] if caught else [],
        reset_s=[],
        warning_times_s=[warning_s],
        n_series=1,
        series_warning_times_s=[[warning_s]],
    )


def _record(series_id: str, model_warning: float, linear_warning: float) -> SeriesRecord:
    return SeriesRecord(
        "e1",
        series_id,
        "fingerprint",
        1.0,
        12,
        ("synthetic test",),
        {MODEL: _score(model_warning), "linear": _score(linear_warning)},
        None,
    )


def _summary(name: str = "linear") -> MethodSummary:
    return MethodSummary(name, 2, 1, 1.0, 0.5, 1.0, 60.0, (1.0, 1.0), (60.0, 60.0))


def test_pooling_keeps_zero_incident_series_in_the_exposure_denominator():
    first = _score(60.0)
    second = MethodScore(
        n_incidents=0,
        caught_early=0,
        late=0,
        n_episodes=1,
        matched=0,
        weeks=1.0,
        series_warning_times_s=[[]],
    )
    summary = summarize("linear", [first, second])
    assert summary.n == 2
    assert summary.incidents == 1
    assert summary.recall == 1.0
    assert summary.precision == 0.5
    assert summary.false_alerts_per_week == 0.5
    assert summary.median_lead_s == 60.0
    assert summary.recall_ci == (1.0, 1.0)
    assert summary.lead_ci == (60.0, 60.0)


def test_corpus_evidence_sums_wall_time_and_deduplicates_caveats():
    records = [_record("a", 120, 60), _record("b", 120, 60)]
    records[1] = SeriesRecord(
        records[1].corpus,
        records[1].series_id,
        records[1].fingerprint,
        2.5,
        records[1].steps,
        ("synthetic test", "extra"),
        records[1].scores,
        records[1].labelled,
    )
    corpus = corpus_evidence("e1", records, ("abc  file",))
    assert corpus.n == 2
    assert corpus.wall_time_s == 3.5
    assert corpus.caveats == ("synthetic test", "extra")
    assert corpus.methods["linear"].n == 2
    assert corpus.labelled_methods is None
    with pytest.raises(ValueError, match="no series records"):
        corpus_evidence("e1", [], ())


def test_verdict_pairs_model_and_linear_warning_times():
    assert decide_verdict([_record("a", 120, 60), _record("b", 180, 60)]) == "beats"
    assert decide_verdict([_record("a", 0, 60)]) == "loses"
    assert decide_verdict([_record("a", 60, 60)]) == "ties"
    assert decide_verdict([]) == "ties"


def test_shards_partition_every_series_once_in_order():
    spec = ReplaySpec("below", 4, 4, 60, 1, 8, threshold=0)
    items = [SeriesItem("e1", str(index), np.zeros(1), spec, ()) for index in range(5)]
    first = select_shard(items, 0, 2)
    second = select_shard(items, 1, 2)
    assert [item.series_id for item in first] == ["0", "2", "4"]
    assert [item.series_id for item in second] == ["1", "3"]
    with pytest.raises(ValueError, match="shard"):
        select_shard(items, 2, 2)


def test_estimate_reports_every_corpus():
    lines = estimate_lines(7, 50, 0.05).splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("e1: 7 series")
    assert lines[1].startswith("e2: 50 series")
    assert lines[2].startswith("e3: 20 series")
    assert lines[3].startswith("e4: 0 series")
    assert all("forecasts" in line and "hours" in line for line in lines)
    with pytest.raises(ValueError, match="positive"):
        estimate_lines(0, 50, 0.05)


def test_estimate_uses_measured_counts_without_the_default_label():
    lines = estimate_lines(7, 0, 0.05, {"e1": 7, "e2": 1296, "e3": 20}).splitlines()
    assert lines[1].startswith("e2: 1296 series, 7 methods")
    assert "default cap" not in lines[1]
    assert lines[3].startswith("e4: 0 series (default cap, not the corpus size), 7 methods")


def test_estimate_labels_every_default_count_as_a_cap():
    lines = estimate_lines(7, 0, 0.05, {"e1": 7}).splitlines()
    assert lines[1].startswith("e2: 50 series (default cap, not the corpus size), 7 methods")
    assert lines[2].startswith("e3: 20 series (default cap, not the corpus size), 7 methods")


def test_markdown_puts_the_verdict_first_and_shows_both_e3_tables():
    e2 = CorpusEvidence(
        "e2",
        2,
        1.5,
        ("abc  BOOM/data/a.arrow",),
        (E2_CAVEAT,),
        {"linear": _summary()},
        None,
    )
    e3 = CorpusEvidence(
        "e3",
        3,
        2.5,
        ("def  data/a.csv",),
        (E3_CAVEAT,),
        {"linear": _summary()},
        {"linear": _summary()},
    )
    report = EvidenceReport("beats", 1.0, (e2, e3))
    text = render_markdown(report, "Test CPU")
    assert text.splitlines()[0] == "beats"
    assert text.count(E2_CAVEAT) == 1
    assert "Test CPU" in text
    assert "abc  BOOM/data/a.arrow" in text
    section = text.split("## e3", 1)[1]
    assert "### Threshold incidents" in section
    assert "### Labelled anomalies" in section
    assert "Wu and Keogh" in section


def test_site_document_carries_the_same_pool_as_json():
    corpus = CorpusEvidence(
        "e3",
        3,
        2.5,
        ("def  data/a.csv",),
        (E3_CAVEAT,),
        {"linear": _summary()},
        {"linear": _summary()},
    )
    document = json.loads(site_document(EvidenceReport("ties", 1.0, (corpus,))))
    assert document["verdict"] == "ties"
    assert document["corpora"]["e3"]["n"] == 3
    assert document["corpora"]["e3"]["methods"]["linear"]["incidents"] == 1
    assert document["corpora"]["e3"]["labelled_methods"]["linear"]["n"] == 2


def _sweep(series_id: str, false_alerts: tuple[int, int], incident: bool) -> SweepRecord:
    """Two candidate points; each series sees one week, so counts are rates per series-week."""
    entries = [
        PointScores(
            OperatingPoint("linear", warn),
            MethodScore(int(incident), int(incident and warn == 600), 0, alerts, 0, 1),
            _score(warn),
        )
        for warn, alerts in zip((300, 600), false_alerts, strict=True)
    ]
    return SweepRecord("e2", series_id, "f", 1.0, 12, (), {"linear": entries})


def test_pooled_selection_meets_a_budget_that_one_series_alone_would_miss():
    noisy = _sweep("noisy", (3, 3), incident=True)
    quiet = [_sweep(f"quiet{i}", (0, 0), incident=False) for i in range(3)]
    resolved = resolve_corpus([noisy, *quiet], budget=1.0)
    assert [record.scores["linear"].warning_times_s for record in resolved] == [[600]] * 4


def test_every_series_is_scored_at_the_one_pooled_point():
    records = [_sweep("a", (0, 0), incident=True), _sweep("b", (0, 0), incident=False)]
    resolved = resolve_corpus(records, budget=1.0)
    assert {record.scores["linear"].warning_times_s[0] for record in resolved} == {600}


def test_corpus_over_budget_even_when_pooled_names_method_and_best_rate():
    records = [_sweep("a", (6, 4), incident=True), _sweep("b", (2, 4), incident=False)]
    with pytest.raises(ValueError, match=r"linear.*budget 1.*best pooled rate is 4\.0000"):
        resolve_corpus(records, budget=1.0)
