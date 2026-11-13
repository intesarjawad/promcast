from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from json import dumps
from pathlib import Path
from statistics import median

import numpy as np
from pydantic import BaseModel, ConfigDict

from promcast.core.bootstrap import bootstrap_ci, decision_rule, paired_warning_difference
from promcast.core.forecaster import Forecaster
from promcast.core.operating import (
    MethodScore,
    OperatingPoint,
    choose_operating_point,
    operating_points,
    pool_scores,
)
from promcast.core.replay import Incident, PointScores, ReplaySpec, replay_sweep
from promcast.core.series import FloatArray

CORPORA = ("e1", "e2", "e3", "e4")
BASELINE_METHODS = ("linear", "seasonal-naive", "ets", "static-margin")
MODEL_METHODS = ("toto-2.0-22m", "toto-2.0-4m", "chronos-2-small")
METHODS = (*MODEL_METHODS, *BASELINE_METHODS)
DEFAULT_BUDGET_PER_WEEK = 1.0
# A headline cell moving further than this between two runs of the same frozen protocol
# signals a dependency or model regression rather than new evidence.
MAX_HEADLINE_DRIFT = 0.10
_VERDICT_MODEL = "toto-2.0-22m"
_VERDICT_BASELINE = "linear"
_ESTIMATE_SERIES = {"e1": 7, "e2": 50, "e3": 20, "e4": 0}
_ESTIMATE_STEPS = {"e1": 14 * 1440, "e2": 4096, "e3": 14 * 288, "e4": 14 * 288}
ESTIMATE_CONTEXT = {"e1": 1440, "e2": 1024, "e3": 576, "e4": 1152}
ESTIMATE_CYCLE = {"e1": 5, "e2": 16, "e3": 1, "e4": 1}


@dataclass(frozen=True)
class SeriesItem:
    """One regular timeline, its replay protocol and the caveats it carries."""

    corpus: str
    series_id: str
    values: FloatArray
    spec: ReplaySpec
    caveats: tuple[str, ...]
    labelled: tuple[Incident, ...] | None = None


@dataclass(frozen=True)
class SeriesRecord:
    """Poolable test scores for one series, cached independently of sharding."""

    corpus: str
    series_id: str
    fingerprint: str
    duration_s: float
    steps: int
    caveats: tuple[str, ...]
    scores: dict[str, MethodScore]
    labelled: dict[str, MethodScore] | None


@dataclass(frozen=True)
class SweepRecord:
    """Cached per-series scores at every candidate point; the corpus picks the point later."""

    corpus: str
    series_id: str
    fingerprint: str
    duration_s: float
    steps: int
    caveats: tuple[str, ...]
    sweeps: dict[str, list[PointScores]]


@dataclass(frozen=True)
class MethodSummary:
    """Pooled counts, exposure and cluster-bootstrap intervals for one method."""

    method: str
    n: int
    incidents: int
    recall: float
    precision: float
    false_alerts_per_week: float
    median_lead_s: float | None
    recall_ci: tuple[float, float] | None
    lead_ci: tuple[float, float] | None


@dataclass(frozen=True)
class CorpusEvidence:
    """One corpus's pooled methods, checksums and wall time."""

    corpus: str
    n: int
    wall_time_s: float
    dataset_sha256: tuple[str, ...]
    caveats: tuple[str, ...]
    methods: dict[str, MethodSummary]
    labelled_methods: dict[str, MethodSummary] | None


@dataclass(frozen=True)
class EvidenceReport:
    """The decision word and every corpus that produced records."""

    verdict: str
    budget_per_week: float
    corpora: tuple[CorpusEvidence, ...]


class _RecordDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus: str
    series_id: str
    fingerprint: str
    duration_s: float
    steps: int
    caveats: list[str]
    sweeps: dict[str, list[PointScores]]


def fingerprint(item: SeriesItem, methods: tuple[str, ...]) -> str:
    """Bind a cached record to the protocol, methods and timeline that produced it."""
    payload = {
        "methods": sorted(methods),
        "spec": asdict(item.spec),
        "steps": int(item.values.size),
        "labelled": item.labelled is not None,
        "record": 2,
    }
    return sha256(dumps(payload, sort_keys=True).encode()).hexdigest()


def record_path(cache_directory: Path, corpus: str, series_id: str) -> Path:
    """Keep series names filesystem-safe while leaving the identity in the record."""
    digest = sha256(series_id.encode()).hexdigest()[:16]
    return cache_directory / corpus / f"{digest}.json"


def record_text(record: SweepRecord) -> str:
    """Serialize one cache record without NaN so every reader sees identical JSON."""
    document = _RecordDocument(
        corpus=record.corpus,
        series_id=record.series_id,
        fingerprint=record.fingerprint,
        duration_s=record.duration_s,
        steps=record.steps,
        caveats=list(record.caveats),
        sweeps=record.sweeps,
    )
    return document.model_dump_json(indent=2) + "\n"


def parse_record(text: str) -> SweepRecord:
    """Validate a cache record before its scores reach a pooled table."""
    document = _RecordDocument.model_validate_json(text)
    return SweepRecord(
        document.corpus,
        document.series_id,
        document.fingerprint,
        document.duration_s,
        document.steps,
        tuple(document.caveats),
        dict(document.sweeps),
    )


def replay_item(
    item: SeriesItem, methods: tuple[str, ...], models: dict[str, Forecaster]
) -> dict[str, list[PointScores]]:
    """Score every method at every candidate point on one series."""
    horizon_s = item.spec.horizon_steps * item.spec.step_s
    points = [point for method in methods for point in operating_points(method, horizon_s)]
    forecasters = {name: model for name, model in models.items() if name in methods}
    return replay_sweep(
        item.values,
        item.spec,
        points,
        forecasters,
        labelled_incidents=None if item.labelled is None else list(item.labelled),
    )


def resolve_corpus(records: list[SweepRecord], budget: float) -> list[SeriesRecord]:
    """Choose one operating point per method from the pooled calibration, then score every series.

    Series without incidents still add their false alerts and series-weeks to the pool.
    """
    methods = sorted({name for record in records for name in record.sweeps})
    chosen: dict[str, OperatingPoint] = {}
    for name in methods:
        members = [record for record in records if name in record.sweeps]
        points = [entry.point for entry in members[0].sweeps[name]]
        for record in members[1:]:
            if [entry.point for entry in record.sweeps[name]] != points:
                raise ValueError(f"{name} candidate points differ between series in one corpus")
        pooled = {
            point: pool_scores([record.sweeps[name][index].calibration for record in members])
            for index, point in enumerate(points)
        }
        try:
            chosen[name] = choose_operating_point(pooled, budget)
        except ValueError as error:
            best = min(score.false_alerts_per_week for score in pooled.values())
            raise ValueError(
                f"no operating point for {name} meets the false-alert budget {budget:g} on the "
                f"pooled calibration split; the best pooled rate is {best:.4f} per series-week"
            ) from error
    resolved: list[SeriesRecord] = []
    for record in records:
        picked = {
            name: next(entry for entry in record.sweeps[name] if entry.point == chosen[name])
            for name in record.sweeps
        }
        labelled = {
            name: entry.labelled_test
            for name, entry in picked.items()
            if entry.labelled_test is not None
        }
        resolved.append(
            SeriesRecord(
                record.corpus,
                record.series_id,
                record.fingerprint,
                record.duration_s,
                record.steps,
                record.caveats,
                {name: entry.test for name, entry in picked.items()},
                labelled or None,
            )
        )
    return resolved


def _interval(clusters: list[FloatArray]) -> tuple[float, float] | None:
    if not clusters:
        return None
    lower, upper = bootstrap_ci(clusters, lambda sample: float(np.median(sample)))
    return round(lower, 8), round(upper, 8)


def summarize(method: str, scores: list[MethodScore]) -> MethodSummary:
    """Pool incident counts and series-weeks, then resample whole series clusters."""
    pooled = pool_scores(scores)
    recall_ci = None
    if pooled.n_incidents:
        recall_ci = _interval(
            [
                (np.array(cluster, dtype=np.float64) > 0).astype(np.float64)
                for cluster in pooled.series_warning_times_s
            ]
        )
    lead_clusters = [
        np.array([lead for lead in cluster if lead > 0], dtype=np.float64)
        for cluster in pooled.series_warning_times_s
    ]
    lead_ci = _interval([cluster for cluster in lead_clusters if cluster.size])
    return MethodSummary(
        method,
        pooled.n_series,
        pooled.n_incidents,
        round(pooled.recall, 8),
        round(pooled.precision, 8),
        round(pooled.false_alerts_per_week, 8),
        float(median(pooled.leads_s)) if pooled.leads_s else None,
        recall_ci,
        lead_ci,
    )


def _summaries(scores: dict[str, list[MethodScore]]) -> dict[str, MethodSummary]:
    return {name: summarize(name, values) for name, values in sorted(scores.items()) if values}


def corpus_evidence(
    corpus: str, records: list[SeriesRecord], dataset_sha256: tuple[str, ...]
) -> CorpusEvidence:
    """Pool each method across every series, including series with no incidents."""
    if not records:
        raise ValueError(f"corpus {corpus} has no series records")
    scores: dict[str, list[MethodScore]] = {}
    labelled: dict[str, list[MethodScore]] = {}
    for record in records:
        for name, score in record.scores.items():
            scores.setdefault(name, []).append(score)
        for name, score in (record.labelled or {}).items():
            labelled.setdefault(name, []).append(score)
    caveats = tuple(dict.fromkeys(caveat for record in records for caveat in record.caveats))
    return CorpusEvidence(
        corpus,
        len(records),
        round(sum(record.duration_s for record in records), 3),
        dataset_sha256,
        caveats,
        _summaries(scores),
        _summaries(labelled) or None,
    )


def decide_verdict(
    records: list[SeriesRecord],
    budget_per_week: float = DEFAULT_BUDGET_PER_WEEK,
    model: str = _VERDICT_MODEL,
) -> str:
    """Pair model and linear warning times, then apply the pre-registered rule."""
    differences: list[FloatArray] = []
    exposure: list[MethodScore] = []
    for record in records:
        candidate = record.scores.get(model)
        baseline = record.scores.get(_VERDICT_BASELINE)
        if candidate is None:
            continue
        exposure.append(candidate)
        if baseline is None or not candidate.warning_times_s:
            continue
        if len(candidate.warning_times_s) != len(baseline.warning_times_s):
            continue
        differences.append(
            paired_warning_difference(
                np.array(candidate.warning_times_s, dtype=np.float64),
                np.array(baseline.warning_times_s, dtype=np.float64),
            )
        )
    if not differences:
        return "ties"
    lower, upper = bootstrap_ci(differences, lambda sample: float(np.median(sample)))
    pooled = pool_scores(exposure)
    return decision_rule((lower, upper), pooled.false_alerts_per_week, budget_per_week)


def select_shard(items: list[SeriesItem], index: int, count: int) -> list[SeriesItem]:
    """Partition by position so every shard and the merge agree on order."""
    if count <= 0 or not 0 <= index < count:
        raise ValueError("shard must be an index inside a positive shard count")
    return [item for position, item in enumerate(items) if position % count == index]


def estimate_lines(
    methods: int,
    max_series: int,
    seconds_per_forecast: float,
    counts: Mapping[str, int] | None = None,
) -> str:
    """Estimate origins and CPU hours for every corpus without loading data.

    ``counts`` holds the series each corpus run would read. A corpus missing from it is
    estimated from the default cap and labelled as such.
    """
    if methods <= 0 or max_series < 0 or seconds_per_forecast <= 0:
        raise ValueError("methods and seconds must be positive; max_series nonnegative")
    measured = counts or {}
    lines: list[str] = []
    for corpus in CORPORA:
        if corpus in measured:
            series, basis = measured[corpus], ""
        else:
            series = (
                (max_series or _ESTIMATE_SERIES["e2"])
                if corpus == "e2"
                else _ESTIMATE_SERIES[corpus]
            )
            basis = " (default cap, not the corpus size)"
        origins = max(
            0,
            (_ESTIMATE_STEPS[corpus] - ESTIMATE_CONTEXT[corpus]) // ESTIMATE_CYCLE[corpus] + 1,
        )
        forecasts = series * origins * methods
        hours = forecasts * seconds_per_forecast / 3600
        lines.append(
            f"{corpus}: {series} series{basis}, {methods} methods, {forecasts} forecasts, "
            f"{hours:.1f} hours"
        )
    return "\n".join(lines) + "\n"


def _format_interval(interval: tuple[float, float] | None) -> str:
    return "unavailable" if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"


def _method_table(methods: dict[str, MethodSummary]) -> list[str]:
    lines = [
        "| Method | n | Incidents | Recall | Precision | False alerts/week | "
        "Median lead (s) | Recall 95% CI | Median lead 95% CI (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for method in methods.values():
        lead = "unavailable" if method.median_lead_s is None else f"{method.median_lead_s:.2f}"
        lines.append(
            f"| {method.method} | {method.n} | {method.incidents} | {method.recall:.4f} | "
            f"{method.precision:.4f} | {method.false_alerts_per_week:.4f} | {lead} | "
            f"{_format_interval(method.recall_ci)} | {_format_interval(method.lead_ci)} |"
        )
    return lines


def _corpus_section(corpus: CorpusEvidence) -> list[str]:
    lines = [
        f"## {corpus.corpus}",
        "",
        f"- Series: {corpus.n}",
        f"- Wall time: {corpus.wall_time_s:.3f} s",
        "- Dataset SHA-256:",
        *[f"  - {checksum}" for checksum in corpus.dataset_sha256],
        "- Caveats:",
        *[f"  - {caveat}" for caveat in corpus.caveats],
        "",
        "### Threshold incidents",
        "",
        *_method_table(corpus.methods),
        "",
    ]
    if corpus.corpus == "e3":
        lines.extend(["### Labelled anomalies", ""])
        if corpus.labelled_methods is None:
            lines.extend(["No labelled-anomaly scores were available.", ""])
        lines.extend([*_method_table(corpus.labelled_methods or {}), ""])
    return lines


def render_markdown(report: EvidenceReport, cpu_model: str, drift: tuple[str, ...] = ()) -> str:
    """Put the decision word first and keep each corpus's counts, caveats and digests.

    `drift` lists accepted headline cells that moved past the drift limit since the last report.
    """
    lines = [
        report.verdict,
        "",
        "# Evidence",
        "",
        f"Decision rule budget: {report.budget_per_week:.4f} false alerts per series-week.",
        f"CPU model: {cpu_model}.",
        "",
    ]
    if drift:
        lines.extend(
            [
                "## Headline drift",
                "",
                f"These cells moved more than {MAX_HEADLINE_DRIFT * 100:g} % relative to the "
                "previous report:",
                "",
                *[f"- {cell}" for cell in drift],
                "",
            ]
        )
    for corpus in report.corpora:
        lines.extend(_corpus_section(corpus))
    return "\n".join(lines)


def site_document(report: EvidenceReport) -> str:
    """Serialize the same pool without prose for the site to consume."""
    document = {
        "verdict": report.verdict,
        "budget_per_week": report.budget_per_week,
        "corpora": {corpus.corpus: asdict(corpus) for corpus in report.corpora},
    }
    return dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
