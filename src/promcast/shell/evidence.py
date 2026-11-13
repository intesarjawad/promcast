import logging
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, TypeAdapter

from promcast.core.evidence import (
    EvidenceReport,
    SeriesItem,
    SweepRecord,
    corpus_evidence,
    decide_verdict,
    fingerprint,
    parse_record,
    record_path,
    record_text,
    replay_item,
    resolve_corpus,
)
from promcast.core.forecaster import Forecaster
from promcast.core.replay import Incident
from promcast.core.scenarios import SCENARIOS
from promcast.shell.datasets import SUMS_FILENAME, lock_sums, read_lock, read_sums, verify_sha256
from promcast.shell.sources import (
    ReplaySource,
    boom,
    homelab,
    labelled_incidents,
    nab,
    synthetic,
)

LOGGER = logging.getLogger(__name__)
PROTOCOL_TAG = "protocol-v1"
PROTOCOL_DOCUMENT = "docs/protocol.md"
NAB_LOCK = "scripts/nab.lock"
_E2_CAVEAT = "Datadog data; Toto may have seen related series"
_E3_CAVEAT = "Labelled boundaries follow NAB point labels; Wu and Keogh document benchmark flaws."
_E4_CAVEAT = "Operator-supplied corpus; the repository does not verify its provenance."


class ProtocolError(ValueError):
    """The tagged protocol is missing or no longer matches the working tree."""


@dataclass(frozen=True)
class CacheOptions:
    """Where per-series results live and whether existing ones are reused."""

    directory: Path
    resume: bool = False


class _TaxonomyEntry(BaseModel):
    type: list[str]


def _source_item(
    corpus: str,
    series_id: str,
    source: ReplaySource,
    labelled: tuple[Incident, ...] | None,
    extra_caveats: tuple[str, ...],
) -> SeriesItem:
    return SeriesItem(
        corpus,
        series_id,
        source.values,
        source.spec,
        (*source.caveats, *extra_caveats),
        labelled,
    )


def _synthetic_items() -> list[SeriesItem]:
    return [_source_item("e1", scenario, synthetic(scenario), None, ()) for scenario in SCENARIOS]


def _boom_targets(data_directory: Path, max_series: int) -> list[str]:
    folder = data_directory / "BOOM"
    taxonomy_path = folder / "dataset_taxonomy.json"
    if not taxonomy_path.is_file():
        raise FileNotFoundError(
            f"BOOM data is missing from {folder}; run scripts/fetch_boom.py first"
        )
    taxonomy = TypeAdapter(dict[str, _TaxonomyEntry]).validate_json(
        taxonomy_path.read_text(encoding="utf-8")
    )
    targets = sorted(name for name, entry in taxonomy.items() if entry.type == ["gauge"])
    return targets if max_series == 0 else targets[:max_series]


def _boom_items(data_directory: Path, max_series: int) -> list[SeriesItem]:
    folder = data_directory / "BOOM"
    return [
        _source_item("e2", target, boom(folder, target), None, (_E2_CAVEAT,))
        for target in _boom_targets(data_directory, max_series)
    ]


def _corpus_root(base: Path) -> Path | None:
    if (base / "data").is_dir():
        return base
    if not base.is_dir():
        return None
    candidates = [path for path in base.iterdir() if (path / "data").is_dir()]
    return candidates[0] if len(candidates) == 1 else None


def _labels(path: Path, key: str, start_ts: float) -> tuple[Incident, ...] | None:
    if not path.is_file():
        return None
    try:
        return tuple(labelled_incidents(path, key, start_ts))
    except ValueError as error:
        if "exactly one" in str(error):
            return None
        raise


def _csv_items(
    corpus: str,
    root: Path,
    paths: list[Path],
    extra_caveats: tuple[str, ...],
    load: Callable[[Path], ReplaySource],
) -> list[SeriesItem]:
    data = root / "data"
    labels = root / "labels" / "combined_labels.json"
    items: list[SeriesItem] = []
    for path in paths:
        relative = path.relative_to(data).as_posix()
        source = load(path)
        items.append(
            _source_item(
                corpus,
                relative,
                source,
                _labels(labels, relative, source.start_ts),
                extra_caveats,
            )
        )
    return items


def _nab_manifest(data_directory: Path) -> tuple[Path, list[list[str]]]:
    root = _corpus_root(data_directory / "NAB")
    if root is None:
        raise FileNotFoundError(
            f"NAB data is missing from {data_directory / 'NAB'}; run scripts/fetch_nab.py first"
        )
    manifest = [line.split("  ", 1) for line in read_sums(root)]
    if not manifest:
        raise FileNotFoundError(f"{root / SUMS_FILENAME} is missing; run scripts/fetch_nab.py")
    for _, relative in manifest:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"{root / relative} is missing; run scripts/fetch_nab.py")
    return root, manifest


def _nab_csv_paths(root: Path, manifest: list[list[str]], max_series: int) -> list[Path]:
    paths = [root / relative for _, relative in manifest if relative.endswith(".csv")]
    return paths[: max_series or None]


def _nab_items(data_directory: Path, max_series: int) -> list[SeriesItem]:
    root, manifest = _nab_manifest(data_directory)
    for digest, relative in manifest:
        verify_sha256(root / relative, digest)
    return _csv_items("e3", root, _nab_csv_paths(root, manifest, max_series), (_E3_CAVEAT,), nab)


def _e4_paths(root: Path, max_series: int) -> list[Path]:
    return sorted((root / "data").rglob("*.csv"))[: max_series or None]


def _e4_items(data_directory: Path, max_series: int) -> list[SeriesItem]:
    root = _corpus_root(data_directory / "e4")
    if root is None:
        return []
    return _csv_items("e4", root, _e4_paths(root, max_series), (_E4_CAVEAT,), homelab)


def corpus_counts(data_directory: Path, max_series: int) -> dict[str, int]:
    """Count the series each run would read, from the listings the run uses.

    Reads no series values. A corpus whose data is missing or unreadable is left out.
    """

    def nab_count() -> int:
        root, manifest = _nab_manifest(data_directory)
        return len(_nab_csv_paths(root, manifest, max_series))

    def e4_count() -> int:
        root = _corpus_root(data_directory / "e4")
        return 0 if root is None else len(_e4_paths(root, max_series))

    listings: dict[str, Callable[[], int]] = {
        "e1": lambda: len(SCENARIOS),
        "e2": lambda: len(_boom_targets(data_directory, max_series)),
        "e3": nab_count,
        "e4": e4_count,
    }
    counts: dict[str, int] = {}
    for corpus, count in listings.items():
        try:
            counts[corpus] = count()
        except (OSError, ValueError):
            continue
    return counts


def verify_nab_lock(root: Path, data_directory: Path, lock: str = NAB_LOCK) -> None:
    """Refuse an E3 corpus whose manifest differs from the per-file lock in the repository."""
    corpus = _corpus_root(data_directory / "NAB")
    if corpus is None:
        raise FileNotFoundError(
            f"NAB data is missing from {data_directory / 'NAB'}; run scripts/fetch_nab.py first"
        )
    if read_sums(corpus) != lock_sums(read_lock(root / lock)):
        raise ValueError(f"NAB files in {corpus} differ from {lock}; rerun scripts/fetch_nab.py")


def series_items(corpus: str, data_directory: Path, max_series: int) -> list[SeriesItem]:
    """Build a corpus in deterministic order so shards and merges agree."""
    if corpus == "e1":
        return _synthetic_items()
    if corpus == "e2":
        return _boom_items(data_directory, max_series)
    if corpus == "e3":
        return _nab_items(data_directory, max_series)
    if corpus == "e4":
        return _e4_items(data_directory, max_series)
    raise ValueError(f"unknown corpus: {corpus}")


def dataset_sums(corpus: str, data_directory: Path) -> tuple[str, ...]:
    """Read the checksum manifest written by the fetch script for this corpus."""
    if corpus == "e1":
        return ("synthetic; no dataset files",)
    if corpus == "e2":
        root = data_directory / "BOOM"
    elif corpus == "e4":
        root = _corpus_root(data_directory / "e4")
    else:
        root = _corpus_root(data_directory / "NAB")
    if root is None:
        return ("checksum manifest missing",)
    return read_sums(root) or ("checksum manifest missing",)


def build_report(
    records: list[SweepRecord], corpora: Sequence[str], budget: float, data_directory: Path
) -> EvidenceReport:
    """Pool by corpus only after every selected series has a cached result."""
    resolved = {
        corpus: resolve_corpus([record for record in records if record.corpus == corpus], budget)
        for corpus in corpora
        if any(record.corpus == corpus for record in records)
    }
    sections = [
        corpus_evidence(corpus, members, dataset_sums(corpus, data_directory))
        for corpus, members in resolved.items()
    ]
    pooled = [record for members in resolved.values() for record in members]
    return EvidenceReport(decide_verdict(pooled, budget), budget, tuple(sections))


def read_cached(path: Path, item: SeriesItem, methods: tuple[str, ...]) -> SweepRecord | None:
    """Return a record only when it was produced by this exact cache key."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    record = parse_record(text)
    if (
        record.corpus != item.corpus
        or record.series_id != item.series_id
        or record.fingerprint != fingerprint(item, methods)
    ):
        return None
    return record


def _write_record(path: Path, record: SweepRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(record_text(record), encoding="utf-8")
    temporary.replace(path)


def run_items(
    items: list[SeriesItem],
    methods: tuple[str, ...],
    models: Mapping[str, Forecaster],
    cache: CacheOptions,
    timer: Callable[[], float] = perf_counter,
) -> list[SweepRecord]:
    """Compute missing series and cache each result independently of the shard."""
    records: list[SweepRecord] = []
    for item in items:
        path = record_path(cache.directory, item.corpus, item.series_id)
        if cache.resume:
            try:
                cached = read_cached(path, item, methods)
            except ValueError as error:
                LOGGER.warning("Recomputing stale cache %s: %s", path, error)
                cached = None
            if cached is not None:
                records.append(cached)
                continue
        started = timer()
        sweeps = replay_item(item, methods, dict(models))
        record = SweepRecord(
            item.corpus,
            item.series_id,
            fingerprint(item, methods),
            timer() - started,
            int(item.values.size),
            item.caveats,
            sweeps,
        )
        _write_record(path, record)
        records.append(record)
    return records


def load_cached(
    items: list[SeriesItem], methods: tuple[str, ...], cache_directory: Path
) -> list[SweepRecord]:
    """Merge shards by reading every record they wrote."""
    records: list[SweepRecord] = []
    for item in items:
        path = record_path(cache_directory, item.corpus, item.series_id)
        record = read_cached(path, item, methods)
        if record is None:
            raise FileNotFoundError(
                f"no cached result for {item.corpus} {item.series_id}; finish all shards first"
            )
        records.append(record)
    return records


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("git")
    if executable is None:
        raise ProtocolError("git is required to check the tagged protocol")
    # The executable is absolute and every argument comes from a fixed subcommand.
    return subprocess.run(  # noqa: S603
        [executable, "-C", str(root), *arguments], capture_output=True, text=True, check=False
    )


def verify_protocol(root: Path, tag: str = PROTOCOL_TAG, document: str = PROTOCOL_DOCUMENT) -> None:
    """Refuse to write evidence when the protocol differs from its pinned revision."""
    resolved = _git(root, "rev-parse", "--verify", f"{tag}^{{commit}}")
    if resolved.returncode != 0:
        raise ProtocolError(
            f"tag {tag} is missing; commit {document} and push the tag before running evidence"
        )
    if not (root / document).is_file():
        raise ProtocolError(f"{document} is missing from {root}")
    difference = _git(root, "diff", "--quiet", tag, "--", document)
    if difference.returncode == 1:
        raise ProtocolError(f"{document} changed since {tag}; restore it before running evidence")
    if difference.returncode != 0:
        raise ProtocolError(
            f"git could not compare {document} with {tag}: {difference.stderr.strip()}"
        )


__all__ = [
    "CacheOptions",
    "ProtocolError",
    "build_report",
    "dataset_sums",
    "load_cached",
    "read_cached",
    "run_items",
    "series_items",
    "verify_nab_lock",
    "verify_protocol",
]
