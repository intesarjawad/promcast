"""Run the evidence pipeline and write its report and site data."""

import argparse
import logging
import sys
from pathlib import Path

from promcast.core.drift import headline_drift
from promcast.core.evidence import (
    BASELINE_METHODS,
    CORPORA,
    MAX_HEADLINE_DRIFT,
    MODEL_METHODS,
    EvidenceReport,
    estimate_lines,
    render_markdown,
    select_shard,
    site_document,
)
from promcast.core.forecaster import Forecaster
from promcast.shell.bench import cpu_model
from promcast.shell.cli import load_forecaster
from promcast.shell.evidence import (
    CacheOptions,
    build_report,
    corpus_counts,
    load_cached,
    run_items,
    series_items,
    verify_nab_lock,
    verify_protocol,
)

ROOT = Path(__file__).resolve().parents[1]


def _shard(text: str) -> tuple[int, int]:
    index, _, count = text.partition("/")
    try:
        parsed = (int(index), int(count))
    except ValueError:
        raise argparse.ArgumentTypeError("shard must look like INDEX/COUNT") from None
    if parsed[1] <= 0 or not 0 <= parsed[0] < parsed[1]:
        raise argparse.ArgumentTypeError("shard must satisfy 0 <= INDEX < COUNT")
    return parsed


def _models(text: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in text.split(",") if name.strip())
    unknown = sorted(set(names) - set(MODEL_METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown models: {', '.join(unknown)}")
    return names


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", choices=CORPORA)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache/promcast/evidence")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".cache/promcast/evidence/cache"
    )
    parser.add_argument("--model-dir", type=Path, default=Path("~/.cache/promcast/models"))
    parser.add_argument("--models", type=_models, default=MODEL_METHODS)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--max-series", type=int, default=0)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--shard", type=_shard)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--seconds-per-forecast", type=float, default=0.05)
    parser.add_argument(
        "--accept-drift",
        action="store_true",
        help="publish headline cells that moved past the drift limit; the report lists them",
    )
    return parser


def _forecasters(names: tuple[str, ...], directory: Path) -> dict[str, Forecaster]:
    return {name: load_forecaster(directory.expanduser() / name, name) for name in names}


class HeadlineDriftError(ValueError):
    """Headline cells moved past the drift limit and nobody accepted the change."""


def write_report(report: EvidenceReport, out: Path, cpu: str, *, accept_drift: bool) -> None:
    """Write the report and its site data unless headline cells drifted unaccepted."""
    document = out / "docs/evidence.md"
    site = out / "site/data/evidence.json"
    drift = headline_drift(site.read_text(encoding="utf-8"), report) if site.is_file() else ()
    if drift and not accept_drift:
        cells = "\n".join(f"  {cell}" for cell in drift)
        raise HeadlineDriftError(
            f"headline cells moved more than {MAX_HEADLINE_DRIFT * 100:g} % since {site}:\n"
            f"{cells}\nNothing was written. Find the cause, such as a dependency or model "
            "change, then rerun with --accept-drift to publish the report with these cells listed."
        )
    document.parent.mkdir(parents=True, exist_ok=True)
    site.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(render_markdown(report, cpu, drift), encoding="utf-8")
    site.write_text(site_document(report), encoding="utf-8")
    sys.stdout.write(f"wrote {document} and {site}\n")


def main() -> int:
    """Run the selected corpora, or print the workload estimate."""
    arguments = _parser().parse_args()
    methods = (*arguments.models, *BASELINE_METHODS)
    if arguments.estimate:
        sys.stdout.write(
            estimate_lines(
                len(methods),
                arguments.max_series,
                arguments.seconds_per_forecast,
                corpus_counts(arguments.data_dir, arguments.max_series),
            )
        )
        return 0
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        verify_protocol(ROOT)
        corpora = arguments.corpus or list(CORPORA)
        if "e3" in corpora:
            verify_nab_lock(ROOT, arguments.data_dir)
        items = [
            item
            for corpus in corpora
            for item in series_items(corpus, arguments.data_dir, arguments.max_series)
        ]
        if arguments.shard is not None:
            items = select_shard(items, *arguments.shard)
        if arguments.merge:
            records = load_cached(items, methods, arguments.cache_dir)
        else:
            records = run_items(
                items,
                methods,
                _forecasters(arguments.models, arguments.model_dir),
                CacheOptions(arguments.cache_dir, arguments.resume),
            )
            if arguments.shard is not None:
                sys.stdout.write(f"cached {len(records)} series\n")
                return 0
        report = build_report(records, corpora, arguments.budget, arguments.data_dir)
        write_report(report, arguments.out, cpu_model(), accept_drift=arguments.accept_drift)
    except (OSError, ValueError, ImportError) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
