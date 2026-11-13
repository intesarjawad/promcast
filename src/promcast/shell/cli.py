from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, runtime_checkable

from promcast.core.config import (
    Config,
    ConfigError,
    ModelConfig,
    cache_dir_override,
    parse_config,
    parse_fragment,
)
from promcast.core.diagnostics import safe_urls
from promcast.core.operating import operating_points
from promcast.core.replay import ReplayResult, replay
from promcast.core.report import recommend, report_json
from promcast.shell import bench, sources
from promcast.shell.config import load_config, read_config
from promcast.shell.errors import (
    CompareExtraMissingError,
    ConfigInvalidError,
    PromcastError,
    ReplayArgumentsError,
    TsdbAuthRejectedError,
    TsdbRejectedError,
    TsdbUnavailableError,
    UnknownModelError,
)
from promcast.shell.model_download import download_model
from promcast.shell.rules import generate_rules
from promcast.shell.tsdb import TsdbError

if TYPE_CHECKING:
    from promcast.core.forecaster import Forecaster
    from promcast.shell.auto_replay import AutoReplay


@runtime_checkable
class _ScanFunction(Protocol):
    async def __call__(
        self,
        arguments: argparse.Namespace,
        load_forecaster: Callable[[Path, str, int], Forecaster],
    ) -> None: ...


@runtime_checkable
class _AutoPreparation(Protocol):
    async def __call__(
        self,
        arguments: argparse.Namespace,
        forecaster: Forecaster,
    ) -> tuple[Config, list[AutoReplay], int]: ...


@runtime_checkable
class _ErrorOutput(Protocol):
    def error_report(self, error: PromcastError) -> str: ...
    def bug_report(self, error: Exception) -> str: ...


@runtime_checkable
class _ReplayOutput(Protocol):
    def plot_replay(self, result: ReplayResult, values: sources.FloatArray, path: Path) -> str: ...
    def terminal_table(
        self, result: ReplayResult, caveats: tuple[str, ...], *, foundation_models: bool
    ) -> str: ...
    def markdown_report(
        self,
        result: ReplayResult,
        caveats: tuple[str, ...],
        secondary: ReplayResult | None,
        foundation_models: bool,
    ) -> str: ...


@runtime_checkable
class _ChronosFactory(Protocol):
    def __call__(self, model_dir: Path, *, batch_size: int, threads: int | None) -> Forecaster: ...


@runtime_checkable
class _TotoFactory(Protocol):
    def __call__(
        self, model_dir: Path, name: str, *, batch_size: int, threads: int | None
    ) -> Forecaster: ...


@runtime_checkable
class _ServeFunction(Protocol):
    async def __call__(self, config: Config, forecaster: Forecaster, model_sha: str) -> None: ...


def load_forecaster(
    model_dir: Path, name: str, batch_size: int = 16, threads: int | None = None
) -> Forecaster:
    """Load a model, downloading pinned and verified files on first use."""
    try:
        return _select_forecaster(model_dir, name, batch_size, threads)
    except FileNotFoundError:
        download_model(model_dir, name)
        return _select_forecaster(model_dir, name, batch_size, threads)


def _select_forecaster(
    model_dir: Path, name: str, batch_size: int, threads: int | None
) -> Forecaster:
    if name == "chronos-2-small":
        try:
            chronos_factory: object = import_module(
                "promcast.core.forecaster_chronos"
            ).Chronos2Forecaster
            if not isinstance(chronos_factory, _ChronosFactory):
                raise TypeError("Chronos2Forecaster must be callable")
            return chronos_factory(model_dir, batch_size=batch_size, threads=threads)
        except ModuleNotFoundError as error:
            if error.name != "chronos":
                raise
            raise CompareExtraMissingError() from None
    if name not in {"toto-2.0-22m", "toto-2.0-4m"}:
        raise UnknownModelError(name)
    toto_factory: object = import_module("promcast.core.forecaster").Toto2Forecaster
    if not isinstance(toto_factory, _TotoFactory):
        raise TypeError("Toto2Forecaster must be callable")
    return toto_factory(model_dir, name, batch_size=batch_size, threads=threads)


def _finite_threshold(value: str) -> float:
    threshold = float(value)
    if not math.isfinite(threshold):
        raise argparse.ArgumentTypeError("threshold must be finite")
    return threshold


def add_scan_arguments(command: argparse.ArgumentParser) -> None:
    """Register shared discovery options for scan, init and automatic replay."""
    command.add_argument(
        "url",
        nargs="?",
        help="Prometheus URL; otherwise use PROMETHEUS_URL, then probe "
        "localhost:9090, localhost:8428, host.docker.internal:9090 and prometheus:9090",
    )
    command.add_argument(
        "--config", type=Path, help="configuration with TSDB authentication and TLS settings"
    )
    command.add_argument(
        "--recipes", type=Path, help="recipe YAML directory; defaults to bundled recipes"
    )
    command.add_argument(
        "--max-series", type=int, default=50, help="maximum series per target (1..500; default: 50)"
    )
    command.add_argument(
        "--query", help="custom PromQL; requires --below or --above in query units"
    )
    direction = command.add_mutually_exclusive_group()
    direction.add_argument(
        "--below",
        type=_finite_threshold,
        help="forecast a breach below this threshold in query units",
    )
    direction.add_argument(
        "--above",
        type=_finite_threshold,
        help="forecast a breach above this threshold in query units",
    )


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        super().error(safe_urls(message))


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="promcast")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a YAML configuration")
    validate.add_argument("file", type=Path, help="configuration file")
    validate.add_argument(
        "--fragment",
        action="store_true",
        help="validate recommended targets and their required model",
    )
    for name, help_text in (
        ("serve", "serve forecasts, metrics and health"),
        ("doctor", "inspect redacted configuration and local model files"),
        ("rules", "print generated Prometheus alert rules"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
    bench = commands.add_parser("bench", help="benchmark CPU forecast cycles")
    bench.add_argument("--json", action="store_true")
    bench.add_argument(
        "--cache-dir",
        type=Path,
        default=cache_dir_override(os.environ) or ModelConfig().cache_dir,
    )
    for name in ("scan", "init"):
        description = (
            "discover capacity gauges and print forecast threshold crossings"
            if name == "scan"
            else "discover targets and write configuration, rules and dashboards"
        )
        command = commands.add_parser(name, help=description, description=description)
        add_scan_arguments(command)
        if name == "scan":
            command.add_argument("--json", action="store_true")
            command.add_argument(
                "--redact",
                action="store_true",
                help="hide the TSDB URL and replace labels with stable aliases",
            )
            command.add_argument("--svg", type=Path, help="also write the board as an SVG file")
        else:
            command.add_argument("--out", type=Path, default=Path("promcast"))
            command.add_argument("--force", action="store_true")
    replay = commands.add_parser("replay", help="replay historical forecasts")
    replay.add_argument("--source", choices=("tsdb", "boom", "nab", "synthetic"))
    for flag in ("scenario", "target", "from", "to", "models", "incidents", "input"):
        replay.add_argument(f"--{flag}")
    replay.add_argument("--auto", action="store_true")
    add_scan_arguments(replay)
    replay.add_argument("--out", type=Path)
    replay.add_argument("--budget", type=float, default=1.0)
    return parser


def _model_digest(config: Config) -> str:
    weights = config.model.cache_dir.expanduser() / config.model.name / "model.safetensors"
    with weights.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def _serve(config: Config) -> None:
    serve_function: object = import_module("promcast.shell.server").serve
    if not isinstance(serve_function, _ServeFunction):
        raise TypeError("serve must be callable")
    model_dir = config.model.cache_dir.expanduser() / config.model.name
    forecaster = await asyncio.to_thread(
        load_forecaster, model_dir, config.model.name, config.model.batch_size
    )
    model_sha = await asyncio.to_thread(_model_digest, config)
    await serve_function(config, forecaster, model_sha)


def _doctor(config: Config) -> int:
    model_dir = config.model.cache_dir.expanduser() / config.model.name
    files = {name: (model_dir / name).is_file() for name in ("config.json", "model.safetensors")}
    document = config.model_dump(mode="json")
    document["tsdb"]["url"] = safe_urls(str(config.tsdb.url))
    report = {
        "config": document,
        "model_files": files,
        "model_sha": _model_digest(config) if files["model.safetensors"] else None,
    }
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0 if all(files.values()) else 1


def _replay_source(arguments: argparse.Namespace, config: Config | None) -> sources.ReplaySource:
    target = None
    if config is not None:
        matches = [target for target in config.targets if target.name == arguments.target]
        if len(matches) != 1:
            raise ReplayArgumentsError("--target must name exactly one configured target")
        target = matches[0]
    if arguments.source == "synthetic":
        return sources.synthetic(arguments.scenario or "accelerating_leak")
    if arguments.source in {"nab", "boom"}:
        if not arguments.input:
            raise ReplayArgumentsError("--input is required for a NAB CSV or BOOM folder")
        if arguments.source == "nab":
            return sources.nab(Path(arguments.input))
        if not arguments.target:
            raise ReplayArgumentsError("--target is required for a BOOM item_id")
        return sources.boom(Path(arguments.input), arguments.target)
    if config is None or target is None or not arguments.__dict__["from"] or not arguments.to:
        raise ReplayArgumentsError("TSDB replay requires --config, --target, --from and --to")
    return asyncio.run(sources.tsdb(config, target, arguments.__dict__["from"], arguments.to))


def _replay_models(arguments: argparse.Namespace, config: Config | None) -> dict[str, Forecaster]:
    # Baselines run without model downloads; explicit --models selects local weights.
    names: list[str] = arguments.models.split(",") if arguments.models else []
    cache = config.model.cache_dir if config else Path("~/.cache/promcast/models")
    return {name: load_forecaster(cache.expanduser() / name, name) for name in names}


def _replay_document(result: ReplayResult, source: sources.ReplaySource) -> str:
    document: dict[str, object] = json.loads(report_json(result))
    document["source"] = {"start_ts": source.start_ts, "caveats": source.caveats}
    document["test_sweep"] = {
        name: [
            {
                "point": asdict(point),
                "score": {
                    **asdict(score),
                    "precision": score.precision,
                    "recall": score.recall,
                    "false_alerts_per_week": score.false_alerts_per_week,
                },
            }
            for point, score in sorted(method.test_sweep.items())
        ]
        for name, method in sorted(result.methods.items())
    }
    if any(method.labelled_test is not None for method in result.methods.values()):
        document["labelled_incidents"] = json.loads(report_json(_labelled_result(result)))
    return json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _labelled_result(result: ReplayResult) -> ReplayResult:
    return replace(
        result,
        methods={
            name: replace(method, test=method.labelled_test)
            for name, method in result.methods.items()
            if method.labelled_test is not None
        },
    )


def _replay(
    arguments: argparse.Namespace,
    auto_source: sources.ReplaySource | None = None,
    auto_config: Config | None = None,
    auto_models: dict[str, Forecaster] | None = None,
) -> None:
    presentation: object = import_module("promcast.shell.plotting")
    if not isinstance(presentation, _ReplayOutput):
        raise TypeError("replay presentation must provide plot and terminal output")

    config = auto_config or (
        load_config(Path(arguments.config), environment=os.environ) if arguments.config else None
    )
    source = auto_source or _replay_source(arguments, config)
    models = auto_models if auto_models is not None else _replay_models(arguments, config)
    names = ["linear", "seasonal-naive", "ets", "static-margin", *models]
    spec = replace(source.spec, budget_per_week=arguments.budget)
    points = [
        point
        for name in names
        for point in operating_points(name, spec.horizon_steps * spec.step_s)
    ]
    labels = (
        sources.labelled_incidents(
            Path(arguments.incidents),
            arguments.input or arguments.target or arguments.scenario or "",
            source.start_ts,
        )
        if arguments.incidents
        else None
    )
    result = replay(source.values, spec, points, models, labelled_incidents=labels)
    recommendation = None
    if config is not None:
        target = next(target for target in config.targets if target.name == arguments.target)
        recommendation = recommend(result, target)
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    alt = presentation.plot_replay(result, source.values, out / "replay.png")
    text = presentation.markdown_report(
        result,
        source.caveats,
        _labelled_result(result) if labels is not None else None,
        bool(models),
    )
    text += f"\n![{alt}](replay.png)\n"
    (out / "replay.md").write_text(text, encoding="utf-8")
    (out / "replay.json").write_text(_replay_document(result, source), encoding="utf-8")
    if recommendation is not None:
        (out / "recommended.yaml").write_text(recommendation, encoding="utf-8")
    sys.stdout.write(
        presentation.terminal_table(result, source.caveats, foundation_models=bool(models))
    )


def _replay_auto(arguments: argparse.Namespace) -> None:
    model_config = (
        load_config(Path(arguments.config), environment=os.environ) if arguments.config else None
    )
    if model_config is None and Path("promcast/promcast.yaml").exists():
        model_config = load_config(Path("promcast/promcast.yaml"), environment=os.environ)
    model = model_config.model if model_config else ModelConfig()
    forecaster = load_forecaster(
        model.cache_dir.expanduser() / model.name, model.name, model.batch_size
    )
    prepare: object = import_module("promcast.shell.auto_replay").prepare_auto
    if not isinstance(prepare, _AutoPreparation):
        raise TypeError("automatic replay must provide preparation")
    config, jobs, _ = asyncio.run(prepare(arguments, forecaster))
    models = _replay_models(arguments, config) if arguments.models else {model.name: forecaster}
    for index, job in enumerate(jobs):
        per_series = argparse.Namespace(**vars(arguments))
        per_series.target = job.target.name
        per_series.models = arguments.models or model.name
        per_series.out = arguments.out / job.target.name / str(index + 1)
        _replay(per_series, job.source, config, models)


def _run_replay(arguments: argparse.Namespace) -> None:
    if arguments.auto:
        _replay_auto(arguments)
    elif arguments.source:
        _replay(arguments)
    else:
        raise ReplayArgumentsError("replay requires --source or --auto")


def _validate(arguments: argparse.Namespace) -> int:
    text = read_config(arguments.file)
    if arguments.fragment:
        parse_fragment(text, environment=os.environ)
    else:
        parse_config(text, environment=os.environ)
    sys.stdout.write("ok\n")
    return 0


async def _run_scan(arguments: argparse.Namespace) -> None:
    run: object = import_module("promcast.shell.scan_cli").run_scan_command
    if not isinstance(run, _ScanFunction):
        raise TypeError("scan must provide a command function")
    await run(arguments, load_forecaster)


def _arguments() -> argparse.Namespace:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.command in {"scan", "init", "replay"} and arguments.url and arguments.config:
        parser.error("choose a URL or --config; authentication uses the configured endpoint")
    if arguments.command == "replay":
        if not arguments.auto:
            missing = [flag for flag in ("source", "out") if getattr(arguments, flag) is None]
            if missing:
                parser.error(
                    "the following arguments are required: "
                    + ", ".join(f"--{flag}" for flag in missing)
                )
        elif arguments.out is None:
            arguments.out = Path("replay")
    return arguments


def _tsdb_failure(error: TsdbError) -> PromcastError:
    if error.status is None:
        return TsdbUnavailableError(error.body)
    if error.status in {401, 403}:
        return TsdbAuthRejectedError(error.status)
    return TsdbRejectedError(error.status, error.body)


def _config_path(arguments: argparse.Namespace) -> str:
    path: object = getattr(arguments, "file", None) or getattr(arguments, "config", None)
    return str(path or "promcast.yaml")


def _error_output() -> _ErrorOutput:
    output: object = import_module("promcast.shell.render")
    if not isinstance(output, _ErrorOutput):
        raise TypeError("render must provide error output")
    return output


def _report(error: PromcastError) -> int:
    sys.stderr.write(_error_output().error_report(error))
    return 2


def _dispatch(arguments: argparse.Namespace) -> int:
    if arguments.command in {"scan", "init"}:
        asyncio.run(_run_scan(arguments))
    elif arguments.command == "replay":
        _run_replay(arguments)
    elif arguments.command == "bench":
        report = asyncio.run(bench.benchmark(arguments.cache_dir.expanduser()))
        sys.stdout.write(bench.render(report, json_output=arguments.json))
    elif arguments.command == "validate":
        return _validate(arguments)
    else:
        config_path: Path = arguments.config
        config = load_config(config_path, environment=os.environ)
        if arguments.command == "rules":
            sys.stdout.write(generate_rules(config))
        elif arguments.command == "doctor":
            return _doctor(config)
        else:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            asyncio.run(_serve(config))
    return 0


def main() -> int:
    """Run the command line interface and return its exit status.

    This is the only place that turns failures into an exit status: 2 for an expected
    failure the user can fix, 1 for anything else.
    """
    arguments = _arguments()
    try:
        return _dispatch(arguments)
    except PromcastError as error:
        return _report(error)
    except ConfigError as error:
        return _report(ConfigInvalidError(str(error), _config_path(arguments)))
    except TsdbError as error:
        return _report(_tsdb_failure(error))
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        sys.stderr.write(_error_output().bug_report(error))
        return 1
