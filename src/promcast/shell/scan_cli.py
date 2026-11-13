from __future__ import annotations

import argparse
import os
import re
import shutil
import ssl
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from promcast.core.config import Config, TSDBConfig, cache_dir_override
from promcast.core.recipes import Recipe
from promcast.shell import board, discover, render
from promcast.shell.adoption import write_init
from promcast.shell.config import load_config
from promcast.shell.errors import (
    MaxSeriesOutOfRangeError,
    NoRecipeMatchedError,
    QueryMetricUnknownError,
    QueryThresholdMissingError,
)
from promcast.shell.scan import ScanDocument, ScanOptions, ScanResult, redact, scan

if TYPE_CHECKING:
    from promcast.core.forecaster import Forecaster


async def discover_config(
    arguments: argparse.Namespace, client: httpx.AsyncClient, config: Config | None = None
) -> tuple[Config, list[Recipe]]:
    """Resolve the TSDB and build matched recipe targets without loading a model."""
    if not 1 <= arguments.max_series <= 500:
        raise MaxSeriesOutOfRangeError(arguments.max_series)
    if config is None:
        path = getattr(arguments, "config", None)
        if path:
            config = load_config(Path(path), environment=os.environ)
        else:
            url = await discover.resolve_url(arguments.url, os.environ, client)
            config = Config(tsdb=TSDBConfig.model_validate({"url": url}), targets=[])
    if (cache_dir := cache_dir_override(os.environ)) is not None:
        config.model.cache_dir = cache_dir
    names = await discover.metric_names(client, config.tsdb)
    if arguments.query:
        if arguments.below is None and arguments.above is None:
            raise QueryThresholdMissingError()
        # Raw scrape detection uses the query's first vector selector.
        metric = next(
            (
                item
                for item in re.finditer(r"[a-zA-Z_:][a-zA-Z0-9_:]*", arguments.query)
                if item[0] in names and not arguments.query[item.end() :].lstrip().startswith("(")
            ),
            None,
        )
        if metric is None:
            raise QueryMetricUnknownError(arguments.query)
        recipe = Recipe.model_validate(
            {
                "name": "custom",
                "requires": [metric[0]],
                "query": arguments.query,
                "direction": "below" if arguments.below is not None else "above",
                "threshold": arguments.below if arguments.below is not None else arguments.above,
                "unit": "count",
                "show": [],
                "source": "https://prometheus.io/docs/prometheus/latest/querying/basics/",
            }
        )
        return config, [recipe]
    recipes = discover.load_recipes(arguments.recipes)
    matched = discover.match(names, recipes)
    if not matched and arguments.recipes:
        raise NoRecipeMatchedError(str(arguments.recipes))
    return config, matched


async def run_scan_command(
    arguments: argparse.Namespace,
    load_forecaster: Callable[[Path, str, int], Forecaster],
) -> None:
    """Run discovery and keep JSON stdout separate from human output."""
    config = load_config(arguments.config, environment=os.environ) if arguments.config else None
    ca_file = config.tsdb.tls.ca_file if config else None
    trust = ssl.create_default_context(cafile=str(ca_file.expanduser())) if ca_file else True
    async with httpx.AsyncClient(verify=trust) as client:
        config, recipes = await discover_config(arguments, client, config)
        status = sys.stderr.isatty()
        columns = shutil.get_terminal_size((80, 24)).columns
        stderr_color = board.detect_color(sys.stderr)

        def show(stage: str, done: int = 0, total: int = 0) -> None:
            line = render.scan_progress(stage, done, total, width=columns - 1, color=stderr_color)
            sys.stderr.write("\r" + line)
            sys.stderr.flush()

        if status:
            show("discovery")
        end_ts = int(time.time())
        if not recipes:
            if arguments.command == "init":
                raise NoRecipeMatchedError()
            document = ScanDocument(
                url=str(config.tsdb.url).rstrip("/"),
                model=config.model.name,
                origin_ts=end_ts,
                rows=[],
                skipped=[],
            )
            if arguments.redact:
                document = redact(document)
            if arguments.json:
                sys.stdout.write(document.model_dump_json() + "\n")
            else:
                sys.stdout.write(
                    render.scan_empty(
                        document.url,
                        [r.name for r in discover.load_recipes()],
                        width=shutil.get_terminal_size((80, 24)).columns,
                        color=board.detect_color(sys.stdout),
                    )
                )
            return
        model = config.model
        forecaster: Forecaster = load_forecaster(
            model.cache_dir.expanduser() / model.name, model.name, model.batch_size
        )
        if status:
            show("forecast")
        result = await scan(
            client,
            config,
            recipes,
            forecaster,
            ScanOptions(
                end_ts,
                arguments.max_series,
                (lambda done, total: show("forecast", done, total)) if status else None,
            ),
        )
        if status:
            sys.stderr.write("\r" + " " * columns + "\r")
    if arguments.command == "init":
        write_init(
            config.model_copy(update={"targets": result.targets}),
            arguments.out,
            force=arguments.force,
        )
        sys.stdout.write(
            f"1. promcast validate {arguments.out}/promcast.yaml\n"
            f"2. docker compose -f {arguments.out}/compose.yml up -d\n"
            f"3. Add {arguments.out}/prometheus-snippet.yml to Prometheus\n"
        )
        return
    if arguments.redact:
        result = ScanResult(
            redact(result.document),
            result.targets,
            result.dropped,
            result.total,
            result.forecasts_per_s,
            result.replay_forecasts_needed,
        )
    width = shutil.get_terminal_size((80, 24)).columns
    if arguments.svg is not None:
        arguments.svg.write_text(board.scan_svg(result, width=width), encoding="utf-8")
    sys.stdout.write(
        result.document.model_dump_json() + "\n"
        if arguments.json
        else render.scan_board(result, width=width, color=board.detect_color(sys.stdout))
    )
