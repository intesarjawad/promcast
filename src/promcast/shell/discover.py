from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from statistics import median
from typing import Literal

import httpx
import numpy as np
import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from promcast.core.config import Config, TargetConfig, TSDBConfig
from promcast.core.recipes import Recipe, candidate_query, scan_target
from promcast.core.series import Labels, Series, align
from promcast.shell.errors import (
    NoPrometheusFound,
    RecipeInvalidError,
    RecipesMissingError,
)
from promcast.shell.tsdb import RangeGrid, TsdbError, api_data, range_query, series_count

_DEFAULT_URLS = (
    "http://localhost:9090",
    "http://localhost:8428",
    "http://host.docker.internal:9090",
    "http://prometheus:9090",
)


def load_recipes(directory: Path | None = None) -> list[Recipe]:
    """Load bundled recipes, then replace matching names from a local directory."""
    bundled = files("promcast").joinpath("recipes")
    recipes: dict[str, Recipe] = {}
    texts = [
        item.read_text()
        for item in sorted(bundled.iterdir(), key=str)
        if item.name.endswith(".yaml")
    ]
    if directory is not None:
        paths = sorted(directory.glob("*.yaml"))
        if not paths:
            raise RecipesMissingError(str(directory))
        texts.extend(path.read_text(encoding="utf-8") for path in paths)
    try:
        for text in texts:
            recipe = Recipe.model_validate(yaml.safe_load(text))
            recipes[recipe.name] = recipe
    except (ValidationError, yaml.YAMLError) as error:
        raise RecipeInvalidError(str(error)) from None
    return sorted(recipes.values(), key=lambda recipe: recipe.name)


def match(names: set[str], recipes: Sequence[Recipe]) -> list[Recipe]:
    """Match recipes whose required metrics all exist."""
    return [recipe for recipe in recipes if set(recipe.requires) <= names]


async def resolve_url(
    arg: str | None, env: Mapping[str, str], client: httpx.AsyncClient | None = None
) -> str:
    """Probe an explicit URL, environment URL, or the ordered local defaults."""
    if client is None:
        async with httpx.AsyncClient() as owned:
            return await resolve_url(arg, env, owned)
    urls: list[str] = (
        [arg]
        if arg
        else [env["PROMETHEUS_URL"]]
        if env.get("PROMETHEUS_URL")
        else list(_DEFAULT_URLS)
    )
    for url in urls:
        try:
            cfg = TSDBConfig.model_validate({"url": url, "timeout_s": 2})
            data = await api_data(client, cfg, "query", {"query": "1"})
            TypeAdapter(dict[str, object]).validate_python(data)
        except TsdbError:
            if arg or env.get("PROMETHEUS_URL"):
                raise
            continue
        except ValidationError as error:
            if arg or env.get("PROMETHEUS_URL"):
                raise TsdbError(None, f"invalid TSDB endpoint {url}: {error}") from None
            continue
        return url.rstrip("/")
    raise NoPrometheusFound(urls)


async def metric_names(client: httpx.AsyncClient, cfg: TSDBConfig) -> set[str]:
    """Read metric names without requesting label values or sample history."""
    data = await api_data(client, cfg, "label/__name__/values", {})
    try:
        return set(TypeAdapter(list[str]).validate_python(data))
    except ValidationError:
        raise TsdbError(None, "invalid metric names response") from None


class Samples(BaseModel):
    """Raw scrape timestamps or per-series count responses."""

    metric: dict[str, str]
    values: list[tuple[float, str]] = []
    value: tuple[float, float] | None = None

    @property
    def labels(self) -> Labels:
        """Normalize identity in the same way as range alignment."""
        return tuple(sorted((k, v) for k, v in self.metric.items() if k != "__name__"))


class SampleResult(BaseModel):
    """Validated discovery sample response."""

    result_type: Literal["matrix", "vector"] = Field(alias="resultType")
    result: list[Samples]


@dataclass(frozen=True)
class History:
    """Aligned candidates and their independent raw observation counts."""

    target: TargetConfig
    series: list[Series]
    intervals: dict[Labels, int]
    counts: dict[Labels, int]
    total: int


async def read_history(
    client: httpx.AsyncClient, config: Config, recipe: Recipe, end_ts: int, max_series: int
) -> History:
    """Choose a scrape-safe grid and count real samples for selected candidates."""
    query = candidate_query(recipe.query, recipe.direction, max_series, end_ts)
    raw_query = f"{recipe.requires[0]}[10m] @ {end_ts}"
    raw = SampleResult.model_validate(
        await api_data(client, config.tsdb, "query", {"query": raw_query, "time": end_ts})
    )
    # Restrict interval selection to the candidates, retaining their recipe labels.
    candidates = SampleResult.model_validate(
        await api_data(client, config.tsdb, "query", {"query": query, "time": end_ts})
    )
    intervals: dict[Labels, int] = {}
    for candidate in candidates.result:
        gaps = [
            max(
                1,
                round(
                    median(b[0] - a[0] for a, b in zip(item.values, item.values[1:], strict=False))
                ),
            )
            for item in raw.result
            if len(item.values) >= 2 and all(item.metric.get(k) == v for k, v in candidate.labels)
        ]
        if gaps:
            intervals[candidate.labels] = max(gaps)
    allowed = [step for step in (300, 60, 15) if step >= max(intervals.values(), default=15)]
    if not allowed:
        allowed = [max(intervals.values())]
    series: list[Series] = []
    target = scan_target(recipe, allowed[0], max_series)
    for step in allowed:
        target = scan_target(recipe, step, max_series)
        steps = int(target.context.total_seconds()) // step
        matrix = await range_query(
            client, config.tsdb, query, RangeGrid(end_ts - (steps - 1) * step, end_ts, step)
        )
        aligned, _ = align(matrix, step, end_ts, steps)
        series = [_trim_start(item) for item in aligned]
        if any(np.count_nonzero(~np.isnan(item.values)) >= 64 for item in series):
            break
    range_s = int(target.context.total_seconds())
    count_query = f"count_over_time({recipe.requires[0]}[{range_s}s] @ {end_ts})"
    counts = SampleResult.model_validate(
        await api_data(client, config.tsdb, "query", {"query": count_query, "time": end_ts})
    )
    real_counts: dict[Labels, int] = {}
    for item in series:
        matching = [
            int(count.value[1])
            for count in counts.result
            if count.value is not None and all(count.metric.get(k) == v for k, v in item.labels)
        ]
        real_counts[item.labels] = min(matching, default=0)
    total = await series_count(client, config.tsdb, recipe.query, end_ts)
    return History(target, series, intervals, real_counts, total)


def _trim_start(series: Series) -> Series:
    observed = np.flatnonzero(~np.isnan(series.values))
    start = int(observed[0]) if observed.size else series.values.size
    return Series(series.labels, series.step_s, series.end_ts, series.values[start:].copy())
