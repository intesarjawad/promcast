from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from promcast.core.alerting import Direction
from promcast.core.config import TargetConfig


class Recipe(BaseModel):
    """Metric requirements and capacity defaults for one scan target."""

    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_-]*$")]
    requires: Annotated[list[str], Field(min_length=1)]
    query: Annotated[str, Field(min_length=1)]
    direction: Direction
    threshold: Annotated[float, Field(allow_inf_nan=False)]
    unit: Literal["ratio", "bytes", "seconds", "count"]
    show: list[str]
    source: HttpUrl


def candidate_query(query: str, direction: Direction, n: int, end_ts: int) -> str:
    """Select the nearest candidates at a fixed range end, including across chunks."""
    if not 1 <= n <= 500:
        raise ValueError("--max-series must be between 1 and 500")
    selector = "bottomk" if direction == "below" else "topk"
    return f"({query}) and {selector}({n}, last_over_time(({query})[10m:] @ {end_ts}))"


def scan_target(recipe: Recipe, step_s: int, max_series: int) -> TargetConfig:
    """Apply scan defaults within the shared configuration limits."""
    horizon_s = min(21600, 512 * step_s)
    return TargetConfig.model_validate(
        {
            "name": recipe.name,
            "known_gauge": recipe.name != "custom",
            "query": recipe.query,
            "step": f"{step_s}s",
            "context": f"{min(345600, 2048 * step_s)}s",
            "horizon": f"{horizon_s}s",
            "season": "1d",
            "direction": recipe.direction,
            "threshold": recipe.threshold,
            "quantile": 0.9,
            "warn": f"{min(3600, horizon_s)}s",
            "method": "toto",
            "linear_window": "1h",
            "max_series": max_series,
        }
    )
