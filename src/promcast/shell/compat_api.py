import json
import logging
import math
import re
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from promcast.core.forecast import Forecast
from promcast.shell.store import Store

LOGGER = logging.getLogger(__name__)
_ERROR = "promcast serves only promcast_forecast_q10/q50/q90 and promcast_baseline_linear"
_METRICS = (
    "promcast_forecast_q10",
    "promcast_forecast_q50",
    "promcast_forecast_q90",
    "promcast_baseline_linear",
)
_SELECTOR = re.compile(
    r"(promcast_forecast_q(?:10|50|90)|promcast_baseline_linear)(?:\{(.*)\})?", re.DOTALL
)
_MATCHER = re.compile(r'\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*("(?:[^"\\]|\\.)*")\s*', re.DOTALL)
_DURATION = re.compile(r"([0-9]+)(ms|s|m|h|d|w|y)")
_SECONDS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}
# Bound work per series using the Prometheus query_range point limit.
_MAX_RANGE_POINTS = 11_000


class _BadDataError(ValueError):
    """A rejection whose message is written for the client."""


@dataclass(frozen=True)
class _Selector:
    metric: str
    labels: Mapping[str, str]


@dataclass(frozen=True)
class _Grid:
    start: float
    end: float
    step: float


def _selector(query: str) -> _Selector:
    match = _SELECTOR.fullmatch(query.strip())
    if match is None:
        raise _BadDataError(_ERROR)
    labels: dict[str, str] = {}
    text = match.group(2) or ""
    while text.strip():
        item = _MATCHER.match(text)
        if item is None or item.group(1) in labels:
            raise _BadDataError(_ERROR)
        labels[item.group(1)] = json.loads(item.group(2))
        text = text[item.end() :]
        if text and not text.startswith(","):
            raise _BadDataError(_ERROR)
        text = text[1:]
    return _Selector(match.group(1), labels)


def _timestamp(value: str) -> float:
    try:
        result = float(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise _BadDataError("timestamps must include a timezone") from None
        result = parsed.timestamp()
    if not math.isfinite(result):
        raise _BadDataError("timestamps must be finite")
    return result


def _step(value: str) -> float:
    try:
        result = float(value)
    except ValueError:
        parts = list(_DURATION.finditer(value))
        if not parts or "".join(part.group() for part in parts) != value:
            raise _BadDataError("step must be seconds or a Prometheus duration") from None
        result = sum(int(part.group(1)) * _SECONDS[part.group(2)] for part in parts)
    if not math.isfinite(result) or result <= 0:
        raise _BadDataError("step must be finite and positive")
    return result


def _series(store: Store, selector: _Selector) -> Iterator[tuple[dict[str, str], Forecast]]:
    for state in store.snapshot():
        for forecast in state.forecasts:
            baseline = selector.metric == "promcast_baseline_linear"
            primary = (forecast.model == "linear") == (state.config.method == "linear")
            if (baseline and forecast.model != "linear") or (not baseline and not primary):
                continue
            labels = {
                **dict(forecast.labels),
                "target": forecast.target,
                "__name__": selector.metric,
            }
            if all(labels.get(name, "") == value for name, value in selector.labels.items()):
                yield labels, forecast


def _value(forecast: Forecast, metric: str, timestamp: float) -> str:
    row = 4 if metric == "promcast_baseline_linear" else int(metric[-2:]) // 10 - 1
    position = (timestamp - forecast.origin_ts) / forecast.step_s - 1
    left = math.floor(position)
    right = min(left + 1, forecast.quantiles.shape[1] - 1)
    start = float(forecast.quantiles[row, left])
    end = float(forecast.quantiles[row, right])
    return str(start + (position - left) * (end - start))


def _bounds(forecast: Forecast) -> tuple[float, float]:
    return (
        forecast.origin_ts + forecast.step_s,
        forecast.origin_ts + forecast.step_s * forecast.quantiles.shape[1],
    )


def _query(store: Store, params: Mapping[str, list[str]], now: float) -> dict[str, object]:
    query = params.get("query", [""])[0]
    timestamp = _timestamp(params.get("time", [str(now)])[0])
    if query == "1+1":
        return {"resultType": "scalar", "result": [timestamp, "2"]}
    selector = _selector(query)
    result: list[dict[str, object]] = []
    for labels, forecast in _series(store, selector):
        start, end = _bounds(forecast)
        if start <= timestamp <= end:
            result.append(
                {
                    "metric": labels,
                    "value": [timestamp, _value(forecast, selector.metric, timestamp)],
                }
            )
    return {"resultType": "vector", "result": result}


def _range(store: Store, params: Mapping[str, list[str]]) -> dict[str, object]:
    selector = _selector(params.get("query", [""])[0])
    grid = _Grid(
        _timestamp(params["start"][0]), _timestamp(params["end"][0]), _step(params["step"][0])
    )
    if grid.end < grid.start:
        raise _BadDataError("end must not precede start")
    intervals = (grid.end - grid.start) / grid.step
    if not math.isfinite(intervals) or intervals >= _MAX_RANGE_POINTS:
        raise _BadDataError("range exceeds 11,000 points per series; increase step")
    result: list[dict[str, object]] = []
    for labels, forecast in _series(store, selector):
        first, last = _bounds(forecast)
        start = max(grid.start, forecast.origin_ts + grid.step, first)
        end = min(grid.end, last)
        if start > end:
            continue
        values = [
            [timestamp, _value(forecast, selector.metric, timestamp)]
            for index in range(math.floor((end - start) / grid.step) + 1)
            if (timestamp := start + index * grid.step) <= end
        ]
        result.append({"metric": labels, "values": values})
    return {"resultType": "matrix", "result": result}


def _discovery(store: Store, params: Mapping[str, list[str]]) -> list[dict[str, str]]:
    selectors = [_selector(item) for item in params.get("match[]", _METRICS)]
    start = _timestamp(params["start"][0]) if "start" in params else -math.inf
    end = _timestamp(params["end"][0]) if "end" in params else math.inf
    if end < start:
        raise _BadDataError("end must not precede start")
    found: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for selector in selectors:
        for labels, forecast in _series(store, selector):
            first, last = _bounds(forecast)
            if first <= end and last >= start:
                found[tuple(sorted(labels.items()))] = labels
    return list(found.values())


def _success(data: object) -> JSONResponse:
    return JSONResponse({"status": "success", "data": data})


def create_router(store: Store, clock: Callable[[], float] = time.time) -> APIRouter:
    """Serve the Grafana Prometheus subset from current forecast snapshots."""
    router = APIRouter(prefix="/api/v1")

    @router.get("/status/buildinfo")
    async def buildinfo() -> JSONResponse:
        return _success(
            {
                "version": "3.15.1",
                "revision": "promcast",
                "branch": "",
                "buildUser": "",
                "buildDate": "",
                "goVersion": "",
            }
        )

    @router.get("/metadata")
    async def metadata() -> JSONResponse:
        return _success({})

    @router.api_route("/query", methods=["GET", "POST"])
    @router.api_route("/query_range", methods=["GET", "POST"])
    @router.api_route("/labels", methods=["GET", "POST"])
    @router.api_route("/label/{name}/values", methods=["GET", "POST"])
    @router.api_route("/series", methods=["GET", "POST"])
    async def dispatch(request: Request) -> JSONResponse:
        try:
            params = parse_qs(request.url.query, keep_blank_values=True)
            if request.method == "POST":
                params.update(parse_qs((await request.body()).decode(), keep_blank_values=True))
            path = request.url.path
            if path.endswith("/query"):
                return _success(_query(store, params, clock()))
            if path.endswith("/query_range"):
                return _success(_range(store, params))
            series = _discovery(store, params)
            if path.endswith("/labels"):
                return _success(sorted({name for labels in series for name in labels}))
            if path.endswith("/values"):
                name = request.path_params["name"]
                return _success(sorted({labels[name] for labels in series if name in labels}))
            return _success(series)
        except (ValueError, KeyError, OverflowError) as error:
            if isinstance(error, _BadDataError):
                message = str(error)
            else:
                LOGGER.exception("rejected %s request", request.url.path)
                message = "invalid request parameters"
            return JSONResponse(
                {"status": "error", "errorType": "bad_data", "error": message}, status_code=400
            )

    return router
