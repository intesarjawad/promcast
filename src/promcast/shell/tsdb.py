import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from promcast.core.config import TSDBConfig
from promcast.core.diagnostics import safe_urls

# Prometheus query_range allows at most 11,000 points per series.
_MAX_CHUNK_POINTS = 11_000
_MAX_ERROR_BODY_CHARS = 512


@dataclass(frozen=True)
class RangeGrid:
    """Bounds and step for a TSDB range query."""

    start_ts: float
    end_ts: float
    step_s: int


class TsdbError(RuntimeError):
    """A safe TSDB diagnostic, with an HTTP status when a response was received."""

    def __init__(self, status: int | None, body: str) -> None:
        self.status = status
        self.body = safe_urls(body)
        source = f"TSDB returned HTTP {status}" if status is not None else "TSDB request failed"
        super().__init__(f"{source}: {self.body}")


class _MatrixSeries(BaseModel):
    metric: dict[str, str]
    values: list[tuple[Annotated[float, Field(allow_inf_nan=False)], str]]


class _Matrix(BaseModel):
    result_type: Literal["matrix"] = Field(alias="resultType")
    result: list[_MatrixSeries]


class _Response(BaseModel):
    status: Literal["success"]
    data: _Matrix


def _read_secret(path: Path) -> str:
    try:
        return path.expanduser().read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        raise TsdbError(
            None, "cannot read authentication file; check path, permissions and UTF-8"
        ) from None


def _authentication(cfg: TSDBConfig) -> tuple[httpx.Headers, httpx.BasicAuth | None, set[str]]:
    headers = httpx.Headers({name: value.get_secret_value() for name, value in cfg.headers.items()})
    secrets = set(headers.values())
    if "Authorization" in headers:
        scheme, _, credential = headers["Authorization"].partition(" ")
        if scheme.lower() in {"bearer", "basic"} and credential:
            secrets.add(credential.strip())
    auth = None
    if cfg.bearer_token_file is not None:
        token = _read_secret(cfg.bearer_token_file)
        headers["Authorization"] = f"Bearer {token}"
        secrets.update((token, headers["Authorization"]))
    if cfg.basic_auth is not None:
        username = cfg.basic_auth.username.get_secret_value()
        password = _read_secret(cfg.basic_auth.password_file)
        auth = httpx.BasicAuth(username, password)
        encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
        secrets.update((username, password, encoded))
    return headers, auth, secrets


def _safe_body(body: str, secrets: set[str]) -> str:
    for secret in sorted(filter(None, secrets), key=len, reverse=True):
        body = body.replace(secret, "<redacted>")
    body = safe_urls(body)
    if len(body) > _MAX_ERROR_BODY_CHARS:
        return body[:_MAX_ERROR_BODY_CHARS] + "..."
    return body


async def range_query(
    client: httpx.AsyncClient,
    cfg: TSDBConfig,
    query: str,
    grid: RangeGrid,
) -> dict[str, object]:
    """Fetch and stitch a matrix on the original grid, including both range bounds.

    The caller owns the client lifetime and must configure its TLS trust using
    cfg.tls.ca_file when supplied. Authentication files are read once per query.
    """
    start_ts, end_ts, step_s = grid.start_ts, grid.end_ts, grid.step_s
    if not math.isfinite(start_ts) or not math.isfinite(end_ts) or step_s <= 0 or end_ts < start_ts:
        raise ValueError("range bounds must be finite and ordered, and step_s positive")
    headers, auth, secrets = _authentication(cfg)
    url = str(cfg.url).rstrip("/") + "/api/v1/query_range"
    series: dict[tuple[tuple[str, str], ...], dict[float, str]] = {}
    total_points = math.floor((end_ts - start_ts) / step_s) + 1
    for offset in range(0, total_points, _MAX_CHUNK_POINTS):
        chunk_start = start_ts + offset * step_s
        chunk_end = min(end_ts, chunk_start + (_MAX_CHUNK_POINTS - 1) * step_s)
        try:
            response = await client.get(
                url,
                params={"query": query, "start": chunk_start, "end": chunk_end, "step": step_s},
                headers=headers,
                auth=auth,
                timeout=cfg.timeout_s,
                follow_redirects=False,
            )
        except httpx.RequestError:
            raise TsdbError(None, "connection failed; check the TSDB address and network") from None
        if not response.is_success:
            raise TsdbError(response.status_code, _safe_body(response.text, secrets))
        try:
            matrix = _Response.model_validate_json(response.content).data
        except ValidationError:
            raise TsdbError(
                response.status_code,
                "invalid query_range response: " + _safe_body(response.text, secrets),
            ) from None
        for item in matrix.result:
            samples = series.setdefault(tuple(sorted(item.metric.items())), {})
            samples.update(item.values)
    return {
        "resultType": "matrix",
        "result": [
            {"metric": dict(labels), "values": sorted(samples.items())}
            for labels, samples in series.items()
        ],
    }


class _CountSeries(BaseModel):
    metric: dict[str, str]
    value: tuple[Annotated[float, Field(allow_inf_nan=False)], Annotated[int, Field(ge=0)]]


class _CountVector(BaseModel):
    result_type: Literal["vector"] = Field(alias="resultType")
    result: list[_CountSeries] = Field(max_length=1)


class _CountResponse(BaseModel):
    status: Literal["success"]
    data: _CountVector


async def series_count(client: httpx.AsyncClient, cfg: TSDBConfig, query: str, end_ts: int) -> int:
    """Count query series at end_ts, returning zero for an empty vector.

    The caller owns the client and configures its TLS trust, as for range_query.
    """
    headers, auth, secrets = _authentication(cfg)
    try:
        response = await client.get(
            str(cfg.url).rstrip("/") + "/api/v1/query",
            params={"query": f"count({query})", "time": end_ts},
            headers=headers,
            auth=auth,
            timeout=cfg.timeout_s,
            follow_redirects=False,
        )
    except httpx.RequestError:
        raise TsdbError(None, "connection failed; check the TSDB address and network") from None
    if not response.is_success:
        raise TsdbError(response.status_code, _safe_body(response.text, secrets))
    try:
        result = _CountResponse.model_validate_json(response.content).data.result
    except ValidationError:
        raise TsdbError(
            response.status_code,
            "invalid count response: " + _safe_body(response.text, secrets),
        ) from None
    return result[0].value[1] if result else 0


async def api_data(
    client: httpx.AsyncClient, cfg: TSDBConfig, endpoint: str, params: dict[str, str | int]
) -> object:
    """Read a discovery API response with the same authentication as range queries."""
    headers, auth, secrets = _authentication(cfg)
    try:
        response = await client.get(
            str(cfg.url).rstrip("/") + "/api/v1/" + endpoint,
            params=params,
            headers=headers,
            auth=auth,
            timeout=cfg.timeout_s,
            follow_redirects=False,
        )
    except httpx.RequestError:
        raise TsdbError(None, "connection failed; check the TSDB address and network") from None
    if not response.is_success:
        raise TsdbError(response.status_code, _safe_body(response.text, secrets))
    try:
        return _APIResponse.model_validate_json(response.content).data
    except ValidationError:
        raise TsdbError(response.status_code, "invalid discovery response") from None


class _APIResponse(BaseModel):
    status: Literal["success"]
    data: object
