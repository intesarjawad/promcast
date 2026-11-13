import asyncio
import base64
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest
import respx

from promcast.core.config import TSDBConfig
from promcast.core.series import align
from promcast.shell.tsdb import RangeGrid, TsdbError, range_query

_URL = "https://prom.example/prometheus/api/v1/query_range"


def _config(**fields: object) -> TSDBConfig:
    return TSDBConfig.model_validate({"url": "https://prom.example/prometheus/", **fields})


def _matrix(*items: Mapping[str, object]) -> dict[str, object]:
    return {"status": "success", "data": {"resultType": "matrix", "result": list(items)}}


def _query(cfg: TSDBConfig, start: float = 0, end: float = 60, step: int = 15):
    async def run():
        async with httpx.AsyncClient() as client:
            return await range_query(client, cfg, "up", RangeGrid(start, end, step))

    return asyncio.run(run())


@pytest.mark.parametrize("points", [1, 11_000, 11_001, 22_001])
def test_chunks_preserve_grid_and_stitch_by_full_labels(points: int):
    start, step = 0.25, 15
    end = start + (points - 1) * step

    def respond(request: httpx.Request):
        params = request.url.params
        chunk_start, chunk_end = float(params["start"]), float(params["end"])
        assert params["query"] == "up"
        assert int(params["step"]) == step
        assert request.extensions["timeout"] == dict.fromkeys(
            ["connect", "read", "write", "pool"], 7.0
        )
        count = round((chunk_end - chunk_start) / step) + 1
        assert count <= 11_000
        samples = [[chunk_start + i * step, str(i)] for i in range(count)]
        items = [
            {"metric": {"job": "a", "__name__": "up"}, "values": samples},
            {"metric": {"__name__": "other", "job": "a"}, "values": samples},
        ]
        # Result order can change between chunks.
        return httpx.Response(200, json=_matrix(*reversed(items) if chunk_start > start else items))

    with respx.mock as router:
        route = router.get(_URL).mock(side_effect=respond)
        result = _query(_config(timeout_s=7), start, end, step)
        assert route.call_count == (points + 10_999) // 11_000
    accepted, skipped = align(result, step, end, points)
    assert not skipped
    assert len(accepted) == 2
    for series in accepted:
        assert len(series.values) == points
        assert not any(series.values != series.values)


def test_empty_and_new_series_and_nan_values_stitch():
    responses = [
        _matrix({"metric": {"job": "a"}, "values": [[0, "NaN"]]}),
        _matrix(),
        _matrix({"metric": {"job": "b"}, "values": [[22_000, "2"]]}),
    ]
    with respx.mock as router:
        router.get(_URL).mock(side_effect=[httpx.Response(200, json=r) for r in responses])
        assert _query(_config(), end=22_000, step=1) == {
            "resultType": "matrix",
            "result": [
                {"metric": {"job": "a"}, "values": [(0.0, "NaN")]},
                {"metric": {"job": "b"}, "values": [(22_000.0, "2")]},
            ],
        }


@pytest.mark.parametrize("auth_kind", ["bearer", "basic", "headers", "authorization"])
@pytest.mark.parametrize("status", [400, 401, 403, 500, 503])
def test_authentication_and_errors_are_redacted(tmp_path: Path, auth_kind: str, status: int):
    secret = "-".join(["private", "credential"])
    encoded = base64.b64encode(f"private-user:{secret}".encode()).decode()
    secret_file = tmp_path / "secret"
    secret_file.write_text(secret + "\n", encoding="utf-8")
    if auth_kind == "bearer":
        cfg = _config(bearer_token_file=secret_file)
        expected_headers = {"Authorization": f"Bearer {secret}"}
    elif auth_kind == "basic":
        cfg = _config(basic_auth={"username": "private-user", "password_file": secret_file})
        expected_headers = {"Authorization": f"Basic {encoded}"}
    elif auth_kind == "authorization":
        cfg = _config(headers={"Authorization": f"Bearer {secret}"})
        expected_headers = {"Authorization": f"Bearer {secret}"}
    else:
        cfg = _config(headers={"X-Scope-OrgID": secret})
        expected_headers = {"X-Scope-OrgID": secret}

    def respond(request: httpx.Request):
        for name, value in expected_headers.items():
            assert request.headers[name] == value
        body = " ".join([*expected_headers.values(), f"prefix{secret}suffix", "private-user"])
        return httpx.Response(status, text=body + "x" * 1000)

    with respx.mock as router:
        route = router.get(_URL).mock(side_effect=respond)
        with pytest.raises(TsdbError) as caught:
            _query(cfg)
        assert route.call_count == 1
    error = caught.value
    assert error.status == status
    assert str(status) in str(error)
    assert secret not in str(error)
    if auth_kind == "basic":
        assert "private-user" not in str(error)
        assert encoded not in str(error)
    assert len(error.body) <= 515
    assert error.body.endswith("...")
    assert error.__suppress_context__ or error.__context__ is None


@pytest.mark.parametrize(
    "body",
    ["not json", '{"status":"error","error":"bad_data"}', '{"status":"success","data":{}}'],
)
def test_malformed_success_response_is_a_tsdb_error(body: str):
    with respx.mock as router:
        router.get(_URL).respond(200, text=body)
        with pytest.raises(TsdbError, match="invalid query_range response") as caught:
            _query(_config())
    assert caught.value.status == 200


def test_transport_error_does_not_expose_request_details():
    with respx.mock as router:
        router.get(_URL).mock(side_effect=httpx.ConnectError("private transport details"))
        with pytest.raises(TsdbError) as caught:
            _query(_config())
    assert caught.value.status is None
    assert "private transport details" not in str(caught.value)


def test_unreadable_secret_does_not_send_request(tmp_path: Path):
    with respx.mock as router:
        with pytest.raises(TsdbError, match="cannot read authentication file"):
            _query(_config(bearer_token_file=tmp_path / "missing"))
        assert not router.calls


@pytest.mark.parametrize(("start", "end", "step"), [(2, 1, 1), (0, 1, 0), (0, float("inf"), 1)])
def test_invalid_ranges_do_not_send_request(start: float, end: float, step: int):
    with respx.mock as router:
        with pytest.raises(ValueError, match="range bounds"):
            _query(_config(), start, end, step)
        assert not router.calls


def test_non_grid_end_and_redirect_are_not_followed():
    def respond(request: httpx.Request):
        assert request.url.params["end"] == "10"
        return httpx.Response(302, headers={"Location": "https://elsewhere.example/"})

    with respx.mock as router:
        route = router.get(_URL).mock(side_effect=respond)
        with pytest.raises(TsdbError) as caught:
            _query(_config(), end=10, step=3)
        assert route.call_count == 1
        assert caught.value.status == 302
