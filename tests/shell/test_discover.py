import asyncio

import httpx
import pytest
import respx

from promcast.shell.discover import resolve_url
from promcast.shell.errors import NoPrometheusFound
from promcast.shell.tsdb import TsdbError


@respx.mock
def test_autodetect_order():
    urls = [
        "http://localhost:9090",
        "http://localhost:8428",
        "http://host.docker.internal:9090",
        "http://prometheus:9090",
    ]
    tried: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        tried.append(str(request.url).split("/api/")[0])
        assert request.extensions["timeout"] == dict.fromkeys(
            ["connect", "read", "write", "pool"], 2
        )
        return httpx.Response(503)

    for url in urls[:2]:
        respx.get(url + "/api/v1/query").mock(side_effect=respond)
    winner = respx.get(urls[2] + "/api/v1/query").respond(
        200, json={"status": "success", "data": {"resultType": "scalar", "result": [1, "1"]}}
    )
    assert asyncio.run(resolve_url(None, {})) == urls[2]
    assert winner.called
    assert tried == urls[:2]


@respx.mock
def test_explicit_url_takes_precedence_and_does_not_fall_back():
    def reject(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query"] == "1"
        return httpx.Response(503)

    route = respx.get("http://chosen:9090/api/v1/query").mock(side_effect=reject)
    with pytest.raises(TsdbError, match="HTTP 503"):
        asyncio.run(resolve_url("http://chosen:9090", {"PROMETHEUS_URL": "http://other:9090"}))
    assert route.call_count == 1


@respx.mock
def test_environment_url_precedes_defaults():
    respx.get("http://environment:9090/api/v1/query").respond(
        200, json={"status": "success", "data": {"resultType": "scalar", "result": [1, "1"]}}
    )
    assert (
        asyncio.run(resolve_url(None, {"PROMETHEUS_URL": "http://environment:9090"}))
        == "http://environment:9090"
    )


@respx.mock
@pytest.mark.parametrize("status", [401, 403, 404, 500])
@pytest.mark.parametrize("source", ["argument", "environment"])
def test_explicit_status_is_preserved(status: int, source: str):

    url = "http://explicit:9090"
    respx.get(url + "/api/v1/query").respond(status, text="denied")
    with pytest.raises(TsdbError) as error:
        asyncio.run(
            resolve_url(
                url if source == "argument" else None,
                {"PROMETHEUS_URL": url} if source == "environment" else {},
            )
        )
    assert error.value.status == status


@respx.mock
def test_discovery_misses_keep_discovery_recovery():
    respx.get().respond(503)
    with pytest.raises(NoPrometheusFound) as error:
        asyncio.run(resolve_url(None, {}))
    assert "no Prometheus found" in error.value.what
    assert "localhost:9090" in error.value.what
