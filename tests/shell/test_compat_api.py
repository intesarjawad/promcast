import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from promcast.core.config import TargetConfig
from promcast.core.forecast import Forecast
from promcast.shell.compat_api import create_router
from promcast.shell.store import Store

LABELS = (("instance", "node-a"),)


class GrafanaCall(BaseModel):
    method: Literal["GET", "POST"]
    path: str
    query_params: dict[str, list[str]]
    form_params: dict[str, list[str]]


CALLS = [
    GrafanaCall.model_validate(call)
    for call in json.loads(Path("tests/fixtures/grafana_calls.json").read_text())
]


@pytest.fixture
def store() -> Store:
    target = TargetConfig.model_validate(
        {
            "name": "space",
            "query": "space",
            "step": "1m",
            "context": "128m",
            "horizon": "1h",
            "season": "1h",
            "direction": "below",
            "threshold": 0,
            "quantile": 0.9,
            "warn": "5m",
            "method": "toto",
            "linear_window": "10m",
            "max_series": 2,
        }
    )
    forecasts = [
        Forecast(
            "space",
            LABELS,
            100,
            60,
            np.arange(9, dtype=np.float64)[:, None] * 10
            + np.array([10, 20, 30], dtype=np.float64)[None, :]
            + offset,
            0,
            model,
        )
        for model, offset in [("toto", 0), ("linear", 100)]
    ]
    result = Store()
    result.update(target, forecasts, [], {LABELS: 5}, 101)
    return result


@pytest.fixture
def client(store: Store) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(store, lambda: 190))
    return TestClient(app)


@pytest.mark.parametrize("call", CALLS, ids=lambda call: f"{call.method} {call.path}")
def test_replay_grafana_calls(client: TestClient, call: GrafanaCall) -> None:
    response = client.request(
        call.method,
        call.path,
        params=urlencode(call.query_params, doseq=True),
        content=urlencode(call.form_params, doseq=True),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_buildinfo_and_metadata(client: TestClient) -> None:
    assert client.get("/api/v1/status/buildinfo").json() == {
        "status": "success",
        "data": {
            "version": "3.15.1",
            "revision": "promcast",
            "branch": "",
            "buildUser": "",
            "buildDate": "",
            "goVersion": "",
        },
    }
    assert client.get("/api/v1/metadata").json() == {"status": "success", "data": {}}


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("1+1", {"resultType": "scalar", "result": [190, "2"]}),
        (
            'promcast_forecast_q50{target="space", instance="node-a"}',
            {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {
                            "__name__": "promcast_forecast_q50",
                            "target": "space",
                            "instance": "node-a",
                        },
                        "value": [190, "55.0"],
                    }
                ],
            },
        ),
    ],
)
def test_instant_query(client: TestClient, method: str, query: str, expected: object) -> None:
    response = (
        client.get("/api/v1/query", params={"query": query})
        if method == "GET"
        else client.post("/api/v1/query", data={"query": query})
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success", "data": expected}


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("promcast_forecast_q10", "15.0"),
        ("promcast_forecast_q90", "95.0"),
        ("promcast_baseline_linear", "155.0"),
    ],
)
def test_quantile_and_baseline(client: TestClient, method: str, metric: str, expected: str) -> None:
    params = {"query": metric, "time": "1970-01-01T00:03:10Z"}
    response = client.request(method, "/api/v1/query", params=params)
    assert response.status_code == 200
    assert response.json()["data"]["result"][0]["value"] == [190, expected]


@pytest.mark.parametrize("timestamp", ["99", "100", "159", "281"])
def test_instant_outside_forecast(client: TestClient, timestamp: str) -> None:
    response = client.get(
        "/api/v1/query", params={"query": "promcast_forecast_q50", "time": timestamp}
    )
    assert response.json()["data"]["result"] == []


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("step", ["60", "1m", "60000ms"])
def test_range_bounds(client: TestClient, method: str, step: str) -> None:
    params = {"query": "promcast_forecast_q50", "start": "0", "end": "400", "step": step}
    response = client.request(method, "/api/v1/query_range", params=params)
    assert response.status_code == 200
    assert response.json()["data"] == {
        "resultType": "matrix",
        "result": [
            {
                "metric": {
                    "__name__": "promcast_forecast_q50",
                    "target": "space",
                    "instance": "node-a",
                },
                "values": [[160, "50.0"], [220, "60.0"], [280, "70.0"]],
            }
        ],
    }


def test_range_interpolation_and_end(client: TestClient) -> None:
    response = client.post(
        "/api/v1/query_range",
        data={"query": "promcast_forecast_q50", "start": "175", "end": "240", "step": "30"},
    )
    assert response.json()["data"]["result"][0]["values"] == [
        [175, "52.5"],
        [205, "57.5"],
        [235, "62.5"],
    ]


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    ("end", "step", "status"),
    [
        (245.9296875, "0.0078125", 200),
        (245.9375, "0.0078125", 400),
        (220, "1ms", 400),
        (220, "1e-320", 400),
    ],
)
def test_range_point_limit(client: TestClient, method: str, end: float, step: str, status: int):
    params = {"query": "promcast_forecast_q50", "start": "160", "end": str(end), "step": step}
    response = client.request(method, "/api/v1/query_range", params=params)
    assert response.status_code == status
    if status == 200:
        assert len(response.json()["data"]["result"][0]["values"]) == 11_000
    else:
        assert response.json()["errorType"] == "bad_data"
        assert "11,000" in response.json()["error"]


@pytest.mark.parametrize(
    "query",
    [
        "up",
        "sum(promcast_forecast_q50)",
        'promcast_forecast_q50{instance=~".*"}',
        'promcast_forecast_q50{instance!="x"}',
        'promcast_forecast_q50{instance="a",instance="b"}',
        'promcast_forecast_q50{instance="unterminated}',
        "promcast_forecast_q20",
    ],
)
@pytest.mark.parametrize("path", ["query", "query_range"])
def test_unsupported_query(client: TestClient, query: str, path: str) -> None:
    response = client.get(
        f"/api/v1/{path}", params={"query": query, "start": "100", "end": "280", "step": "60"}
    )
    assert response.status_code == 400
    assert response.json() == {
        "status": "error",
        "errorType": "bad_data",
        "error": "promcast serves only promcast_forecast_q10/q50/q90 and promcast_baseline_linear",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"step": "0"},
        {"step": "-1"},
        {"step": "NaN"},
        {"step": "bad"},
        {"start": "inf"},
        {"end": "99"},
        {"start": "1970-01-01"},
    ],
)
def test_invalid_range(client: TestClient, overrides: dict[str, str]) -> None:
    params = {
        "query": "promcast_forecast_q50",
        "start": "100",
        "end": "280",
        "step": "60",
        **overrides,
    }
    response = client.get("/api/v1/query_range", params=params)
    assert response.status_code == 400
    assert response.json()["errorType"] == "bad_data"


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_discovery(client: TestClient, method: str) -> None:
    assert client.request(method, "/api/v1/labels").json()["data"] == [
        "__name__",
        "instance",
        "target",
    ]
    assert client.request(method, "/api/v1/label/instance/values").json()["data"] == ["node-a"]
    assert client.request(method, "/api/v1/label/absent/values").json()["data"] == []
    series = client.request(method, "/api/v1/series").json()["data"]
    assert {item["__name__"] for item in series} == {
        "promcast_forecast_q10",
        "promcast_forecast_q50",
        "promcast_forecast_q90",
        "promcast_baseline_linear",
    }
    params = [
        ("match[]", 'promcast_forecast_q50{instance="node-b"}'),
        ("match[]", "promcast_baseline_linear"),
    ]
    filtered = client.request(method, "/api/v1/series", params=urlencode(params)).json()["data"]
    assert len(filtered) == 1
    assert filtered[0]["__name__"] == "promcast_baseline_linear"
    assert client.get("/api/v1/series", params={"start": "281"}).json()["data"] == []


def test_empty_store_and_missing_parameters() -> None:
    app = FastAPI()
    app.include_router(create_router(Store(), lambda: 190))
    client = TestClient(app)
    assert client.get("/api/v1/labels").json()["data"] == []
    assert (
        client.get("/api/v1/query", params={"query": "promcast_forecast_q50"}).json()["data"][
            "result"
        ]
        == []
    )
    assert (
        client.get("/api/v1/query_range", params={"query": "promcast_forecast_q50"}).status_code
        == 400
    )


def test_linear_target_uses_linear_as_primary(store: Store, client: TestClient) -> None:
    state = store.snapshot()[0]
    target = state.config.model_copy(update={"method": "linear"})
    store.update(target, [state.forecasts[1]], [], state.last_values, 102)
    for metric in ("promcast_forecast_q50", "promcast_baseline_linear"):
        response = client.get("/api/v1/query", params={"query": metric})
        assert response.json()["data"]["result"][0]["value"] == [190, "155.0"]
    store.update(target, [], [], {}, 103)
    assert client.get("/api/v1/labels").json()["data"] == []


def test_selector_matching_and_escaped_label(store: Store, client: TestClient) -> None:
    state = store.snapshot()[0]
    source = state.forecasts[0]
    labels = (("instance", 'node-"a\\b'),)
    forecast = Forecast(
        source.target,
        labels,
        source.origin_ts,
        source.step_s,
        source.quantiles,
        source.missing_fraction,
        source.model,
    )
    store.update(state.config, [forecast], [], {labels: 5}, 102)
    query = 'promcast_forecast_q50{instance="node-\\"a\\\\b"}'
    result = client.get("/api/v1/query", params={"query": query}).json()["data"]["result"]
    assert result[0]["metric"]["instance"] == 'node-"a\\b'
    response = client.get(
        "/api/v1/query", params={"query": 'promcast_forecast_q50{target="other"}'}
    )
    assert response.json()["data"]["result"] == []


def test_internal_failure_does_not_leak_exception_text(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_: object) -> None:
        raise KeyError("/srv/promcast/secret-path")

    monkeypatch.setattr("promcast.shell.compat_api._discovery", explode)
    app = FastAPI()
    app.include_router(create_router(store, lambda: 150.0))
    response = TestClient(app).get("/api/v1/labels")
    assert response.status_code == 400
    assert response.json()["error"] == "invalid request parameters"
    assert "secret-path" not in response.text
