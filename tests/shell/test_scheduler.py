import asyncio
import json
import logging
import threading
from collections.abc import Awaitable
from pathlib import Path

import httpx
import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from promcast.core.config import Config, parse_config
from promcast.core.forecaster import FakeForecaster
from promcast.core.series import FloatArray, Series
from promcast.shell.exposition import Collector
from promcast.shell.scheduler import Clock, CycleResources, Scheduler
from promcast.shell.server import create_app
from promcast.shell.store import Store, TargetSnapshot

EXAMPLE = Path("promcast.example.yaml")
NOW = 1_800_000_123.0


@pytest.fixture
def config() -> Config:
    return parse_config(EXAMPLE.read_text(), environment={})


def matrix(config: Config, *, trailing: int = 0, infinite: bool = False) -> dict[str, object]:
    target = config.targets[0]
    step_s = int(target.step.total_seconds())
    end_ts = NOW // step_s * step_s
    size = int(target.context / target.step)
    values = [
        [end_ts - (size - 1 - index) * step_s, str(1e10 - index * 1e5)]
        for index in range(size - trailing)
    ]
    if infinite:
        values[0][1] = "+Inf"
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [{"metric": {"__name__": "disk", "instance": "node-a"}, "values": values}],
        },
    }


def runtime(config: Config, client: httpx.AsyncClient) -> tuple[Scheduler, CollectorRegistry]:
    store = Store()
    collector = Collector(store, "0.1.0", config.model.name, "test-sha")
    registry = CollectorRegistry()
    registry.register(collector)
    scheduler = Scheduler(
        config, CycleResources(client, FakeForecaster(), store, collector), Clock(wall=lambda: NOW)
    )
    return scheduler, registry


@pytest.mark.parametrize(("direction", "selector"), [("below", "bottomk"), ("above", "topk")])
@pytest.mark.parametrize("max_series", [1, 50])
def test_threshold_candidates_and_exact_cumulative_drops(
    config: Config, direction: str, selector: str, max_series: int
) -> None:
    async def scenario() -> None:
        target = config.targets[0].model_copy(
            update={"query": "free / size", "max_series": max_series, "direction": direction}
        )
        selected_config = config.model_copy(update={"targets": [target]})
        end_ts = int(NOW // 300 * 300)
        queries: list[str] = []
        count_queries: list[str] = []
        candidates = [f"node-{index:03}" for index in range(max_series - 1)] + ["zz-endangered"]

        def respond(request: httpx.Request) -> httpx.Response:
            params = request.url.params
            if request.url.path.endswith("/query"):
                count_queries.append(params["query"])
                assert params["time"] == str(end_ts)
                return count_response(120)
            queries.append(params["query"])
            assert params["query"] == (
                f"(free / size) and {selector}({max_series}, "
                f"last_over_time((free / size)[10m:] @ {end_ts}))"
            )
            start, end = float(params["start"]), float(params["end"])
            step = int(params["step"])
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "resultType": "matrix",
                        "result": [
                            {
                                "metric": {"instance": name},
                                "values": [
                                    [start + index * step, str(100 - index * 0.01)]
                                    for index in range(round((end - start) / step) + 1)
                                ],
                            }
                            for name in candidates
                        ],
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            scheduler, registry = runtime(selected_config, client)
            for cycle in range(1, 3):
                await scheduler.cycle()
                state = scheduler.resources.store.snapshot()[0]
                assert {dict(item.labels)["instance"] for item in state.forecasts} == set(
                    candidates
                )
                assert (
                    registry.get_sample_value(
                        "promcast_series_dropped_total", {"target": target.name}
                    )
                    == (120 - max_series) * cycle
                )
                assert not state.skipped
            assert scheduler.healthy
        assert len(queries) == 2
        assert count_queries == ["count(free / size)"] * 2

    asyncio.run(scenario())


def count_response(total: int | None = 1) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [] if total is None else [{"metric": {}, "value": [NOW, str(total)]}],
            },
        },
    )


def matrix_response(request: httpx.Request, config: Config) -> httpx.Response:
    if request.url.path.endswith("/query"):
        return count_response()
    return httpx.Response(200, json=matrix(config))


def test_audit_scores_history_exports_methods_and_expires_after_fetch_failure(
    config: Config,
) -> None:
    async def scenario() -> None:
        now = NOW
        failing = False
        target = config.targets[0]
        step = int(target.step.total_seconds())
        origin = int(NOW // step * step)

        def respond(request: httpx.Request) -> httpx.Response:
            if failing:
                return httpx.Response(503)
            if request.url.path.endswith("/query"):
                return count_response()
            start = int(request.url.params["start"])
            end = int(request.url.params["end"])
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "resultType": "matrix",
                        "result": [
                            {
                                "metric": {"instance": "node-a"},
                                "values": [
                                    [timestamp, str(1e10 - (timestamp - origin) / step * 1e5)]
                                    for timestamp in range(start, end + 1, step)
                                ],
                            }
                        ],
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            store = Store()
            collector = Collector(store, "test", "fake", "")
            registry = CollectorRegistry()
            registry.register(collector)
            scheduler = Scheduler(
                config,
                CycleResources(client, FakeForecaster(), store, collector),
                Clock(wall=lambda: now),
            )
            labels = {"target": target.name, "lead": "10m"}
            await scheduler.cycle()
            assert registry.get_sample_value("promcast_selfaudit_samples", labels) is None
            now += 600
            await scheduler.cycle()
            assert registry.get_sample_value("promcast_selfaudit_samples", labels) == 1
            assert (
                registry.get_sample_value(
                    "promcast_selfaudit_coverage_ratio", {**labels, "method": "toto"}
                )
                == 0
            )
            assert (
                registry.get_sample_value("promcast_selfaudit_mase", {**labels, "method": "toto"})
                == 2
            )
            assert registry.get_sample_value(
                "promcast_selfaudit_mase", {**labels, "method": "linear"}
            ) == pytest.approx(0)
            await scheduler.cycle()
            assert registry.get_sample_value("promcast_selfaudit_samples", labels) == 1
            now += 86401
            failing = True
            await scheduler.cycle()
            assert registry.get_sample_value("promcast_selfaudit_samples", labels) is None
            assert (
                registry.get_sample_value(
                    "promcast_selfaudit_coverage_ratio", {**labels, "method": "toto"}
                )
                is None
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["http", "transport", "invalid", "negative", "fractional"])
def test_count_failure_preserves_publication_and_redacts_diagnostics(
    config: Config, failure: str, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        failing = False
        cfg = config.model_copy(
            update={
                "tsdb": config.tsdb.model_validate(
                    {
                        **config.tsdb.model_dump(),
                        "headers": {"Authorization": "Bearer private-count-token"},
                    }
                )
            }
        )

        def respond(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer private-count-token"
            assert request.extensions["timeout"] == dict.fromkeys(
                ["connect", "read", "write", "pool"], cfg.tsdb.timeout_s
            )
            if request.url.path.endswith("/query_range"):
                return httpx.Response(200, json=matrix(cfg))
            if not failing:
                return count_response(120)
            if failure == "transport":
                raise httpx.ConnectError("private-count-token")
            if failure == "http":
                return httpx.Response(503, text="prefixprivate-count-tokensuffix")
            if failure == "invalid":
                return httpx.Response(200, text="prefixprivate-count-tokensuffix")
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "resultType": "vector",
                        "result": [
                            {"metric": {}, "value": [NOW, "-1" if failure == "negative" else "1.5"]}
                        ],
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            scheduler, _ = runtime(cfg, client)
            await scheduler.cycle()
            snapshot = scheduler.resources.store.snapshot()
            assert snapshot[0].dropped_total == 70
            failing = True
            for _ in range(3):
                await scheduler.cycle()
                assert scheduler.resources.store.snapshot() == snapshot
            assert not scheduler.healthy
            failing = False
            await scheduler.cycle()
            assert scheduler.healthy
            assert scheduler.resources.store.snapshot()[0].dropped_total == 140
        assert "private-count-token" not in caplog.text
        assert all(
            json.loads(record.message)["status"] == "tsdb_error" for record in caplog.records
        )

    asyncio.run(scenario())


def test_health_metrics_compat_and_transport_failure_retention(config: Config) -> None:
    async def scenario() -> None:
        failing = False
        requests: list[httpx.Request] = []

        def tsdb(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if not failing and request.url.path.endswith("/query"):
                return count_response()
            return (
                httpx.Response(503, text="unavailable")
                if failing
                else httpx.Response(200, json=matrix(config, trailing=2))
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(tsdb)) as client:
            scheduler, registry = runtime(config, client)
            app = create_app(scheduler, registry)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as api:
                assert (await api.get("/healthz")).status_code == 503
                await scheduler.cycle()
                assert (await api.get("/healthz")).status_code == 200
                snapshot = scheduler.resources.store.snapshot()
                actual_origin = NOW // 300 * 300 - 600
                assert snapshot[0].forecasts[0].origin_ts == actual_origin
                assert snapshot[0].last_values[(("instance", "node-a"),)] == 1e10 - 1149 * 1e5
                assert float(requests[0].url.params["end"]) == NOW // 300 * 300
                assert float(requests[0].url.params["start"]) == NOW // 300 * 300 - 1151 * 300
                metrics = await api.get("/metrics")
                assert metrics.headers["content-type"].startswith("text/plain")
                assert (
                    'promcast_forecast_origin_timestamp_seconds{instance="node-a",target="root_disk_free"}'
                    in metrics.text
                )
                assert "promcast_cycle_duration_seconds_count" in metrics.text
                response = await api.get(
                    "/api/v1/query",
                    params={
                        "query": 'promcast_forecast_q50{target="root_disk_free"}',
                        "time": actual_origin + 300,
                    },
                )
                assert response.json()["data"]["result"][0]["value"][0] == actual_origin + 300
                failing = True
                for index in range(3):
                    await scheduler.cycle()
                    assert (await api.get("/healthz")).status_code == (503 if index == 2 else 200)
                    assert scheduler.resources.store.snapshot() == snapshot
                failing = False
                await scheduler.cycle()
                assert (await api.get("/healthz")).status_code == 200

    asyncio.run(scenario())


@pytest.mark.parametrize("empty", [True, False])
def test_skips_or_empty_results_do_not_count_as_success(config: Config, empty: bool) -> None:
    async def scenario() -> None:
        response: dict[str, object] = (
            {"status": "success", "data": {"resultType": "matrix", "result": []}}
            if empty
            else matrix(config, infinite=True)
        )

        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/query"):
                return count_response(None if empty else 1)
            return httpx.Response(200, json=response)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            scheduler, registry = runtime(config, client)
            await scheduler.cycle()
            assert not scheduler.healthy
            assert (
                registry.get_sample_value(
                    "promcast_last_success_timestamp_seconds", {"target": "root_disk_free"}
                )
                is None
            )
            if not empty:
                assert scheduler.resources.store.snapshot()[0].skipped[0].reason == "missing"

    asyncio.run(scenario())


def test_structured_target_logs_and_overrun(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: matrix_response(request, config))
        ) as client:
            scheduler, _ = runtime(config, client)
            times = iter([0.0, 0.0, 121.0, 121.0])
            scheduler = Scheduler(
                config, scheduler.resources, Clock(wall=lambda: NOW, monotonic=lambda: next(times))
            )
            with caplog.at_level(logging.INFO, logger="promcast.shell.scheduler"):
                assert await scheduler.cycle() == 121
            records = [
                json.loads(record.message)
                for record in caplog.records
                if record.name == "promcast.shell.scheduler"
            ]
            assert records[0]["event"] == "target_cycle"
            assert records[0]["status"] == "ok"
            assert records[0]["forecasts"] == 2
            assert records[0]["target"] == "root_disk_free"
            assert records[1] == {
                "event": "cycle_overrun",
                "duration_s": 121.0,
                "next": "promcast bench",
            }

    asyncio.run(scenario())


def test_inference_does_not_block_http_or_overlap(config: Config) -> None:
    async def scenario() -> None:
        entered = threading.Event()
        release = threading.Event()
        thread_ids: list[int] = []

        class BlockingForecaster(FakeForecaster):
            def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
                thread_ids.append(threading.get_ident())
                entered.set()
                assert release.wait(timeout=5)
                return super().forecast(batch, horizon_steps)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: matrix_response(request, config))
        ) as client:
            original, registry = runtime(config, client)
            resources = original.resources
            scheduler = Scheduler(
                config,
                CycleResources(client, BlockingForecaster(), resources.store, resources.collector),
                Clock(wall=lambda: NOW),
            )
            first = asyncio.create_task(scheduler.cycle())
            second = asyncio.create_task(scheduler.cycle())
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=create_app(scheduler, registry)),
                    base_url="http://test",
                ) as api:
                    response = await asyncio.wait_for(api.get("/metrics"), timeout=2)
                    assert response.status_code == 200
                    assert len(thread_ids) == 1
                    assert thread_ids[0] != threading.get_ident()
            finally:
                release.set()
                await asyncio.gather(first, second)
            assert len(thread_ids) == 2

    asyncio.run(scenario())


def test_lifespan_starts_immediately_and_stops_during_wait(config: Config) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: matrix_response(request, config))
        ) as client:
            scheduler, registry = runtime(config, client)
            app = create_app(scheduler, registry)
            async with app.router.lifespan_context(app):

                async def ready() -> None:
                    while not scheduler.healthy:
                        await asyncio.sleep(0)

                await asyncio.wait_for(ready(), timeout=3)
            assert not scheduler.healthy

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_nonfinite_model_output_is_skipped_and_logged(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    class NonfiniteForecaster(FakeForecaster):
        def forecast(self, batch: list[Series], horizon_steps: int) -> list[FloatArray]:
            return [np.full((9, horizon_steps), np.nan) for _ in batch]

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: matrix_response(request, config))
        ) as client:
            original, _ = runtime(config, client)
            resources = original.resources
            scheduler = Scheduler(
                config,
                CycleResources(client, NonfiniteForecaster(), resources.store, resources.collector),
                Clock(wall=lambda: NOW),
            )
            await scheduler.cycle()
            assert not scheduler.healthy
            assert scheduler.resources.store.snapshot()[0].skipped[0].reason == "model_output"
            assert '"reason": "model_output"' in caplog.text

    asyncio.run(scenario())


def test_malformed_sample_records_failure_and_loop_recovers(
    config: Config, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        stop = asyncio.Event()
        cycles = 0
        scheduler: Scheduler
        snapshot: tuple[TargetSnapshot, ...] = ()
        original_wait_for = asyncio.wait_for

        async def wait_for(awaitable: Awaitable[bool], timeout: float) -> bool:
            nonlocal cycles, snapshot
            cycles += 1
            state = scheduler.resources.store.snapshot()
            if cycles == 1:
                snapshot = state
            elif cycles < 5:
                assert state == snapshot
                assert scheduler.healthy == (cycles < 4)
            else:
                assert scheduler.healthy
                stop.set()
                return await awaitable
            return await original_wait_for(awaitable, timeout=0)

        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/query"):
                return count_response()
            body = json.dumps(matrix(config))
            if 1 <= cycles < 4:
                body = body.replace('"10000000000.0"', '"invalid-private-sample"')
            return httpx.Response(200, content=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            scheduler, _ = runtime(config, client)
            monkeypatch.setattr(asyncio, "wait_for", wait_for)
            await scheduler.run(stop)
        assert cycles == 5
        failures = [json.loads(record.message) for record in caplog.records]
        assert len(failures) == 3
        assert all(record["status"] == "invalid_data" for record in failures)
        assert "invalid-private-sample" not in caplog.text

    asyncio.run(scenario())
