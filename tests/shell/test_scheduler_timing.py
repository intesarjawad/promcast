import asyncio
from collections.abc import Awaitable
from pathlib import Path

import httpx
import pytest

from promcast.core.config import parse_config
from promcast.core.forecaster import FakeForecaster
from promcast.shell.exposition import Collector
from promcast.shell.scheduler import Clock, CycleResources, Scheduler
from promcast.shell.store import Store


def count_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "success",
            "data": {"resultType": "vector", "result": [{"metric": {}, "value": [0, "1"]}]},
        },
    )


@pytest.mark.parametrize("elapsed_s", [30.0, 121.0])
def test_loop_waits_remaining_interval_and_restarts_overrun_without_catchup(
    elapsed_s: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        config = parse_config(Path("promcast.example.yaml").read_text(), environment={})
        stop = asyncio.Event()
        waits: list[float] = []
        requests: list[httpx.Request] = []
        count_requests: list[httpx.Request] = []
        original_wait_for = asyncio.wait_for

        async def wait_for(awaitable: Awaitable[bool], timeout: float) -> bool:
            waits.append(timeout)
            if timeout > 0:
                stop.set()
            return await original_wait_for(awaitable, timeout=timeout)

        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/query"):
                count_requests.append(request)
                return count_response()
            requests.append(request)
            if len(requests) == 2:
                stop.set()
            return httpx.Response(
                200, json={"status": "success", "data": {"resultType": "matrix", "result": []}}
            )

        times = iter(
            [0.0, 0.0, elapsed_s, elapsed_s, elapsed_s, elapsed_s, 2 * elapsed_s, 2 * elapsed_s]
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            store = Store()
            resources = CycleResources(
                client, FakeForecaster(), store, Collector(store, "test", "fake", "")
            )
            scheduler = Scheduler(
                config,
                resources,
                Clock(wall=lambda: 1_800_000_000.0, monotonic=lambda: next(times)),
            )
            monkeypatch.setattr(asyncio, "wait_for", wait_for)
            await scheduler.run(stop)
        assert waits == ([90.0] if elapsed_s == 30 else [0.0, 0.0])
        assert len(requests) == len(waits)
        assert len(count_requests) == len(waits)
        assert not scheduler.healthy

    asyncio.run(scenario())


def test_successful_target_does_not_mask_another_targets_failures() -> None:
    async def scenario() -> None:
        config = parse_config(
            Path("promcast.example.yaml").read_text().replace("context: 4d", "context: 320m"),
            environment={},
        )
        target = config.targets[0]
        config.targets.append(target.model_copy(update={"name": "other", "query": "other"}))
        failing = False

        def respond(request: httpx.Request) -> httpx.Response:
            if failing and request.url.params["query"].startswith(f"({target.query}) and"):
                return httpx.Response(500)
            if request.url.path.endswith("/query"):
                return count_response()
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "resultType": "matrix",
                        "result": [
                            {
                                "metric": {},
                                "values": [
                                    [1_800_000_000 - (63 - index) * 300, str(100 - index)]
                                    for index in range(64)
                                ],
                            }
                        ],
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            store = Store()
            resources = CycleResources(
                client, FakeForecaster(), store, Collector(store, "test", "fake", "")
            )
            scheduler = Scheduler(config, resources, Clock(wall=lambda: 1_800_000_000.0))
            await scheduler.cycle()
            assert scheduler.healthy
            failing = True
            for _ in range(3):
                await scheduler.cycle()
            assert not scheduler.healthy
            assert len(store.snapshot()[1].forecasts) == 2
            failing = False
            await scheduler.cycle()
            assert scheduler.healthy

    asyncio.run(scenario())
