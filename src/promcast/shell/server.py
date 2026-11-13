import asyncio
import ssl
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import version

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from promcast.core.config import Config
from promcast.core.forecaster import Forecaster
from promcast.shell.compat_api import create_router
from promcast.shell.exposition import Collector
from promcast.shell.scheduler import CycleResources, Scheduler
from promcast.shell.store import Store


def create_app(scheduler: Scheduler, registry: CollectorRegistry) -> FastAPI:
    """Serve health, exposition and compat queries while one scheduler runs."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        stop = asyncio.Event()
        task = asyncio.create_task(scheduler.run(stop))
        try:
            yield
        finally:
            stop.set()
            await task

    app = FastAPI(lifespan=lifespan)
    app.include_router(create_router(scheduler.resources.store))

    @app.get("/healthz")
    async def health() -> JSONResponse:
        ready = scheduler.healthy
        return JSONResponse(
            {"status": "ok" if ready else "unavailable"}, status_code=200 if ready else 503
        )

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return app


async def serve(config: Config, forecaster: Forecaster, model_sha: str) -> None:
    """Own transport and server lifetimes on the caller's asyncio loop."""
    host, _, port_text = config.server.listen.rpartition(":")
    port = int(port_text)
    if not host or not 0 < port < 65536:
        raise ValueError("server.listen must be host:port with port between 1 and 65535")
    ca_file = config.tsdb.tls.ca_file
    trust = ssl.create_default_context(cafile=str(ca_file.expanduser())) if ca_file else True
    store = Store()
    collector = Collector(store, version("promcast"), config.model.name, model_sha)
    registry = CollectorRegistry()
    registry.register(collector)
    async with httpx.AsyncClient(verify=trust) as client:
        scheduler = Scheduler(config, CycleResources(client, forecaster, store, collector))
        app = create_app(scheduler, registry)
        server = uvicorn.Server(uvicorn.Config(app, host=host.strip("[]"), port=port))
        await server.serve()
