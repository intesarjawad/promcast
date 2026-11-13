"""Record the pinned Grafana backend's Prometheus requests for compat API tests."""

import asyncio
import json
import logging
import shutil
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TypedDict
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import httpx

GRAFANA_IMAGE = "grafana/grafana:13.3.0"
FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/grafana_calls.json"
START_SECONDS = 1793750400  # 2026-11-04T00:00:00Z, fixed for replayable panel requests.
END_SECONDS = START_SECONDS + 3600
STARTUP_TIMEOUT_SECONDS = 120
type Json = bool | int | float | str | list[Json] | dict[str, Json] | None


class Call(TypedDict):
    """Preserve parameter placement and repeated values for HTTP replay."""

    method: str
    path: str
    query_params: dict[str, list[str]]
    form_params: dict[str, list[str]]


def prometheus_response(path: str, params: dict[str, list[str]]) -> dict[str, Json]:
    """Answer the health expression and synthetic forecast queries as Prometheus."""
    if path == "/api/v1/status/buildinfo":
        return {
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
    if path == "/api/v1/query":
        timestamp = float(params.get("time", [str(START_SECONDS)])[0])
        return {
            "status": "success",
            "data": {"resultType": "scalar", "result": [timestamp, "2"]},
        }
    if path == "/api/v1/query_range":
        start = float(params["start"][0])
        end = float(params["end"][0])
        return {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "promcast_forecast_q50"},
                        "values": [[start, "1"], [end, "1"]],
                    }
                ],
            },
        }
    return {"status": "success", "data": []}


@contextmanager
def prometheus_stub(calls: list[Call]) -> Generator[int]:
    """Expose a synthetic datasource to Docker through host.docker.internal."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.respond()

        def do_POST(self) -> None:
            self.respond()

        def log_message(self, *_args: object, **_kwargs: object) -> None:
            return

        def respond(self) -> None:
            target = urlsplit(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
            query_params = parse_qs(target.query, keep_blank_values=True)
            form_params = parse_qs(body, keep_blank_values=True)
            calls.append(
                {
                    "method": self.command,
                    "path": target.path,
                    "query_params": query_params,
                    "form_params": form_params,
                }
            )
            payload = json.dumps(
                prometheus_response(target.path, query_params | form_params)
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


async def docker(*arguments: str) -> str:
    """Run Docker without a shell and propagate daemon or container failures."""
    executable = shutil.which("docker")
    if executable is None:
        raise RuntimeError("Docker CLI is required for the Grafana probe")
    process = await asyncio.create_subprocess_exec(
        executable, *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Docker {arguments[0]} failed: {stderr.decode().strip()}")
    return stdout.decode().strip()


def wait_for_grafana(client: httpx.Client) -> None:
    """Require the pinned server version before recording any datasource calls."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            response = client.get("/api/health")
        except httpx.TransportError:
            time.sleep(0.5)
            continue
        if response.is_success:
            if response.json()["version"] != "13.3.0":
                raise RuntimeError(f"Unexpected Grafana version: {response.text}")
            return
        time.sleep(0.5)
    raise TimeoutError("Grafana did not become healthy within 120 seconds")


def exercise_grafana(client: httpx.Client, stub_port: int) -> None:
    """Save a minimal dashboard and execute its target through the datasource API."""
    datasource = {"type": "prometheus", "uid": "promcast"}
    response = client.post(
        "/api/datasources",
        json={
            **datasource,
            "name": "Promcast probe",
            "access": "proxy",
            "url": f"http://host.docker.internal:{stub_port}",
            "jsonData": {"httpMethod": "POST", "timeInterval": "1m"},
        },
    )
    response.raise_for_status()
    target = {
        "refId": "A",
        "datasource": datasource,
        "expr": "promcast_forecast_q50",
        "range": True,
        "instant": False,
        "format": "time_series",
        "intervalMs": 60000,
        "maxDataPoints": 100,
    }
    response = client.post(
        "/api/dashboards/db",
        json={
            "dashboard": {
                "uid": "promcast-probe",
                "title": "Promcast probe",
                "schemaVersion": 39,
                "panels": [
                    {
                        "id": 1,
                        "title": "Forecast",
                        "type": "timeseries",
                        "datasource": datasource,
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
                        "targets": [target],
                    }
                ],
            },
            "overwrite": True,
        },
    )
    response.raise_for_status()
    response = client.get("/api/datasources/uid/promcast/health")
    response.raise_for_status()
    if response.json()["status"] != "OK":
        raise RuntimeError(f"Datasource health failed: {response.text}")
    response = client.post(
        "/api/ds/query",
        json={
            "from": str(START_SECONDS * 1000),
            "to": str(END_SECONDS * 1000),
            "queries": [target],
        },
    )
    response.raise_for_status()
    result = response.json()["results"]["A"]
    if result.get("error") or not result.get("frames"):
        raise RuntimeError(f"Panel query failed: {response.text}")


async def main() -> None:
    """Replace the replay fixture only after Grafana health and panel checks pass."""
    calls: list[Call] = []
    name = f"promcast-grafana-probe-{uuid4().hex}"
    with prometheus_stub(calls) as stub_port:
        await docker(
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--publish",
            "127.0.0.1::3000",
            "--add-host",
            "host.docker.internal:host-gateway",
            "--env",
            "GF_AUTH_ANONYMOUS_ENABLED=true",
            "--env",
            "GF_AUTH_ANONYMOUS_ORG_ROLE=Admin",
            "--env",
            "GF_ANALYTICS_REPORTING_ENABLED=false",
            "--env",
            "GF_ANALYTICS_CHECK_FOR_UPDATES=false",
            "--env",
            "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES=false",
            GRAFANA_IMAGE,
        )
        try:
            address = (await docker("port", name, "3000/tcp")).splitlines()[0]
            with httpx.Client(base_url=f"http://{address}", timeout=10, trust_env=False) as client:
                wait_for_grafana(client)
                exercise_grafana(client, stub_port)
        except Exception:
            logging.exception(
                "Grafana probe failed; container logs: %s", await docker("logs", name)
            )
            raise
        finally:
            await docker("rm", "--force", name)
    paths = {call["path"] for call in calls}
    health_recorded = any(
        call["path"] == "/api/v1/query"
        and (call["query_params"] | call["form_params"]).get("query") == ["1+1"]
        for call in calls
    )
    if not health_recorded or not {"/api/v1/status/buildinfo", "/api/v1/query_range"} <= paths:
        raise RuntimeError(f"Required Grafana calls missing: {calls}")
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(calls, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
