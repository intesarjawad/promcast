import asyncio
import importlib.util
import math
import shutil
import sys
from http.client import HTTPConnection
from io import StringIO
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING

import pytest
import yaml
from prometheus_client.parser import text_string_to_metric_families

ANCHOR = 1_800_000_000.125
SCRIPT = Path(__file__).resolve().parents[2] / "demo" / "scenario.py"

if TYPE_CHECKING:
    from demo import scenario
else:
    spec = importlib.util.spec_from_file_location("scenario", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    scenario = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scenario)


def samples(text: str) -> dict[str, list[tuple[float, float | None]]]:
    result: dict[str, list[tuple[float, float | None]]] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        result.setdefault(fields[0].split("{")[0], []).append(
            (float(fields[1]), float(fields[2]) if len(fields) == 3 else None)
        )
    return result


def assert_equal_exports(text: str):
    parsed = samples(text)
    assert parsed["demo_disk_free_bytes"] == parsed["node_filesystem_avail_bytes"]
    assert all(value == 500_000_000_000 for value, _ in parsed["node_filesystem_size_bytes"])


def test_anchor_is_continuous():
    at_anchor = scenario.disk_free_bytes(ANCHOR, ANCHOR)
    assert at_anchor > 0
    assert at_anchor < scenario.no_leak_bytes(ANCHOR, ANCHOR)
    for delta in (-0.000001, 0.000001):
        assert scenario.disk_free_bytes(ANCHOR + delta, ANCHOR) == pytest.approx(
            at_anchor, abs=1000
        )


def test_backfill_labels_equal_scraped_live_labels():
    compose = yaml.safe_load((SCRIPT.parents[1] / "docker-compose.yml").read_text())
    config = yaml.safe_load(compose["configs"]["prometheus"]["content"])
    job = next(job for job in config["scrape_configs"] if job["job_name"] == "scenario")
    scrape_labels = {
        "job": job["job_name"],
        "instance": job["static_configs"][0]["targets"][0],
    }
    history = StringIO()
    scenario.write_history(history, ANCHOR, 1)
    backfilled = {
        (sample.name, tuple(sorted(sample.labels.items())))
        for family in text_string_to_metric_families(history.getvalue())
        for sample in family.samples
    }
    scraped = {
        (sample.name, tuple(sorted({**sample.labels, **scrape_labels}.items())))
        for family in text_string_to_metric_families(scenario.metrics(ANCHOR, ANCHOR).decode())
        for sample in family.samples
    }
    assert len(backfilled) == len(scraped) == 3
    assert backfilled == scraped
    filesystem_labels = {"mountpoint": "/demo", "fstype": "ext4", "device": "demo"}
    assert scraped == {
        (
            name,
            tuple(sorted({**scrape_labels, **labels}.items())),
        )
        for name, labels in (
            ("demo_disk_free_bytes", {}),
            ("node_filesystem_avail_bytes", filesystem_labels),
            ("node_filesystem_size_bytes", filesystem_labels),
        )
    }


def test_no_leak_before_first_start():
    for minute in range(-72 * 60, -24):
        timestamp = ANCHOR + minute * 60
        assert scenario.disk_free_bytes(timestamp, ANCHOR) == scenario.no_leak_bytes(
            timestamp, ANCHOR
        )


@pytest.mark.parametrize("cycle", range(5))
def test_leak_full_hold_and_cleanup(cycle: int):
    start = ANCHOR - 25 * 60 + cycle * 3 * 3600
    assert scenario.disk_free_bytes(start, ANCHOR) == scenario.no_leak_bytes(start, ANCHOR)
    assert scenario.disk_free_bytes(start - 0.000001, ANCHOR) == pytest.approx(
        scenario.disk_free_bytes(start + 0.000001, ANCHOR), abs=1000
    )
    for minute in (1, 25, 64):
        timestamp = start + minute * 60
        assert (
            0
            < scenario.disk_free_bytes(timestamp, ANCHOR)
            < scenario.no_leak_bytes(timestamp, ANCHOR)
        )
    for age in (65 * 60, 70 * 60, 75 * 60 - 0.001):
        assert scenario.disk_free_bytes(start + age, ANCHOR) == 0
    for minute in (75, 76, 179, 180):
        timestamp = start + minute * 60
        assert scenario.disk_free_bytes(timestamp, ANCHOR) == scenario.no_leak_bytes(
            timestamp, ANCHOR
        )


def test_second_leak_starts_after_exactly_three_hours():
    first_start = ANCHOR - 25 * 60
    second_start = first_start + 3 * 3600
    assert scenario.disk_free_bytes(second_start, ANCHOR) == scenario.no_leak_bytes(
        second_start, ANCHOR
    )
    for elapsed in (0.001, 60, 25 * 60, 64 * 60):
        first = first_start + elapsed
        second = second_start + elapsed
        assert scenario.disk_free_bytes(second, ANCHOR) / scenario.no_leak_bytes(
            second, ANCHOR
        ) == pytest.approx(
            scenario.disk_free_bytes(first, ANCHOR) / scenario.no_leak_bytes(first, ANCHOR)
        )
        assert scenario.disk_free_bytes(second, ANCHOR) < scenario.no_leak_bytes(second, ANCHOR)


def test_backfill_minute_grid_and_export_equality(tmp_path: Path):
    out = tmp_path / "work" / "history.om"
    anchor_file = out.with_name("anchor")
    scenario.backfill(out, anchor_file, 72, ANCHOR)
    assert float(anchor_file.read_text()) == ANCHOR
    text = out.read_text()
    assert text.endswith("# EOF\n")
    assert_equal_exports(text)
    free = samples(text)["demo_disk_free_bytes"]
    assert len(free) == 72 * 60 + 1
    assert [timestamp for _, timestamp in free] == [
        ANCHOR + minute * 60 for minute in range(-72 * 60, 1)
    ]
    assert free[-1][0] == scenario.disk_free_bytes(ANCHOR, ANCHOR)
    assert (
        free[-1][0]
        == samples(scenario.metrics(ANCHOR, ANCHOR).decode())["demo_disk_free_bytes"][0][0]
    )


def test_backfill_preserves_existing_history_and_anchor(tmp_path: Path):
    history = tmp_path / "history.om"
    anchor = tmp_path / "anchor"
    scenario.backfill(history, anchor, 1, ANCHOR)
    original = history.read_bytes(), anchor.read_bytes()
    scenario.backfill(history, anchor, 1, ANCHOR + 3600)
    assert (history.read_bytes(), anchor.read_bytes()) == original


def test_backfill_rejects_incomplete_history(tmp_path: Path):
    history = tmp_path / "history.om"
    history.write_text("partial")
    with pytest.raises(ValueError, match="incomplete"):
        scenario.backfill(history, tmp_path / "anchor", 1, ANCHOR)
    assert history.read_text() == "partial"


def test_backfill_rejects_truncated_pair(tmp_path: Path):
    history = tmp_path / "history.om"
    anchor = tmp_path / "anchor"
    history.write_text("partial")
    anchor.write_text("")
    with pytest.raises(ValueError, match="incomplete"):
        scenario.backfill(history, anchor, 1, ANCHOR)
    assert history.read_text() == "partial"


def test_live_http_uses_shared_anchor():
    for offset in (0, 40 * 60, 51 * 60, 3 * 3600):
        timestamp = ANCHOR + offset
        with scenario.create_server(
            ("127.0.0.1", 0), ANCHOR, lambda timestamp=timestamp: timestamp
        ) as server:
            thread = Thread(target=server.handle_request)
            thread.start()
            try:
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                connection.request("GET", "/metrics")
                response = connection.getresponse()
                body = response.read().decode()
                assert response.status == 200
                assert "application/openmetrics-text" in str(response.getheader("Content-Type"))
                assert body.endswith("# EOF\n")
                assert_equal_exports(body)
                assert samples(body)["demo_disk_free_bytes"][0] == (
                    scenario.disk_free_bytes(timestamp, ANCHOR),
                    None,
                )
                connection.close()
            finally:
                thread.join(timeout=5)
            assert not thread.is_alive()


def test_cli_backfill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = tmp_path / "history.om"
    anchor_file = tmp_path / "anchor"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--backfill",
            "--hours",
            "1",
            "--out",
            str(output),
            "--anchor-file",
            str(anchor_file),
        ],
    )
    scenario.main()
    assert math.isfinite(float(anchor_file.read_text()))
    assert len(samples(output.read_text())["demo_disk_free_bytes"]) == 61


def test_invalid_hours_do_not_write_files(tmp_path: Path):
    with pytest.raises(ValueError, match="positive"):
        scenario.backfill(tmp_path / "history.om", tmp_path / "anchor", 0, ANCHOR)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="positive"):
        scenario.write_history(StringIO(), ANCHOR, -1)


def test_promtool_creates_block_from_one_hour(tmp_path: Path):
    promtool = shutil.which("promtool")
    assert promtool is not None, "promtool must be installed to validate demo history"
    history = tmp_path / "history.om"
    scenario.backfill(history, tmp_path / "anchor", 1, ANCHOR)
    blocks = tmp_path / "blocks"

    async def import_history() -> tuple[int | None, bytes, bytes]:
        process = await asyncio.create_subprocess_exec(
            promtool,
            "tsdb",
            "create-blocks-from",
            "openmetrics",
            str(history),
            str(blocks),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout, stderr

    returncode, stdout, stderr = asyncio.run(import_history())
    assert returncode == 0, (stdout + stderr).decode()
    assert list(blocks.glob("*/meta.json"))
