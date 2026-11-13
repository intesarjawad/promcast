import json
import re
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

import numpy as np
import pytest

from promcast.core import scenarios
from promcast.core.evidence import ESTIMATE_CONTEXT, ESTIMATE_CYCLE
from promcast.core.replay import ReplaySpec
from promcast.core.series import Series, Skipped, align
from promcast.shell import sources
from promcast.shell.datasets import write_sums
from promcast.shell.evidence import series_items

ROOT = Path(__file__).resolve().parents[2]
_STEP_SECONDS = {"1m": 60, "5m": 300}


class _Table(Protocol):
    @property
    def schema(self) -> object: ...


@runtime_checkable
class _Arrow(Protocol):
    def table(self, data: dict[str, object]) -> _Table: ...


class _Writer(Protocol):
    def write_table(self, table: _Table) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class _IPC(Protocol):
    def new_stream(self, sink: BinaryIO, schema: object) -> _Writer: ...


def _parameter_table() -> tuple[str, dict[str, dict[str, str]]]:
    text = (ROOT / "docs/protocol.md").read_text(encoding="utf-8")
    section = text.split("## 2. Parameters per corpus", 1)[1].split("\n## ", 1)[0]
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.splitlines()
        if line.startswith("|") and not line.startswith("|---")
    ]
    header, body = rows[0], rows[1:]
    table = {row[0].split()[0].lower(): dict(zip(header, row, strict=True)) for row in body}
    return section, table


def _leading_int(cell: str) -> int:
    match = re.match(r"\d+", cell)
    assert match is not None, cell
    return int(match.group())


def _assert_common(row: dict[str, str], spec: ReplaySpec) -> None:
    assert spec.context_steps == _leading_int(row["context"])
    assert spec.horizon_steps == _leading_int(row["horizon"])
    assert spec.cycle_steps == _leading_int(row["cycle"])
    assert spec.for_cycles == 2
    if row["threshold"].startswith("auto"):
        assert spec.threshold is None
    if not row["step"].startswith("native"):
        assert spec.step_s == _STEP_SECONDS[row["step"].split()[0]]
        assert spec.season_steps == _leading_int(row["season"])


def _five_minute_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"2014-01-01 00:{minute:02d}:00,{minute}" for minute in range(0, 60, 5)]
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _boom_folder(folder: Path) -> None:
    arrow: object = import_module("pyarrow")
    ipc: object = import_module("pyarrow.ipc")
    assert isinstance(arrow, _Arrow)
    assert isinstance(ipc, _IPC)
    folder.mkdir(parents=True)
    taxonomy = {"fine": {"type": ["gauge"]}, "five": {"type": ["gauge"]}}
    (folder / "dataset_taxonomy.json").write_text(json.dumps(taxonomy), encoding="utf-8")
    table = arrow.table(
        {
            "start": [datetime(2020, 1, 1, tzinfo=UTC)] * 2,
            "item_id": ["fine", "five"],
            "freq": ["10S", "5T"],
            "target": [[list(range(100))], [list(range(100))]],
        }
    )
    with (folder / "data.arrow").open("wb") as stream:
        writer = ipc.new_stream(stream, table.schema)
        writer.write_table(table)
        writer.close()


def test_every_corpus_uses_the_frozen_parameter_table(tmp_path: Path):
    section, table = _parameter_table()
    assert sorted(table) == ["e1", "e2", "e3", "e4"]
    default = re.search(r"direction `(\w+)` unless stated", section)
    assert default is not None
    default_direction = default.group(1)

    e1 = table["e1"]
    assert e1["threshold"].startswith("per scenario, defined in `scenarios.py`")
    assert Path(scenarios.__file__).name == "scenarios.py"
    items = series_items("e1", tmp_path, 0)
    assert [item.series_id for item in items] == list(scenarios.SCENARIOS)
    suite = [*scenarios.SCENARIOS, *scenarios.REFUSED_SCENARIOS]
    assert len(set(suite)) == len(suite) == _leading_int(e1["size cap"])
    for item in items:
        _assert_common(e1, item.spec)
        assert (item.spec.direction, item.spec.threshold) == scenarios.SCENARIOS[item.series_id]

    e2 = table["e2"]
    assert e2["step"] == "native `freq`"
    assert e2["season"].startswith("1 d in steps if ≤ 2048, else none")
    assert e2["threshold"] == "auto"
    _boom_folder(tmp_path / "BOOM")
    boom = {item.series_id: item.spec for item in series_items("e2", tmp_path, 0)}
    for spec in boom.values():
        _assert_common(e2, spec)
        assert spec.direction == default_direction
    assert (boom["fine"].step_s, boom["fine"].season_steps) == (10, 0)
    assert (boom["five"].step_s, boom["five"].season_steps) == (300, 288)

    assert table["e3"]["threshold"] == "auto"
    nab_csv = tmp_path / "NAB/data/realAWSCloudwatch/example.csv"
    _five_minute_csv(nab_csv)
    write_sums(tmp_path / "NAB", [nab_csv])
    e3_items = series_items("e3", tmp_path, 0)
    assert len(e3_items) == 1
    _assert_common(table["e3"], e3_items[0].spec)
    assert e3_items[0].spec.direction == default_direction

    e4 = table["e4"]
    override = re.fullmatch(r"auto \(`(\w+)` for (\w+) and (\w+)\)", e4["threshold"])
    assert override is not None
    override_direction, *keywords = override.groups()
    for name in ("node_filesystem_avail_bytes", "node_memory_MemAvailable_bytes", "node_load1"):
        _five_minute_csv(tmp_path / f"e4/data/{name}.csv")
    e4_items = series_items("e4", tmp_path, 0)
    assert len(e4_items) == 3
    for item in e4_items:
        _assert_common(e4, item.spec)
        overridden = any(keyword in item.series_id for keyword in keywords)
        expected = override_direction if overridden else default_direction
        assert item.spec.direction == expected

    for corpus, row in table.items():
        assert ESTIMATE_CONTEXT[corpus] == _leading_int(row["context"])
        assert ESTIMATE_CYCLE[corpus] == _leading_int(row["cycle"])


def test_corpus_gap_rule_is_the_live_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    section, _ = _parameter_table()
    rule = next(line for line in section.splitlines() if line.startswith("Gaps (E3 and E4):"))
    assert "`promcast.core.series.align`" in rule
    calls: list[tuple[int, float, int]] = []

    def spy(
        matrix: dict[str, object], step_s: int, end_ts: float, context_steps: int
    ) -> tuple[list[Series], list[Skipped]]:
        calls.append((step_s, end_ts, context_steps))
        return align(matrix, step_s, end_ts, context_steps)

    monkeypatch.setattr(sources, "align", spy)
    path = tmp_path / "NAB/data/realAWSCloudwatch/example.csv"
    path.parent.mkdir(parents=True)
    path.write_text("timestamp,value\n2014-01-01 00:00:00,1\n2014-01-01 00:10:00,2\n")
    write_sums(tmp_path / "NAB", [path])
    (item,) = series_items("e3", tmp_path, 0)
    end_ts = datetime(2014, 1, 1, 0, 10, tzinfo=UTC).timestamp()
    assert calls == [(300, end_ts, 3)]
    assert np.isnan(item.values[1])
