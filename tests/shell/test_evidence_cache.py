import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from promcast.core.evidence import (
    SeriesItem,
    record_path,
    render_markdown,
    select_shard,
    site_document,
)
from promcast.core.forecaster import FakeForecaster
from promcast.core.replay import ReplaySpec
from promcast.shell.evidence import (
    CacheOptions,
    ProtocolError,
    build_report,
    load_cached,
    read_cached,
    run_items,
    verify_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("toto-2.0-22m", "linear", "seasonal-naive", "ets", "static-margin")


class _Model(FakeForecaster):
    name = "toto-2.0-22m"


def _item(series_id: str, offset: float = 0.0) -> SeriesItem:
    values = np.concatenate([np.full(60, 10.0) - offset, np.linspace(9.0, -9.0, 140) - offset])
    spec = ReplaySpec("below", 64, 8, 60, 5, 24, threshold=0, budget_per_week=1e6)
    return SeriesItem("e1", series_id, values, spec, ("synthetic test",))


def _git(root: Path, *arguments: str) -> None:
    executable = shutil.which("git")
    assert executable is not None
    subprocess.run(  # noqa: S603
        [executable, "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def tagged_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    document = root / "docs/protocol.md"
    document.write_text("protocol\n", encoding="utf-8")
    executable = shutil.which("git")
    assert executable is not None
    subprocess.run(  # noqa: S603
        [executable, "init", str(root)], check=True, capture_output=True, text=True
    )
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "commit.gpgSign", "false")
    _git(root, "config", "tag.gpgSign", "false")
    _git(root, "add", "docs/protocol.md")
    _git(root, "commit", "-m", "protocol")
    _git(root, "tag", "protocol-v1")
    return root, document


def test_protocol_passes_until_one_byte_changes(tagged_repo: tuple[Path, Path]):
    root, document = tagged_repo
    verify_protocol(root)
    document.write_text("protocol!\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="changed since protocol-v1"):
        verify_protocol(root)


def test_missing_protocol_tag_is_actionable(tmp_path: Path):
    with pytest.raises(ProtocolError, match="tag protocol-v1 is missing"):
        verify_protocol(tmp_path)


def test_deleted_protocol_is_refused_even_when_tagged(tagged_repo: tuple[Path, Path]):
    root, document = tagged_repo
    document.unlink()
    with pytest.raises(ProtocolError, match=r"docs/protocol\.md is missing"):
        verify_protocol(root)


def test_protocol_freezes_the_replay_decision_record_verbatim():
    protocol = (ROOT / "docs/protocol.md").read_text(encoding="utf-8")
    decision = (ROOT / "docs/adr/0005-replay-protocol.md").read_text(encoding="utf-8")
    assert decision in protocol
    assert "Pre-registered decision rule (primary endpoint)" in protocol
    assert "| E3 NAB | 5m (native) | 576 (2 d) | 24 (2 h) | 288 | 1 | auto |" in protocol


def test_two_shards_merge_byte_identically_to_the_unsharded_run(tmp_path: Path):
    items = [_item("a"), _item("b", 1.0)]
    models = {"toto-2.0-22m": _Model()}
    unsharded = run_items(
        items,
        METHODS,
        models,
        CacheOptions(tmp_path / "cache-unsharded"),
        timer=lambda: 2.0,
    )
    for index in range(2):
        run_items(
            select_shard(items, index, 2),
            METHODS,
            models,
            CacheOptions(tmp_path / "cache-sharded"),
            timer=lambda: 2.0,
        )
    merged = load_cached(items, METHODS, tmp_path / "cache-sharded")
    unsharded_report = build_report(unsharded, ["e1"], 1.0, tmp_path / "data")
    merged_report = build_report(merged, ["e1"], 1.0, tmp_path / "data")
    assert render_markdown(unsharded_report, "cpu") == render_markdown(merged_report, "cpu")
    assert site_document(unsharded_report) == site_document(merged_report)


def test_resume_reuses_records_and_recomputes_only_a_corrupt_one(tmp_path: Path):
    items = [_item("a"), _item("b", 1.0)]
    models = {"toto-2.0-22m": _Model()}
    cache = CacheOptions(tmp_path / "cache")
    first = run_items(items, METHODS, models, cache, timer=lambda: 1.0)
    calls = 0

    def timer() -> float:
        nonlocal calls
        calls += 1
        return 1.0

    second = run_items(items, METHODS, models, CacheOptions(cache.directory, True), timer=timer)
    assert calls == 0
    assert [record.series_id for record in second] == [record.series_id for record in first]
    path = record_path(cache.directory, "e1", "a")
    path.write_text("{", encoding="utf-8")
    third = run_items(items, METHODS, models, CacheOptions(cache.directory, True), timer=timer)
    assert calls == 2
    assert len(third) == 2


def test_cache_is_invalidated_when_the_method_set_changes(tmp_path: Path):
    item = _item("a")
    run_items(
        [item],
        METHODS,
        {"toto-2.0-22m": _Model()},
        CacheOptions(tmp_path / "cache"),
        timer=lambda: 1.0,
    )
    path = record_path(tmp_path / "cache", "e1", "a")
    assert read_cached(path, item, METHODS) is not None
    assert read_cached(path, item, ("linear",)) is None
