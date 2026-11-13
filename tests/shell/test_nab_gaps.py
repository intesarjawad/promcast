import hashlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from promcast.shell.datasets import FileLock, LockedFile

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import nab_gaps
else:
    spec = importlib.util.spec_from_file_location("nab_gaps", ROOT / "scripts/nab_gaps.py")
    assert spec is not None
    assert spec.loader is not None
    nab_gaps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nab_gaps)

GAPPY = (
    "timestamp,value\n"
    "2014-01-01 00:00:00,1\n"
    "2014-01-01 00:07:00,9\n"
    "2014-01-01 00:15:00,2\n"
    "2014-01-01 00:15:00,3\n"
    "2014-01-01 00:20:00,4\n"
)


def test_gaps_counts_missing_runs_duplicates_and_off_grid_rows(tmp_path: Path):
    path = tmp_path / "series.csv"
    path.write_text(GAPPY, encoding="utf-8")
    assert nab_gaps.gaps(path) == nab_gaps.Gaps(
        rows=5, missing=2, longest_steps=2, duplicates=1, off_grid=1
    )


def test_a_complete_series_has_no_gaps(tmp_path: Path):
    path = tmp_path / "series.csv"
    path.write_text("timestamp,value\n2014-01-01 00:00:00,1\n2014-01-01 00:05:00,2\n")
    assert nab_gaps.gaps(path) == nab_gaps.Gaps(2, 0, 0, 0, 0)


def _lock(tmp_path: Path, path: str) -> Path:
    data = GAPPY.encode()
    lock = FileLock(
        repository="numenta/NAB",
        files=(
            LockedFile(
                path=path,
                commit="0" * 40,
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            ),
        ),
    )
    lock_file = tmp_path / "nab.lock"
    lock_file.write_text(lock.model_dump_json(), encoding="utf-8")
    return lock_file


def test_main_prints_one_line_per_locked_series(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    relative = "data/realAWSCloudwatch/a.csv"
    (tmp_path / "data/realAWSCloudwatch").mkdir(parents=True)
    (tmp_path / relative).write_text(GAPPY, encoding="utf-8")
    lock_file = _lock(tmp_path, relative)
    assert nab_gaps.main(["--corpus", str(tmp_path), "--lock-file", str(lock_file)]) == 0
    assert capsys.readouterr().out == (
        "realAWSCloudwatch/a.csv: 5 rows, 2 missing, longest gap 2 steps (0.2 h), "
        "1 duplicate, 1 off-grid\n"
    )


def test_main_names_the_fetch_script_when_a_file_is_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    lock_file = _lock(tmp_path, "data/realAWSCloudwatch/a.csv")
    assert nab_gaps.main(["--corpus", str(tmp_path / "none"), "--lock-file", str(lock_file)]) == 1
    assert "run scripts/fetch_nab.py" in capsys.readouterr().err
