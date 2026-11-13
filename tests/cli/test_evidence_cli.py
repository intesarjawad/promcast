import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from promcast.core.evidence import CorpusEvidence, EvidenceReport, MethodSummary

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import evidence
else:
    spec = importlib.util.spec_from_file_location("evidence", ROOT / "scripts/evidence.py")
    assert spec is not None
    assert spec.loader is not None
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)


def _git(root: Path, *arguments: str) -> None:
    executable = shutil.which("git")
    assert executable is not None
    subprocess.run(  # noqa: S603
        [executable, "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _tagged_repo(path: Path) -> Path:
    (path / "docs").mkdir(parents=True)
    (path / "docs/protocol.md").write_text("protocol\n", encoding="utf-8")
    executable = shutil.which("git")
    assert executable is not None
    subprocess.run(  # noqa: S603
        [executable, "init", str(path)], check=True, capture_output=True, text=True
    )
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "commit.gpgSign", "false")
    _git(path, "config", "tag.gpgSign", "false")
    _git(path, "add", "docs/protocol.md")
    _git(path, "commit", "-m", "protocol")
    _git(path, "tag", "protocol-v1")
    return path


def test_estimate_prints_every_corpus_without_loading_data(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(sys, "argv", ["evidence.py", "--estimate", "--corpus", "e3"])
    assert evidence.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":", 1)[0] for line in lines] == ["e1", "e2", "e3", "e4"]
    assert all("forecasts" in line and "hours" in line for line in lines)


def test_estimate_counts_boom_gauges_from_the_taxonomy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    boom = tmp_path / "BOOM"
    boom.mkdir()
    (boom / "dataset_taxonomy.json").write_text(
        json.dumps(
            {
                "cpu": {"type": ["gauge"]},
                "disk": {"type": ["gauge"]},
                "requests": {"type": ["counter"]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["evidence.py", "--estimate", "--data-dir", str(tmp_path)])
    assert evidence.main() == 0
    e2 = capsys.readouterr().out.splitlines()[1]
    assert e2.startswith("e2: 2 series, ")
    assert "default cap" not in e2


def test_estimate_labels_boom_as_default_cap_when_data_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(
        sys, "argv", ["evidence.py", "--estimate", "--data-dir", str(tmp_path / "missing")]
    )
    assert evidence.main() == 0
    e2 = capsys.readouterr().out.splitlines()[1]
    assert e2.startswith("e2: 50 series (default cap, not the corpus size), ")


def test_edited_protocol_makes_a_corpus_run_exit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root = _tagged_repo(tmp_path / "repo")
    (root / "docs/protocol.md").write_text("protocol!\n", encoding="utf-8")
    monkeypatch.setattr(evidence, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["evidence.py", "--corpus", "e3"])
    assert evidence.main() == 1
    assert "changed since protocol-v1" in capsys.readouterr().err


def _report(recall: float) -> EvidenceReport:
    summary = MethodSummary("linear", 4, 8, recall, 0.5, 1.0, 600.0, None, None)
    corpus = CorpusEvidence("e3", 4, 1.0, (), (), {"linear": summary}, None)
    return EvidenceReport("ties", 1.0, (corpus,))


def test_first_report_is_written_without_a_drift_section(tmp_path: Path):
    evidence.write_report(_report(0.5), tmp_path, "Test CPU", accept_drift=False)
    assert "## Headline drift" not in (tmp_path / "docs/evidence.md").read_text()
    assert (tmp_path / "site/data/evidence.json").is_file()


def test_headline_drift_refuses_to_overwrite_the_published_report(tmp_path: Path):
    evidence.write_report(_report(0.5), tmp_path, "Test CPU", accept_drift=False)
    before = (tmp_path / "docs/evidence.md").read_text()
    site_before = (tmp_path / "site/data/evidence.json").read_text()
    with pytest.raises(evidence.HeadlineDriftError, match=r"e3 linear recall: 0\.5000 -> 0\.3000"):
        evidence.write_report(_report(0.3), tmp_path, "Test CPU", accept_drift=False)
    assert (tmp_path / "docs/evidence.md").read_text() == before
    assert (tmp_path / "site/data/evidence.json").read_text() == site_before


def test_accepted_drift_is_published_and_listed(tmp_path: Path):
    evidence.write_report(_report(0.5), tmp_path, "Test CPU", accept_drift=False)
    evidence.write_report(_report(0.3), tmp_path, "Test CPU", accept_drift=True)
    text = (tmp_path / "docs/evidence.md").read_text()
    assert text.splitlines()[0] == "ties"
    assert "## Headline drift" in text
    assert "- e3 linear recall: 0.5000 -> 0.3000 (-40.0%)" in text
    assert '"recall": 0.3' in (tmp_path / "site/data/evidence.json").read_text()


def test_unchanged_rerun_is_written(tmp_path: Path):
    evidence.write_report(_report(0.5), tmp_path, "Test CPU", accept_drift=False)
    evidence.write_report(_report(0.52), tmp_path, "Test CPU", accept_drift=False)
    assert '"recall": 0.52' in (tmp_path / "site/data/evidence.json").read_text()
