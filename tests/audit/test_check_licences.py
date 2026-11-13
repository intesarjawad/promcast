import importlib.util
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import check_licences
else:
    spec = importlib.util.spec_from_file_location(
        "check_licences", ROOT / "scripts/check_licences.py"
    )
    assert spec is not None
    assert spec.loader is not None
    check_licences = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_licences)


@pytest.mark.parametrize(
    "licence",
    [
        "MIT License",
        "BSD-3-Clause",
        "0BSD",
        "Apache Software License",
        "Apache-2.0 WITH LLVM-exception",
        "Python Software Foundation License",
        "PSF-2.0",
        "ISC License (ISCL)",
        "Mozilla Public License 2.0 (MPL 2.0)",
        "MPL-2.0",
        "GNU Lesser General Public License v3 (LGPLv3)",
        "LGPL-2.1-or-later",
        "The Unlicense (Unlicense)",
        "Apache Software License; BSD License",
        "MIT OR GPL-3.0-only",
        "(MIT OR Apache-2.0) AND BSD-3-Clause",
    ],
)
def test_allowed_families_pass(licence: str):
    assert check_licences.allowed(licence)


@pytest.mark.parametrize(
    "licence",
    [
        "GNU General Public License v3 (GPLv3)",
        "GPL-2.0-or-later",
        "AGPL-3.0-only",
        "Mozilla Public License 1.1 (MPL 1.1)",
        "UNKNOWN",
        "",
        "MIT AND Zlib",
        "Apache Software License; GNU General Public License (GPL)",
        "(MIT OR Apache-2.0) AND CC0-1.0",
    ],
)
def test_other_licences_fail(licence: str):
    assert not check_licences.allowed(licence)


def _run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], text: str
) -> tuple[int, str, str]:
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    code = check_licences.main()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_main_prints_each_package_needing_an_adr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    rows = [
        {"Name": "httpx", "Version": "0.28.1", "License": "BSD License"},
        {"Name": "copyleft", "Version": "1.0", "License": "GPL-3.0-only"},
        {"Name": "mystery", "Version": "2.0", "License": "UNKNOWN"},
    ]
    code, out, err = _run(monkeypatch, capsys, json.dumps(rows))
    assert code == 1
    assert out == "copyleft 1.0: GPL-3.0-only\nmystery 2.0: UNKNOWN\n"
    assert "2 of 3 packages" in err
    assert "ADR" in err


def test_main_passes_when_every_licence_is_allowed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    rows = [{"Name": "rich", "Version": "15.0.0", "License": "MIT License"}]
    code, out, _ = _run(monkeypatch, capsys, json.dumps(rows))
    assert code == 0
    assert out == "Every package uses an allowed licence (1 checked)\n"


@pytest.mark.parametrize("text", ["", "{}", '[{"Name": "x"}]'])
def test_main_rejects_input_that_is_not_pip_licenses_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], text: str
):
    code, _, err = _run(monkeypatch, capsys, text)
    assert code == 1
    assert "pip-licenses --format=json" in err


def test_an_adr_accepts_a_package_only_at_the_reviewed_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    rows = [
        {"Name": "augurs", "Version": "0.10.3", "License": "UNKNOWN"},
        {"Name": "augurs", "Version": "0.10.4", "License": "UNKNOWN"},
    ]
    code, out, _ = _run(monkeypatch, capsys, json.dumps(rows))
    assert code == 1
    assert out == "augurs 0.10.4: UNKNOWN\n"


def test_every_accepted_package_names_an_adr_that_records_it():
    for (name, version), adr in check_licences.ACCEPTED.items():
        text = (ROOT / adr).read_text(encoding="utf-8")
        assert f"{name} {version}" in text
