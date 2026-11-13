import subprocess
import sys
from pathlib import Path

import pytest

from promcast.core.config import ConfigError
from promcast.shell.cli import load_forecaster


@pytest.mark.parametrize("missing", ["chronos", "unrelated_dependency"])
def test_optional_comparison_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    def unavailable(name: str) -> object:
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr("promcast.core.forecaster_chronos.import_module", unavailable)
    expected = ConfigError if missing == "chronos" else ModuleNotFoundError
    with pytest.raises(expected) as error:
        load_forecaster(tmp_path, "chronos-2-small")
    if missing == "chronos":
        assert "install the compare extra" in str(error.value)


def test_validation_does_not_import_model_adapters_or_chronos() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import promcast.core.config; "
            "assert 'promcast.shell.config' not in sys.modules; "
            "from promcast.shell.cli import main; "
            "sys.argv = ['promcast', 'validate', 'promcast.example.yaml']; "
            "assert main() == 0; "
            "assert 'chronos' not in sys.modules; "
            "assert 'promcast.core.forecaster_chronos' not in sys.modules; "
            "assert 'promcast.core.forecaster' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\n"
