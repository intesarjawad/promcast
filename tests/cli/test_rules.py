import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from promcast.core.config import parse_config
from promcast.shell.cli import main
from promcast.shell.rules import generate_rules


@dataclass(frozen=True)
class RuleCase:
    config_path: str
    rules_path: str
    warning: int
    pending: str
    stale: int
    stale_pending: str


@pytest.mark.parametrize(
    "case",
    [
        RuleCase("promcast.example.yaml", "alerts/promcast.rules.yml", 120, "10m", 360, "5m"),
        RuleCase("demo/promcast.yaml", "demo/promcast.rules.yml", 60, "2m", 180, "1m"),
    ],
)
def test_rules_match_config_and_checked_in_files(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: RuleCase,
) -> None:
    monkeypatch.setattr(sys, "argv", ["promcast", "rules", "--config", case.config_path])
    assert main() == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == Path(case.rules_path).read_text()
    assert f"/ 60 < {case.warning}\n" in output.out
    assert f"for: {case.pending}\n" in output.out
    assert f"> {case.stale}\n" in output.out
    assert f"for: {case.stale_pending}\n" in output.out
    assert '{{ $value | printf "%.0f" }} min' in output.out


def test_per_target_rules_use_method_and_cadence() -> None:
    text = Path("demo/promcast.yaml").read_text()
    config = parse_config(text, environment={})
    second = config.targets[0].model_copy(update={"name": "other", "method": "linear"})
    config.targets.append(second)
    output = generate_rules(config)
    assert output.count("alert: PromcastBreachSoon") == 2
    assert output.count("alert: PromcastForecastStale") == 2
    assert 'target="other",method="linear"' in output
    assert output == generate_rules(config)


def test_rules_preserve_seconds_and_fractional_minutes() -> None:
    config = parse_config(
        Path("demo/promcast.yaml")
        .read_text()
        .replace("cycle: 1m", "cycle: 61s")
        .replace("warn: 1h", "warn: 31s"),
        environment={},
    )
    output = generate_rules(config)
    assert "for: 122s" in output
    assert "for: 61s" in output
    assert "> 183" in output
    assert "/ 60 < 0.5166666666666667" in output


def test_rules_do_not_import_model_adapters() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from promcast.shell.cli import main; "
            "sys.argv = ['promcast', 'rules', '--config', 'promcast.example.yaml']; "
            "assert main() == 0; assert 'torch' not in sys.modules; "
            "assert 'promcast.core.forecaster' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
