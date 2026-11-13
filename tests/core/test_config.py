import json
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr

from promcast.core.config import ConfigError, load_config, parse_config, parse_fragment
from promcast.shell.config import load_config as shell_load_config

EXAMPLE_PATH = Path(__file__).parents[1] / "fixtures/config.yaml"
EXAMPLE = EXAMPLE_PATH.read_text(encoding="utf-8")


def test_example_loads() -> None:
    assert load_config is shell_load_config
    config = load_config(EXAMPLE_PATH, environment={})
    target = config.targets[0]
    assert str(config.tsdb.url) == "http://prometheus:9090/"
    assert config.cycle == timedelta(minutes=2)
    assert target.context / target.step == 1152
    assert target.horizon / target.step == 72
    assert target.threshold == 5e9


@pytest.mark.parametrize(
    ("original", "replacement", "field", "message"),
    [
        ("step: 5m", "step: 0m", "targets.0.step", "greater than"),
        ("step: 5m", "step: 1.5m", "targets.0.step", "duration"),
        ("context: 4d", "context: 1h", "targets.0.context", "64"),
        ("context: 4d", "context: 8d", "targets.0.context", "2048"),
        ("horizon: 6h", "horizon: 2d", "targets.0.horizon", "512"),
        ("warn: 2h", "warn: 7h", "targets.0.warn", "horizon"),
        ("linear_window: 1h", "linear_window: 5d", "targets.0.linear_window", "context"),
        ("linear_window: 1h", "linear_window: 5m", "targets.0.linear_window", "2"),
        ("quantile: 0.9", "quantile: 0.95", "targets.0.quantile", "0.9"),
        ("max_series: 50", "max_series: 501", "targets.0.max_series", "500"),
        ("name: root_disk_free", "name: 9disk", "targets.0.name", "pattern"),
        ("method: toto", "method: other", "targets.0.method", "linear"),
        ("direction: below", "direction: other", "targets.0.direction", "above"),
        ("timeout_s: 30", "timeout_s: 0", "tsdb.timeout_s", "greater than"),
        ("batch_size: 16", "batch_size: 0", "model.batch_size", "greater than"),
        ("threshold: 5e9", "threshold: .nan", "targets.0.threshold", "finite"),
        ("url: http://prometheus:9090", "url: file:///tmp/db", "tsdb.url", "http"),
        (
            "basic_auth: null",
            "basic_auth: {username: u, password_file: /run/secrets/p, password: hidden}",
            "tsdb.basic_auth.password",
            "Extra inputs",
        ),
        ("tls: {ca_file: null}", "tls: {verify: false}", "tsdb.tls.verify", "Extra inputs"),
        ("max_series: 50", "max_series: 50\n    typo: true", "targets.0.typo", "Extra inputs"),
        ("cycle: 2m", "cycle: 2m\ntypo: true", "typo", "Extra inputs"),
    ],
)
def test_invalid_fields_name_the_path(
    original: str, replacement: str, field: str, message: str
) -> None:
    with pytest.raises(ConfigError) as error:
        parse_config(EXAMPLE.replace(original, replacement), environment={})
    assert field in str(error.value)
    assert message in str(error.value)


def test_authentication_modes_are_mutually_exclusive() -> None:
    document = EXAMPLE.replace("bearer_token_file: null", "bearer_token_file: /run/secrets/t")
    document = document.replace(
        "basic_auth: null", "basic_auth: {username: u, password_file: /run/secrets/p}"
    )
    with pytest.raises(ConfigError, match=r"tsdb.basic_auth.*bearer_token_file"):
        parse_config(document, environment={})


@pytest.mark.parametrize(
    "userinfo", ["private-user:private-password", "private-user", ":private-password", "", ":"]
)
def test_url_userinfo_is_rejected_without_disclosing_credentials(userinfo: str) -> None:
    document = EXAMPLE.replace("http://prometheus:9090", "${URL}")
    with pytest.raises(ConfigError, match=r"tsdb.url.*basic_auth") as error:
        parse_config(document, environment={"URL": f"https://{userinfo}@metrics.example"})
    diagnostic = "".join(traceback.format_exception(error.value))
    assert "private-user" not in diagnostic
    assert "private-password" not in diagnostic


@pytest.mark.parametrize("horizon", ["0s", "1s", "299s"])
def test_horizon_below_step_is_rejected_at_field_path(horizon: str) -> None:
    document = EXAMPLE.replace("horizon: 6h", f"horizon: {horizon}").replace("warn: 2h", "warn: 0s")
    with pytest.raises(ConfigError, match=r"targets.0.horizon.*at least step"):
        parse_config(document, environment={})


def test_horizon_equal_to_step_is_accepted() -> None:
    document = EXAMPLE.replace("horizon: 6h", "horizon: 5m").replace("warn: 2h", "warn: 5m")
    config = parse_config(document, environment={})
    assert config.targets[0].horizon == config.targets[0].step


@pytest.mark.parametrize(
    "url",
    [
        "https://private-user:private-password@bad\uff0fhost",
        "https://private-user:private-password@[broken",
        r"https:\\private-user:private-password@metrics.example",
    ],
)
def test_malformed_or_normalized_url_cannot_disclose_credentials(url: str) -> None:
    with pytest.raises(ConfigError, match=r"tsdb.url") as error:
        parse_config(EXAMPLE.replace("http://prometheus:9090", "${URL}"), environment={"URL": url})
    diagnostic = "".join(traceback.format_exception(error.value))
    assert "private-user" not in diagnostic
    assert "private-password" not in diagnostic


@pytest.mark.parametrize("mode", ["bearer", "basic"])
def test_authentication_paths_do_not_read_secret_files(mode: str) -> None:
    document = EXAMPLE.replace("bearer_token_file: null", "bearer_token_file: /missing/token")
    if mode == "basic":
        document = EXAMPLE.replace(
            "basic_auth: null", "basic_auth: {username: u, password_file: /missing/password}"
        )
    config = parse_config(document, environment={})
    if mode == "bearer":
        assert config.tsdb.bearer_token_file == Path("/missing/token")
    else:
        assert config.tsdb.basic_auth is not None
        assert config.tsdb.basic_auth.password_file == Path("/missing/password")


def test_cache_directory_environment_overrides_the_document() -> None:
    config = parse_config(EXAMPLE, environment={"PROMCAST_CACHE_DIR": "/models"})
    assert config.model.cache_dir == Path("/models")


def test_cache_directory_environment_leaves_the_document_when_empty() -> None:
    config = parse_config(EXAMPLE, environment={"PROMCAST_CACHE_DIR": ""})
    assert config.model.cache_dir == Path("~/.cache/promcast/models")


def test_environment_expansion_is_literal_and_uses_the_supplied_environment() -> None:
    document = EXAMPLE.replace("headers: {}", 'headers: {Authorization: "Bearer ${TOKEN}"}')
    document = document.replace("http://prometheus:9090", "${TSDB_URL}")
    document = document.replace("cache_dir: ~/.cache/promcast/models", "cache_dir: ${CACHE}/models")
    replacement_value = 'abc: [x]\nquery: "injected" # ${UNCHANGED}'
    config = parse_config(
        document,
        environment={
            "TOKEN": replacement_value,
            "TSDB_URL": "https://metrics.example/prometheus",
            "CACHE": "/cache",
        },
    )
    assert str(config.tsdb.url) == "https://metrics.example/prometheus"
    assert config.tsdb.headers["Authorization"].get_secret_value() == f"Bearer {replacement_value}"
    assert config.model.cache_dir == Path("/cache/models")


def test_missing_environment_variable_names_its_field_and_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOKEN", "must-not-be-used")
    document = EXAMPLE.replace("headers: {}", 'headers: {Authorization: "${TOKEN}"}')
    with pytest.raises(ConfigError, match=r"tsdb.headers.Authorization.*TOKEN"):
        parse_config(document, environment={})


def test_environment_values_are_not_expanded_again() -> None:
    document = EXAMPLE.replace(
        "basic_auth: null", "basic_auth: {username: '${USER}', password_file: /secrets/p}"
    )
    config = parse_config(document, environment={"USER": "literal-${OTHER}"})
    assert config.tsdb.basic_auth is not None
    assert config.tsdb.basic_auth.username.get_secret_value() == "literal-${OTHER}"


def test_redaction_in_repr_and_model_dump_json() -> None:
    document = EXAMPLE.replace(
        "headers: {}", 'headers: {Authorization: "Bearer ${TOKEN}", X-Scope-OrgID: private-team}'
    ).replace("basic_auth: null", "basic_auth: {username: private-user, password_file: /secrets/p}")
    config = parse_config(document, environment={"TOKEN": "sensitive-token"})
    header = config.tsdb.headers["Authorization"]
    assert isinstance(header, SecretStr)
    assert header.get_secret_value() == "Bearer sensitive-token"
    assert str(header) == "<redacted>"
    for display in (repr(config), str(config), repr(config.model_dump()), config.model_dump_json()):
        assert "<redacted>" in display
        for secret in ("sensitive-token", "private-team", "private-user"):
            assert secret not in display
    dumped = json.loads(config.model_dump_json())
    assert dumped["tsdb"]["headers"]["Authorization"] == "<redacted>"
    assert dumped["tsdb"]["basic_auth"]["username"] == "<redacted>"


def test_validation_errors_do_not_echo_unvalidated_secrets() -> None:
    document = EXAMPLE.replace("headers: {}", "headers: {Authorization: {token: private-token}}")
    with pytest.raises(ConfigError) as error:
        parse_config(document, environment={})
    assert "tsdb.headers.Authorization" in str(error.value)
    assert "private-token" not in str(error.value)


def test_fragment_reuses_target_validation() -> None:
    document = "targets:" + EXAMPLE.split("targets:", 1)[1]
    fragment = parse_fragment(document, environment={})
    assert fragment.targets == parse_config(EXAMPLE, environment={}).targets
    with pytest.raises(ConfigError, match=r"targets.0.quantile"):
        parse_fragment(document.replace("quantile: 0.9", "quantile: 0.95"), environment={})
    with pytest.raises(ConfigError, match="tsdb"):
        parse_fragment(EXAMPLE, environment={})


@pytest.mark.parametrize("document", ["", "null", "[]", "a string"])
def test_non_mapping_document_is_rejected(document: str) -> None:
    with pytest.raises(ConfigError):
        parse_config(document, environment={})


def test_yaml_errors_do_not_echo_source_secrets() -> None:
    document = 'tsdb: ["private-token"'
    with pytest.raises(ConfigError, match="YAML") as error:
        parse_config(document, environment={})
    assert "private-token" not in str(error.value)
    assert "private-token" not in "".join(traceback.format_exception(error.value))


def test_missing_file_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"missing\.yaml"):
        load_config(tmp_path / "missing.yaml", environment={})


@pytest.mark.parametrize(
    "field", ["cycle", "step", "context", "horizon", "season", "warn", "linear_window"]
)
@pytest.mark.parametrize("value", ["-1m", "1.5h", "1w", "1h30m", "5M", "5", "' 5m'", "'5m '"])
def test_duration_grammar_is_shared_by_every_duration(field: str, value: str) -> None:
    lines = EXAMPLE.splitlines()
    document = "\n".join(
        f"{line.split(':', 1)[0]}: {value}" if line.lstrip().startswith(f"{field}:") else line
        for line in lines
    )
    path = "cycle" if field == "cycle" else f"targets.0.{field}"
    with pytest.raises(ConfigError) as error:
        parse_config(document, environment={})
    assert f"{path}:" in str(error.value)
    assert "duration" in str(error.value)


@pytest.mark.parametrize("quantile", [0.5, 0.6, 0.7, 0.8, 0.9])
def test_all_supported_quantiles_are_accepted(quantile: float) -> None:
    config = parse_config(EXAMPLE.replace("quantile: 0.9", f"quantile: {quantile}"), environment={})
    assert config.targets[0].quantile == quantile


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("context: 4d", "context: 320m"),
        ("context: 4d", "context: 10240m"),
        ("horizon: 6h", "horizon: 2560m"),
        ("warn: 2h", "warn: 6h"),
        ("linear_window: 1h", "linear_window: 4d"),
        ("linear_window: 1h", "linear_window: 10m"),
        ("max_series: 50", "max_series: 500"),
    ],
)
def test_validation_limits_are_inclusive(original: str, replacement: str) -> None:
    parse_config(EXAMPLE.replace(original, replacement), environment={})


def test_all_environment_references_resolve_in_each_scalar(tmp_path: Path) -> None:
    document = EXAMPLE.replace("cache_dir: ~/.cache/promcast/models", "cache_dir: ${ROOT}/${LEAF}")
    config = parse_config(document, environment={"ROOT": str(tmp_path), "LEAF": "weights"})
    assert config.model.cache_dir == tmp_path / "weights"


def test_empty_environment_value_remains_a_redacted_secret() -> None:
    document = EXAMPLE.replace("headers: {}", 'headers: {Authorization: "${EMPTY}"}')
    config = parse_config(document, environment={"EMPTY": ""})
    assert config.tsdb.headers["Authorization"].get_secret_value() == ""
    assert str(config.tsdb.headers["Authorization"]) == "<redacted>"
    assert json.loads(config.model_dump_json())["tsdb"]["headers"]["Authorization"] == "<redacted>"


def test_independent_field_errors_are_reported_together() -> None:
    document = EXAMPLE.replace("context: 4d", "context: 1h").replace(
        "quantile: 0.9", "quantile: 0.95"
    )
    with pytest.raises(ConfigError) as error:
        parse_config(document, environment={})
    assert "targets.0.context" in str(error.value)
    assert "targets.0.quantile" in str(error.value)


def test_secret_files_and_ca_paths_support_environment_expansion() -> None:
    document = EXAMPLE.replace("bearer_token_file: null", "bearer_token_file: ${SECRETS}/token")
    document = document.replace("tls: {ca_file: null}", "tls: {ca_file: '${SECRETS}/ca.pem'}")
    config = parse_config(document, environment={"SECRETS": "/run/secrets"})
    assert config.tsdb.bearer_token_file == Path("/run/secrets/token")
    assert config.tsdb.tls.ca_file == Path("/run/secrets/ca.pem")
