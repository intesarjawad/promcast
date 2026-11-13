import pytest

from promcast.core.config import ConfigError
from promcast.shell import errors, render
from promcast.shell.errors import PromcastError

NEXT_COMMANDS = ("promcast", "uvx", "docker", "kubectl", "export", "rm", "edit")

# One instance per failure the command line reports; a new subclass must be added here.
CASES: list[PromcastError] = [
    errors.NoPrometheusFoundError(["http://localhost:9090", "http://prometheus:9090"]),
    errors.TsdbUnavailableError("connection failed; check the TSDB address and network"),
    errors.TsdbAuthRejectedError(401),
    errors.TsdbRejectedError(500, "internal error"),
    errors.ConfigInvalidError("invalid configuration:\ntargets.0.quantile: bad", "promcast.yaml"),
    errors.ConfigMissingError("promcast/promcast.yaml"),
    errors.CompareExtraMissingError(),
    errors.UnknownModelError("nope"),
    errors.ModelFilesMissingError("local Toto model requires /models/toto/config.json"),
    errors.MaxSeriesOutOfRangeError(501),
    errors.QueryThresholdMissingError(),
    errors.QueryMetricUnknownError("up + 1"),
    errors.NoRecipeMatchedError(),
    errors.NoRecipeMatchedError("recipes"),
    errors.RecipesMissingError("recipes"),
    errors.RecipeInvalidError("name: field required"),
    errors.InitTargetExistsError("promcast/promcast.yaml"),
    errors.ReplayArgumentsError("replay requires --source or --auto"),
    errors.ReplayCounterRefusedError("counter"),
    errors.ReplayHistoryTooShortError(259200, 734400),
    errors.ReplayProbeUnavailableError(),
]


def subclasses(base: type[PromcastError]) -> set[type[PromcastError]]:
    found: set[type[PromcastError]] = set()
    for child in base.__subclasses__():
        found |= {child} | subclasses(child)
    return found


def test_every_subclass_is_in_the_catalogue() -> None:
    assert subclasses(PromcastError) == {type(error) for error in CASES}


@pytest.mark.parametrize("error", CASES, ids=lambda error: type(error).__name__)
def test_each_case_has_two_one_line_parts_and_a_runnable_next_step(error: PromcastError) -> None:
    assert error.what
    assert error.next_step.startswith(NEXT_COMMANDS)
    assert "\n" not in error.what + error.next_step
    assert str(error) == error.what


@pytest.mark.parametrize("error", CASES, ids=lambda error: type(error).__name__)
def test_report_is_two_lines(error: PromcastError) -> None:
    lines = render.error_report(error).splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("error: ")
    assert lines[1].startswith("next:  ")


def test_report_text_for_a_case() -> None:
    error = errors.NoPrometheusFoundError(["http://localhost:9090", "http://prometheus:9090"])
    assert render.error_report(error) == (
        "error: no Prometheus found; tried http://localhost:9090, http://prometheus:9090\n"
        "next:  promcast scan http://localhost:9090\n"
    )


def test_multi_line_detail_is_joined() -> None:
    error = errors.ConfigInvalidError("invalid configuration:\ntargets.0.quantile: bad", "a.yaml")
    assert render.error_report(error) == (
        "error: invalid configuration: targets.0.quantile: bad\n"
        "next:  edit a.yaml, then run promcast validate a.yaml\n"
    )


def test_bug_report_points_to_the_issue_form() -> None:
    assert render.bug_report(KeyError("x\ny")) == (
        "error: unexpected KeyError: 'x\\ny'\n"
        "next:  report it at https://github.com/intesarjawad/promcast/issues/new?template=bug.yml\n"
    )


def test_compare_extra_error_is_still_a_configuration_error() -> None:
    assert isinstance(errors.CompareExtraMissingError(), ConfigError)


@pytest.mark.parametrize("kind", ["discovery", "config", "status"])
def test_diagnostic_urls_remove_credentials(kind: str):
    url = "https://alice:password@metrics.example/tenant?access_token=secret&api_key=key&view=cpu"
    error = (
        errors.NoPrometheusFoundError([url])
        if kind == "discovery"
        else errors.ConfigInvalidError(url, url)
        if kind == "config"
        else errors.TsdbRejectedError(500, url)
    )
    output = render.error_report(error)
    for secret in ("alice", "password", "secret", "=key"):
        assert secret not in output
    assert "metrics.example/tenant" in output
    assert "view=cpu" in output


def test_unexpected_error_sanitizes_urls():
    assert "password" not in render.bug_report(ValueError("https://u:password@host/?token=key"))
    assert "=key" not in render.bug_report(ValueError("https://u:password@host/?token=key"))


def test_authentication_recovery_names_the_discovery_configuration_option():
    output = render.error_report(errors.TsdbAuthRejectedError(401))
    assert "promcast scan --config promcast.yaml" in output
    assert "tsdb.bearer_token_file" in output
