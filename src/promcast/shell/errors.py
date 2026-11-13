from promcast.core.config import ConfigError
from promcast.core.diagnostics import safe_urls


class PromcastError(Exception):
    """A failure the user can fix, reported as an `error:` line and a `next:` line."""

    def __init__(self, what: str, next_step: str) -> None:
        self.what = safe_urls(" ".join(what.split()))
        self.next_step = safe_urls(" ".join(next_step.split()))
        super().__init__(self.what)


class NoPrometheusFoundError(PromcastError):
    """No candidate endpoint answered a Prometheus query."""

    def __init__(self, tried: list[str]) -> None:
        super().__init__(
            "no Prometheus found; tried " + ", ".join(tried),
            "promcast scan http://localhost:9090",
        )


class TsdbUnavailableError(PromcastError):
    """The TSDB could not be reached, or a file needed to authenticate could not be read."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            detail, "edit tsdb.url in your configuration, or give promcast scan another URL"
        )


class TsdbAuthRejectedError(PromcastError):
    """The TSDB answered 401 or 403."""

    def __init__(self, status: int) -> None:
        super().__init__(
            f"TSDB rejected the credentials (HTTP {status})",
            "edit tsdb.basic_auth, tsdb.bearer_token_file or tsdb.headers in promcast.yaml, "
            "then run promcast scan --config promcast.yaml",
        )


class TsdbRejectedError(PromcastError):
    """The TSDB answered with an error status or a body that is not a Prometheus response."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(
            f"TSDB returned HTTP {status}: {detail}", "promcast scan http://localhost:9090"
        )


class ConfigInvalidError(PromcastError):
    """A configuration file could not be read or validated."""

    def __init__(self, detail: str, path: str) -> None:
        super().__init__(detail, f"edit {path}, then run promcast validate {path}")


class ConfigMissingError(PromcastError):
    """The configuration file named on the command line does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(f"config does not exist: {path}", "promcast init")


class CompareExtraMissingError(PromcastError, ConfigError):
    """The Chronos comparison was requested without its optional dependency.

    It stays a ConfigError so callers that load a model by name can still catch the
    configuration failure it has always been.
    """

    def __init__(self) -> None:
        super().__init__(
            "install the compare extra to use chronos-2-small",
            "uvx --from 'promcast[compare]' promcast replay --models chronos-2-small",
        )


class ModelFilesMissingError(PromcastError):
    """Pinned model files could not be downloaded or verified."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"model files not found ({detail})",
            "promcast scan after restoring network access; for offline use, copy the verified "
            "model folder to PROMCAST_CACHE_DIR/<model>; "
            "remove a file with a bad checksum and retry",
        )


class UnknownModelError(PromcastError):
    """A model name that no forecaster implements."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown model: {name}", "promcast replay --models toto-2.0-22m")


class MaxSeriesOutOfRangeError(PromcastError):
    """--max-series is outside the supported range."""

    def __init__(self, value: int) -> None:
        super().__init__(f"--max-series {value} is outside 1..500", "promcast scan --max-series 50")


class QueryThresholdMissingError(PromcastError):
    """--query was given without a direction and threshold."""

    def __init__(self) -> None:
        super().__init__(
            "--query requires --below or --above", "promcast scan --query '<PromQL>' --below 0"
        )


class QueryMetricUnknownError(PromcastError):
    """No metric of the Prometheus can be read out of a --query expression."""

    def __init__(self, query: str) -> None:
        super().__init__(
            f"cannot identify a metric in {query}",
            "edit a recipe YAML for this query, then run promcast scan --recipes DIR",
        )


class NoRecipeMatchedError(PromcastError):
    """The Prometheus has none of the metrics the recipes need."""

    def __init__(self, directory: str | None = None) -> None:
        super().__init__(
            "no recipe metrics matched"
            if directory is None
            else f"no recipe metric matched --recipes {directory}",
            "promcast scan --query '<PromQL>' --below 0",
        )


class RecipesMissingError(PromcastError):
    """The --recipes folder holds no recipe files."""

    def __init__(self, directory: str) -> None:
        super().__init__(
            f"no recipes in {directory}", "promcast scan, without --recipes, uses the bundled ones"
        )


class RecipeInvalidError(PromcastError):
    """A recipe file is not valid YAML or does not follow the recipe format."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"invalid recipe: {detail}", "edit the recipe YAML, then rerun the command"
        )


class InitTargetExistsError(PromcastError):
    """Adoption would overwrite an existing file."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"init target exists: {path}",
            f"promcast init --out DIR, or add --force to overwrite {path}",
        )


class ReplayArgumentsError(PromcastError):
    """The replay options do not name a complete source."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail, "promcast replay --auto")


class ReplayCounterRefusedError(PromcastError):
    """A synthetic scenario is a cumulative counter, which replay never forecasts."""

    def __init__(self, scenario: str) -> None:
        super().__init__(
            f"synthetic scenario {scenario} is a counter, not a gauge; "
            "replay forecasts gauges only",
            "promcast replay --source synthetic --scenario accelerating_leak",
        )


class ReplayHistoryTooShortError(PromcastError):
    """No series has enough history for an automatic replay."""

    def __init__(self, available_s: int, required_s: int) -> None:
        available_h, available_m = divmod(available_s // 60, 60)
        required_h, required_m = divmod(required_s // 60, 60)
        wait_h, wait_m = divmod((required_s - available_s + 59) // 60, 60)
        super().__init__(
            f"no series has enough history for replay --auto: {available_h}h {available_m:02d}m "
            f"available, {required_h}h {required_m:02d}m required for the nearest eligible series",
            f"edit TSDB retention to keep the required history; collect at least "
            f"{wait_h}h {wait_m:02d}m more, then rerun the same promcast replay --auto command",
        )


class ReplayProbeUnavailableError(PromcastError):
    """No series could be forecast to time the replay."""

    def __init__(self) -> None:
        super().__init__("no eligible inference probe for replay ETA", "promcast scan")


NoPrometheusFound = NoPrometheusFoundError
InitTargetExists = InitTargetExistsError
