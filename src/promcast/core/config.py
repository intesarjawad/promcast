import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    PlainSerializer,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
)

if TYPE_CHECKING:
    from promcast.shell.config import load_config

__all__ = [
    "BasicAuthConfig",
    "Config",
    "ConfigError",
    "Duration",
    "ModelConfig",
    "RedactedSecret",
    "SecretValue",
    "ServerConfig",
    "TLSConfig",
    "TSDBConfig",
    "TargetConfig",
    "TargetsFragment",
    "cache_dir_override",
    "load_config",
    "parse_config",
    "parse_fragment",
]

_CACHE_DIR_VARIABLE = "PROMCAST_CACHE_DIR"
_DURATION = re.compile(r"([0-9]+)(s|m|h|d)", re.ASCII)
_ENVIRONMENT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9)
# Limits of the forecasting model's context and horizon, in steps.
_MIN_CONTEXT_STEPS = 64
_MAX_CONTEXT_STEPS = 2048
_MAX_HORIZON_STEPS = 512
_MIN_LINEAR_WINDOW_STEPS = 2
_MAX_SERIES = 500


class ConfigError(ValueError):
    """Configuration could not be read or validated; the message is safe to display."""


@dataclass(frozen=True)
class _Environment:
    values: Mapping[str, str]


def _expand_environment(value: object, info: ValidationInfo) -> object:
    if not isinstance(value, str) or not isinstance(info.context, _Environment):
        return value
    environment = info.context.values

    def replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        if variable not in environment:
            raise ValueError(f"environment variable {variable} is not set")
        return environment[variable]

    return _ENVIRONMENT_REFERENCE.sub(replace, value)


def _parse_duration(value: object) -> timedelta:
    if isinstance(value, timedelta):
        return value
    if not isinstance(value, str) or (match := _DURATION.fullmatch(value)) is None:
        raise ValueError("duration must be an integer followed by s, m, h or d (for example 5m)")
    try:
        seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
        return timedelta(seconds=int(match.group(1)) * seconds_per_unit)
    except OverflowError as error:
        raise ValueError("duration exceeds the supported timedelta range") from error


Duration = Annotated[timedelta, BeforeValidator(_parse_duration)]


class RedactedSecret(SecretStr):
    """SecretStr with the same display marker in diagnostics and JSON."""

    def __str__(self) -> str:
        return "<redacted>"

    def __repr__(self) -> str:
        return "RedactedSecret('<redacted>')"


SecretValue = Annotated[
    RedactedSecret,
    PlainSerializer(str, return_type=str, when_used="json"),
]


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, validate_default=True)

    @field_validator("*", mode="before")
    @classmethod
    def _expand_field(cls, value: object, info: ValidationInfo) -> object:
        return _expand_environment(value, info)


class BasicAuthConfig(_ConfigModel):
    """Basic authentication references; file contents are read by the HTTP shell."""

    username: SecretValue
    password_file: Path


class TLSConfig(_ConfigModel):
    """Optional custom trust bundle for TSDB requests."""

    ca_file: Path | None = None


def _reject_url_userinfo(value: object) -> object:
    if isinstance(value, str | HttpUrl):
        try:
            authority = urlsplit(str(value)).netloc
        except ValueError:
            raise ValueError("URL must be a valid HTTP or HTTPS URL") from None
        if "@" in authority:
            raise ValueError("URL userinfo is forbidden; use basic_auth for credentials")
    return value


class TSDBConfig(_ConfigModel):
    """Prometheus-compatible endpoint and transport authentication."""

    # Check both raw empty userinfo and credentials exposed by URL normalization.
    url: Annotated[
        HttpUrl, BeforeValidator(_reject_url_userinfo), AfterValidator(_reject_url_userinfo)
    ]
    timeout_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 30
    headers: dict[str, Annotated[SecretValue, BeforeValidator(_expand_environment)]] = Field(
        default_factory=dict
    )
    bearer_token_file: Path | None = None
    basic_auth: BasicAuthConfig | None = None
    tls: TLSConfig = Field(default_factory=TLSConfig)

    @field_validator("basic_auth")
    @classmethod
    def _exclusive_auth(
        cls, value: BasicAuthConfig | None, info: ValidationInfo
    ) -> BasicAuthConfig | None:
        if value is not None and info.data.get("bearer_token_file") is not None:
            raise ValueError("choose only one of basic_auth and bearer_token_file")
        return value


class ModelConfig(_ConfigModel):
    """Local forecasting model selection and batch size."""

    name: Literal["toto-2.0-22m", "toto-2.0-4m"] = "toto-2.0-22m"
    batch_size: Annotated[int, Field(gt=0)] = 16
    cache_dir: Path = Path("~/.cache/promcast/models")


class ServerConfig(_ConfigModel):
    """Metrics server binding."""

    listen: str = "0.0.0.0:8787"


class TargetConfig(_ConfigModel):
    """One forecasting target, with durations normalized to timedeltas."""

    name: Annotated[str, Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_-]*$")]
    query: str
    step: Annotated[Duration, Field(gt=timedelta(0))]
    context: Duration
    horizon: Duration
    season: Duration
    direction: Literal["above", "below"]
    threshold: Annotated[float, Field(allow_inf_nan=False)]
    quantile: float
    warn: Duration
    method: Literal["toto", "linear"]
    linear_window: Duration
    known_gauge: bool = False
    max_series: Annotated[int, Field(gt=0, le=_MAX_SERIES)]

    @field_validator("quantile")
    @classmethod
    def _check_quantile(cls, value: float) -> float:
        if value not in _QUANTILES:
            raise ValueError("quantile must be one of 0.5, 0.6, 0.7, 0.8, 0.9")
        return value

    # Field order keeps dependent validation at the field's own error path.
    @field_validator("context", "horizon")
    @classmethod
    def _check_step_ratio(cls, value: timedelta, info: ValidationInfo) -> timedelta:
        step: timedelta | None = info.data.get("step")
        if step is not None:
            if (
                info.field_name == "context"
                and not _MIN_CONTEXT_STEPS <= value / step <= _MAX_CONTEXT_STEPS
            ):
                raise ValueError(
                    f"context/step must be between {_MIN_CONTEXT_STEPS} and {_MAX_CONTEXT_STEPS}"
                )
            if info.field_name == "horizon":
                if value < step:
                    raise ValueError("horizon must be at least step")
                if value / step > _MAX_HORIZON_STEPS:
                    raise ValueError(f"horizon/step must be at most {_MAX_HORIZON_STEPS}")
        return value

    @field_validator("warn")
    @classmethod
    def _check_warning_horizon(cls, value: timedelta, info: ValidationInfo) -> timedelta:
        horizon: timedelta | None = info.data.get("horizon")
        if horizon is not None and value > horizon:
            raise ValueError("warn must be at most horizon")
        return value

    @field_validator("linear_window")
    @classmethod
    def _check_linear_window(cls, value: timedelta, info: ValidationInfo) -> timedelta:
        context: timedelta | None = info.data.get("context")
        step: timedelta | None = info.data.get("step")
        if context is not None and value > context:
            raise ValueError("linear_window must be at most context")
        if step is not None and value / step < _MIN_LINEAR_WINDOW_STEPS:
            raise ValueError(f"linear_window/step must be at least {_MIN_LINEAR_WINDOW_STEPS}")
        return value


class Config(_ConfigModel):
    """Complete runtime configuration."""

    tsdb: TSDBConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    cycle: Annotated[Duration, Field(gt=timedelta(0))] = timedelta(minutes=2)
    targets: list[TargetConfig]


class TargetsFragment(_ConfigModel):
    """Replay recommendations with an optional required forecasting model."""

    targets: list[TargetConfig]
    model: ModelConfig | None = None


def _parse_document[Model: _ConfigModel](
    text: str, model: type[Model], environment: Mapping[str, str]
) -> Model:
    try:
        document: object = yaml.safe_load(text)
    except yaml.YAMLError as error:
        # YAML exceptions can contain entire source lines, including credentials.
        location = ""
        if isinstance(error, yaml.MarkedYAMLError) and error.problem_mark is not None:
            location = (
                f" at line {error.problem_mark.line + 1}, column {error.problem_mark.column + 1}"
            )
        raise ConfigError(
            f"invalid YAML{location}; check indentation, quotes and mapping syntax"
        ) from None
    try:
        return model.model_validate(document, context=_Environment(environment))
    except ValidationError as error:
        problems: list[str] = []
        for problem in error.errors(include_input=False, include_context=False, include_url=False):
            path = ".".join(str(part) for part in problem["loc"]) or "config"
            problems.append(f"{path}: {problem['msg']}")
        raise ConfigError("invalid configuration:\n" + "\n".join(problems)) from None


def cache_dir_override(environment: Mapping[str, str]) -> Path | None:
    """Return the model cache directory named by PROMCAST_CACHE_DIR, if any."""
    value = environment.get(_CACHE_DIR_VARIABLE)
    return Path(value) if value else None


def parse_config(text: str, *, environment: Mapping[str, str]) -> Config:
    """Validate YAML using only the supplied environment, without reading secret files."""
    config = _parse_document(text, Config, environment)
    if (override := cache_dir_override(environment)) is not None:
        config.model.cache_dir = override
    return config


def parse_fragment(text: str, *, environment: Mapping[str, str]) -> TargetsFragment:
    """Validate recommended targets and their required model with the runtime rules."""
    return _parse_document(text, TargetsFragment, environment)


def __getattr__(name: str) -> object:
    """Resolve the legacy file loader in the I/O shell without coupling parsing to it."""
    if name == "load_config":
        return import_module("promcast.shell.config").load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
