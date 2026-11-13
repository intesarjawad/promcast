from collections.abc import Mapping
from pathlib import Path

from promcast.core.config import Config, ConfigError, parse_config


def read_config(path: Path) -> str:
    """Read UTF-8 YAML with safe, actionable file errors."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ConfigError(
            f"cannot read configuration {path}; check the path, permissions and UTF-8"
        ) from None


def load_config(path: Path, *, environment: Mapping[str, str]) -> Config:
    """Load a configuration using an explicitly supplied environment."""
    return parse_config(read_config(path), environment=environment)
