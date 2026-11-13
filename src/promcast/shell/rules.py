import yaml

from promcast.core.config import Config, TargetConfig


def _duration(seconds: int) -> str:
    for suffix, unit in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % unit == 0:
            return f"{seconds // unit}{suffix}"
    return f"{seconds}s"


def _target_rules(config: Config, target: TargetConfig) -> list[dict[str, object]]:
    cycle_s = int(config.cycle.total_seconds())
    effective_s = max(cycle_s, int(target.step.total_seconds()))
    warning_minutes = str(target.warn.total_seconds() / 60).removesuffix(".0")
    # Prometheus identifies duplicate rules by alert name and static labels.
    identity = {"target": target.name} if len(config.targets) > 1 else {}
    return [
        {
            "alert": "PromcastBreachSoon",
            "expr": (
                f'(promcast_breach_timestamp_seconds{{target="{target.name}",'
                f'method="{target.method}"}} - time()) / 60 < {warning_minutes}'
            ),
            "for": _duration(2 * effective_s),
            "labels": {"severity": "warning", **identity},
            "annotations": {
                "summary": (
                    '{{ $labels.target }} forecast to breach in {{ $value | printf "%.0f" }} min'
                ),
                "lead_minutes": '{{ $value | printf "%.0f" }}',
                "forecast_origin": (
                    '{{ with printf "promcast_forecast_origin_timestamp_seconds{target=%q}" '
                    "$labels.target | query }}{{ . | first | value | humanizeTimestamp }}{{ end }}"
                ),
                "model": config.model.name,
            },
        },
        {
            "alert": "PromcastForecastStale",
            "expr": (
                f'time() - promcast_last_success_timestamp_seconds{{target="{target.name}"}} '
                f"> {3 * cycle_s}"
            ),
            "for": _duration(effective_s),
            "labels": {"severity": "info", **identity},
        },
    ]


def generate_rules(config: Config) -> str:
    """Render deterministic Prometheus rules using each target's effective cadence."""
    rules = [rule for target in config.targets for rule in _target_rules(config, target)]
    return yaml.safe_dump(
        {"groups": [{"name": "promcast", "rules": rules}]}, sort_keys=False, width=1000
    )
