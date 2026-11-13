import json
from dataclasses import asdict
from statistics import median

import numpy as np
import yaml

from promcast.core.bootstrap import bootstrap_ci
from promcast.core.config import TargetConfig
from promcast.core.operating import MethodScore
from promcast.core.replay import MethodResult, ReplayResult


def _score_json(score: MethodScore) -> dict[str, object]:
    data: dict[str, object] = asdict(score)
    data.update(
        precision=score.precision,
        recall=score.recall,
        false_alerts_per_week=score.false_alerts_per_week,
    )
    for key in ("precision", "recall", "false_alerts_per_week", "weeks", "coverage_80"):
        value = data[key]
        if isinstance(value, float):
            data[key] = round(value, 8)
    for key, values in (
        ("leads_s", score.leads_s),
        ("reset_s", score.reset_s),
        ("warning_times_s", score.warning_times_s),
        ("wql_by_lead", score.wql_by_lead),
    ):
        data[key] = [round(value, 8) if value is not None else None for value in values]
    data["series_warning_times_s"] = [
        [round(value, 8) for value in cluster] for cluster in score.series_warning_times_s
    ]
    return data


def _intervals(score: MethodScore) -> dict[str, tuple[float, float] | None]:
    clusters = [
        np.array(cluster, dtype=np.float64)
        for cluster in (score.series_warning_times_s or [score.warning_times_s])
    ]
    recall = (
        bootstrap_ci(
            [(cluster > 0).astype(np.float64) for cluster in clusters],
            lambda sample: float(np.mean(sample)),
        )
        if score.n_incidents
        else None
    )
    leads = [cluster[cluster > 0] for cluster in clusters]
    lead = (
        bootstrap_ci(leads, lambda sample: float(np.median(sample))) if score.caught_early else None
    )
    return {
        "recall_95": (round(recall[0], 8), round(recall[1], 8)) if recall else None,
        "median_lead_s_95": (round(lead[0], 8), round(lead[1], 8)) if lead else None,
    }


def report_json(result: ReplayResult) -> str:
    """Serialize sorted methods, complete counts and fixed precision without NaN."""
    methods: dict[str, object] = {}
    for name, method in sorted(result.methods.items()):
        methods[name] = {
            "point": asdict(method.point),
            "n_incidents_cal": method.calibration.n_incidents,
            "calibration": _score_json(method.calibration),
            "test": _score_json(method.test),
            "intervals": _intervals(method.test),
            "test_budget_exceeded": method.test.false_alerts_per_week
            > 1.5 * result.spec.budget_per_week,
            "fallbacks": method.fallbacks,
            "sweep": [
                {"point": asdict(point), "score": _score_json(score)}
                for point, score in sorted(method.sweep.items())
            ],
        }
    document = {
        "spec": asdict(result.spec),
        "threshold": round(result.threshold, 8),
        "split": result.split,
        "methods": methods,
    }
    return json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"


def report_markdown(result: ReplayResult) -> str:
    """Render the calibration budget beside test rates, counts and warning quality."""
    lines = [
        "# Replay",
        "",
        f"Calibration false-alert budget: {result.spec.budget_per_week:.4f} per series-week.",
        f"Threshold: {result.threshold:.4f}. Split index: {result.split}.",
        "",
        "| Method | n_incidents_cal | n incidents | n series | Series-weeks | Recall | "
        "Precision | Median lead (s) | False alerts/week | Coverage 80% | "
        "Recall 95% CI | Median lead 95% CI (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    notes: list[str] = []
    for name, method in sorted(result.methods.items()):
        test = method.test
        intervals = _intervals(test)
        formatted_intervals = [
            f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "unavailable"
            for interval in intervals.values()
        ]
        flag = (
            " (above 1.5 x budget)"
            if test.false_alerts_per_week > 1.5 * result.spec.budget_per_week
            else ""
        )
        lead = f"{median(test.leads_s):.2f}" if test.leads_s else "unavailable"
        coverage = f"{test.coverage_80:.4f}" if test.coverage_80 is not None else "unavailable"
        lines.append(
            f"| {name}{flag} | {method.calibration.n_incidents} | {test.n_incidents} | "
            f"{test.n_series} | {test.weeks:.6f} | {test.recall:.4f} | {test.precision:.4f} | "
            f"{lead} | {test.false_alerts_per_week:.4f} | {coverage} | "
            f"{formatted_intervals[0]} | {formatted_intervals[1]} |"
        )
        if method.fallbacks:
            notes.append(f"{name} fallback: {', '.join(method.fallbacks)}.")
    lines.extend(["", *notes, "", "Weighted quantile loss by lead is included in replay.json.", ""])
    return "\n".join(lines)


def _rank(method: MethodResult) -> tuple[float, float, int, str]:
    calibration = method.calibration
    if not calibration.n_incidents:
        return (0.0, calibration.false_alerts_per_week, method.point.warn_s, method.point.method)
    return (
        -calibration.recall,
        -median(calibration.leads_s) if calibration.leads_s else 0.0,
        method.point.warn_s,
        method.point.method,
    )


def recommend(result: ReplayResult, target: TargetConfig) -> str:
    """Recommend the best calibration-selected live method as a validated fragment."""
    eligible = [
        method
        for name, method in result.methods.items()
        if (name == "linear" or name.startswith("toto-"))
        and method.calibration.false_alerts_per_week <= result.spec.budget_per_week
    ]
    if not eligible:
        raise ValueError("no live-capable method meets the calibration false-alert budget")
    chosen = min(eligible, key=_rank)
    point = chosen.point
    window_s = (
        point.window_s if point.method == "linear" else int(target.linear_window.total_seconds())
    )
    context_s = max(int(target.context.total_seconds()), window_s)
    data: dict[str, object] = {
        "name": target.name,
        "query": target.query,
        "step": f"{result.spec.step_s}s",
        "context": f"{context_s}s",
        "horizon": f"{result.spec.horizon_steps * result.spec.step_s}s",
        "season": f"{int(target.season.total_seconds())}s",
        "direction": result.spec.direction,
        "threshold": result.threshold,
        "quantile": point.quantile,
        "warn": f"{point.warn_s}s",
        "method": "linear" if point.method == "linear" else "toto",
        "linear_window": f"{window_s}s",
        "max_series": target.max_series,
        "known_gauge": target.known_gauge,
    }
    TargetConfig.model_validate(data)
    best = min(result.methods.values(), key=_rank)
    comment = ""
    if best.point.method != "linear" and not best.point.method.startswith("toto-"):
        comment = f"# Replay-only calibration winner: {best.point.method}\n"
    fragment: dict[str, object] = {"targets": [data]}
    if point.method.startswith("toto-"):
        fragment["model"] = {"name": point.method}
    return comment + yaml.safe_dump(fragment, sort_keys=True)
