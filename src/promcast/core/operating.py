from dataclasses import dataclass, field
from statistics import median


@dataclass(frozen=True, order=True)
class OperatingPoint:
    """One method's warning threshold and forecast parameters, in seconds."""

    method: str
    warn_s: int
    quantile: float = 0.5
    window_s: int = 0
    margin_k: float = 0.0


@dataclass(frozen=True)
class MethodScore:
    """Episode counts and incident-aligned warning times for one evaluation split."""

    n_incidents: int
    caught_early: int
    late: int
    n_episodes: int
    matched: int
    weeks: float
    leads_s: list[float] = field(default_factory=list[float])
    reset_s: list[float] = field(default_factory=list[float])
    warning_times_s: list[float] = field(default_factory=list[float])
    coverage_80: float | None = None
    wql_by_lead: list[float | None] = field(default_factory=list[float | None])
    n_series: int = 1
    series_warning_times_s: list[list[float]] = field(default_factory=list[list[float]])

    @property
    def precision(self) -> float:
        """Count late episodes as matched for precision."""
        return self.matched / self.n_episodes if self.n_episodes else 0.0

    @property
    def recall(self) -> float:
        """Count each early-caught incident once."""
        return self.caught_early / self.n_incidents if self.n_incidents else 0.0

    @property
    def false_alerts_per_week(self) -> float:
        """Normalize unmatched episodes by series-weeks of exposure."""
        return (self.n_episodes - self.matched) / self.weeks


def pool_scores(scores: list[MethodScore]) -> MethodScore:
    """Pool counts and exposure, including series without incidents."""
    if not scores:
        raise ValueError("pooling requires at least one series")
    return MethodScore(
        n_incidents=sum(s.n_incidents for s in scores),
        caught_early=sum(s.caught_early for s in scores),
        late=sum(s.late for s in scores),
        n_episodes=sum(s.n_episodes for s in scores),
        matched=sum(s.matched for s in scores),
        weeks=sum(s.weeks for s in scores),
        leads_s=[lead for s in scores for lead in s.leads_s],
        reset_s=[reset for s in scores for reset in s.reset_s],
        warning_times_s=[lead for s in scores for lead in s.warning_times_s],
        n_series=sum(s.n_series for s in scores),
        series_warning_times_s=[
            cluster for s in scores for cluster in (s.series_warning_times_s or [s.warning_times_s])
        ],
    )


def choose_operating_point(
    calibration: dict[OperatingPoint, MethodScore], budget: float
) -> OperatingPoint:
    """Select on calibration only; fail explicitly if no point meets the budget."""
    if not calibration or budget < 0:
        raise ValueError("selection requires calibration scores and a nonnegative budget")
    if not any(s.n_incidents for s in calibration.values()):
        return min(
            calibration,
            key=lambda p: (calibration[p].false_alerts_per_week, p.warn_s, p),
        )
    feasible = [p for p, s in calibration.items() if s.false_alerts_per_week <= budget]
    if not feasible:
        raise ValueError("no calibration operating point meets the false-alert budget")
    return min(
        feasible,
        key=lambda p: (
            -calibration[p].recall,
            -median(calibration[p].leads_s) if calibration[p].leads_s else 0.0,
            p.warn_s,
            p,
        ),
    )


def operating_points(method: str, horizon_s: int) -> list[OperatingPoint]:
    """Build the operating-point sweep, capping warning times at the horizon."""
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    warnings = sorted({min(minutes * 60, horizon_s) for minutes in (10, 20, 30, 45, 60, 90, 120)})
    if method == "linear":
        return [OperatingPoint(method, w, window_s=h * 3600) for w in warnings for h in (1, 6, 24)]
    if method == "static-margin":
        return [OperatingPoint(method, w, margin_k=k) for w in warnings for k in (0.5, 1, 2, 3)]
    quantiles = (0.5,) if method == "seasonal-naive" else (0.5, 0.7, 0.9)
    return [OperatingPoint(method, w, q) for w in warnings for q in quantiles]
