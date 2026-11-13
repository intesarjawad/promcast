from promcast.core.alerting import Direction

# Direction and threshold per E1 scenario. The leaks, the step and the saturation
# curve fall through zero; the seasonal peak and the noisy flat line never breach,
# so any alert on them is a false alert.
SCENARIOS: dict[str, tuple[Direction, float]] = {
    "linear_leak": ("below", 0.0),
    "accelerating_leak": ("below", 0.0),
    "daily_seasonal_peak": ("above", 100.0),
    "step_change": ("below", 0.0),
    "saturation_curve": ("below", 0.0),
    "noisy_flat": ("below", 0.0),
    "gappy_accelerating_leak": ("below", 0.0),
}

# E1 scenarios that replay must refuse: a cumulative counter is not a gauge, so it
# has no direction or threshold to forecast against.
REFUSED_SCENARIOS: tuple[str, ...] = ("counter",)
