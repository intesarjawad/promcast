from pathlib import Path

import numpy as np
import pytest
from prometheus_client import CollectorRegistry
from test_scenario import ANCHOR, scenario

from promcast.core.forecast import run_cycle
from promcast.core.forecaster import Toto2Forecaster
from promcast.core.series import Series
from promcast.shell.config import load_config
from promcast.shell.exposition import Collector
from promcast.shell.store import Store


@pytest.mark.slow
@pytest.mark.parametrize("toto_model_dir", ["toto-2.0-22m"], indirect=True)
def test_startup_cycles_export_breach_inside_warning(toto_model_dir: Path):
    target = load_config(Path("demo/promcast.yaml"), environment={}).targets[0]
    forecaster = Toto2Forecaster(toto_model_dir, "toto-2.0-22m", threads=1)
    store = Store()
    registry = CollectorRegistry()
    registry.register(Collector(store, "0.1.0", forecaster.name, "test"))
    labels = (("instance", "scenario:9100"), ("job", "scenario"))
    # Include range-grid rounding and the minute of backfill before live scrapes.
    for offset_s in (-59, 0, 60, 120, 180):
        origin_ts = ANCHOR + offset_s
        values = np.array(
            [
                scenario.disk_free_bytes(origin_ts + minute * 60, ANCHOR)
                for minute in range(-1439, 1)
            ]
        )
        forecasts, skipped = run_cycle(
            target, [Series(labels, 60, origin_ts, values)], forecaster, origin_ts
        )
        assert skipped == []
        assert len(forecasts) == 2
        store.update(target, forecasts, skipped, {labels: float(values[-1])}, origin_ts)
        breach_ts = registry.get_sample_value(
            "promcast_breach_timestamp_seconds",
            {"target": target.name, "method": "toto", **dict(labels)},
        )
        assert breach_ts is not None, f"no breach timestamp at offset {offset_s}s"
        assert 0 < breach_ts - origin_ts < target.warn.total_seconds()
        assert scenario.disk_free_bytes(ANCHOR + 40 * 60, ANCHOR) == 0
