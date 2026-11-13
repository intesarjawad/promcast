"""Regenerate the Toto 22m golden fixture from a configured local model directory."""

import argparse
from pathlib import Path

import numpy as np

from promcast.core.forecaster import Toto2Forecaster
from promcast.core.series import Series


def main() -> None:
    """Write the fixed 512-step sine-plus-trend context and 96-step forecast."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    steps = np.arange(512, dtype=np.float64)
    values = 10 + np.sin(steps / 5) + steps / 100
    model = Toto2Forecaster(args.model_dir, "toto-2.0-22m", threads=1)
    quantiles = model.forecast([Series((), 60, 0, values)], 96)[0]
    destination = Path(__file__).parent / "fixtures/golden/toto22m_sine.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, values=values, quantiles=quantiles)


if __name__ == "__main__":
    main()
