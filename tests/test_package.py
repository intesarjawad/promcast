from importlib.metadata import version

import promcast


def test_installed_package():
    assert promcast.__name__ == "promcast"
    assert version("promcast") == "0.1.0"
