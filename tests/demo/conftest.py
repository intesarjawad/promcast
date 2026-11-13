import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.core import conftest as model_fixtures
else:
    spec = importlib.util.spec_from_file_location(
        "model_fixtures", Path(__file__).parents[1] / "core" / "conftest.py"
    )
    assert spec is not None
    assert spec.loader is not None
    model_fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_fixtures)

toto_model_dir = model_fixtures.toto_model_dir
