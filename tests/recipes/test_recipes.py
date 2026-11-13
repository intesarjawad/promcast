import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from promcast.core.alerting import Direction
from promcast.core.recipes import Recipe, candidate_query, scan_target
from promcast.core.series import align
from promcast.shell.discover import load_recipes, match

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.mark.parametrize("recipe", load_recipes(), ids=[recipe.name for recipe in load_recipes()])
def test_each_recipe_matches_its_names_and_range_fixture(recipe: Recipe):
    names = json.loads((FIXTURES / f"labels_{recipe.name}.json").read_text())["data"]
    assert match(set(names), load_recipes()) == [recipe]
    data = json.loads((FIXTURES / f"range_{recipe.name}.json").read_text())["data"]
    series, rejected = align(data, 15, 1700000000 + 79 * 15, 80)
    assert len(series) == 1
    assert not rejected
    assert series[0].values.size == 80
    assert scan_target(recipe, 15, 50).name == recipe.name
    assert recipe.source.scheme == "https"


def test_synthetic_stock_names_match_only_self_recipes():
    names = json.loads((FIXTURES / "labels_stock_prometheus.json").read_text())["data"]
    assert [r.name for r in match(set(names), load_recipes())] == [
        "prometheus-self",
        "prometheus-tsdb",
    ]


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("below", "(a / b) and bottomk(50, last_over_time((a / b)[10m:] @ 1700000000))"),
        ("above", "(a / b) and topk(50, last_over_time((a / b)[10m:] @ 1700000000))"),
    ],
)
def test_candidate_golden(direction: Direction, expected: str):
    assert candidate_query("a / b", direction, 50, 1700000000) == expected


def test_recipe_override_and_unknown_field(tmp_path: Path):
    recipe = load_recipes()[0]
    changed = recipe.model_dump(mode="json") | {"threshold": 0.1}
    (tmp_path / "override.yaml").write_text(yaml.safe_dump(changed))
    assert load_recipes(tmp_path)[0].threshold == 0.1
    with pytest.raises(ValidationError):
        Recipe.model_validate(changed | {"typo": 1})
