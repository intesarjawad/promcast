import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import respx
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
if TYPE_CHECKING:
    from scripts import check_prices
else:
    spec = importlib.util.spec_from_file_location("check_prices", ROOT / "scripts/check_prices.py")
    assert spec is not None
    assert spec.loader is not None
    check_prices = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_prices)

PAGE = """<html><head><style>p { color: red }</style><script>var price = "$1";</script></head>
<body><p>Pro <span>$</span><span>19</span> / month&nbsp;+ usage</p>
<ul><li>10 forecasts per instance</li></ul>
<p>To request an increase, <a href="/contact">contact us</a>.</p></body></html>"""


def _fact(**overrides: str) -> "check_prices.Fact":
    fields = {
        "id": "example",
        "kind": "price",
        "url": "https://vendor.example/pricing",
        "quote": "Pro $19 / month + usage",
        "read": "2026-11-09",
        **overrides,
    }
    return check_prices.Fact.model_validate(fields)


def test_page_text_keeps_visible_text_without_whitespace():
    text = check_prices.page_text(PAGE)
    assert "Pro$19/month+usage" in text
    assert "price" not in text
    assert "color" not in text


def test_quote_split_across_elements_still_matches():
    assert check_prices.stale_facts([_fact()], lambda _: PAGE) == []


def test_changed_quote_and_missing_qualifier_are_reported():
    facts = [
        _fact(id="moved", quote="Pro $29 / month + usage"),
        _fact(
            id="limit",
            kind="limit",
            quote="10 forecasts per instance",
            qualifier="Limits cannot be raised",
        ),
    ]
    problems = check_prices.stale_facts(facts, lambda _: PAGE)
    assert len(problems) == 2
    assert problems[0].startswith("moved: the quote read 2026-11-09 no longer appears at")
    assert problems[1].startswith("limit: the qualifier read 2026-11-09")
    assert all("correct the fact, its date and every copy" in problem for problem in problems)
    assert "(this fetch has: Pro$19/month+usage" in problems[0]


def test_each_url_is_fetched_once():
    fetched: list[str] = []

    def fetch(url: str) -> str:
        fetched.append(url)
        return PAGE

    check_prices.stale_facts([_fact(), _fact(id="again")], fetch)
    assert fetched == ["https://vendor.example/pricing"]


def test_a_shell_page_is_refetched_before_a_fact_is_reported():
    pages = iter(["<html><body>Loading</body></html>", PAGE])
    assert check_prices.stale_facts([_fact()], lambda _: next(pages), sleep=lambda _: None) == []


def test_a_fact_missing_from_every_fetch_is_reported_with_the_page_size():
    fetched: list[str] = []

    def fetch(url: str) -> str:
        fetched.append(url)
        return "<html><body>Loading</body></html>"

    problems = check_prices.stale_facts([_fact()], fetch, sleep=lambda _: None)
    assert len(fetched) == 3
    assert len(problems) == 1
    assert "nothing like it; 7 characters of visible text" in problems[0]


def test_a_limit_needs_its_qualifier():
    with pytest.raises(ValidationError, match="needs the qualifier"):
        _fact(kind="limit")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        _fact(note="free text")


def test_committed_facts_are_dated_and_qualified():
    sources = check_prices.Sources.model_validate_json(
        (ROOT / "docs/sources/prices.json").read_text()
    )
    assert sources.facts
    assert len({fact.id for fact in sources.facts}) == len(sources.facts)
    assert all(fact.url.startswith("https://") for fact in sources.facts)


def test_capped_limits_are_found_in_readme_and_site_only(tmp_path: Path):
    (tmp_path / "README.md").write_text("Ten forecasts, Capped at ten.\n")
    (tmp_path / "site").mkdir()
    (tmp_path / "site/index.html").write_text("<p>ok</p>\n<p>capped at 10</p>\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/protocol.md").write_text("warn (capped at horizon)\n")
    assert check_prices.capped_limits(tmp_path) == [
        "README.md:1: say 'by default, raised on request' instead of 'capped at'",
        "site/index.html:2: say 'by default, raised on request' instead of 'capped at'",
    ]


@respx.mock
def test_main_reports_a_stale_fact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(check_prices, "ROOT", tmp_path)
    prices = tmp_path / "prices.json"
    fact = _fact(quote="Pro $29 / month + usage").model_dump(mode="json")
    prices.write_text(json.dumps({"facts": [fact]}))
    route = respx.get("https://vendor.example/pricing").respond(200, text=PAGE)
    assert check_prices.main(["--prices", str(prices)]) == 1
    assert route.calls.last.request.headers["accept-encoding"] == "identity"
    assert "example: the quote" in capsys.readouterr().err


@respx.mock
def test_main_passes_when_every_fact_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(check_prices, "ROOT", tmp_path)
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"facts": [_fact().model_dump(mode="json")]}))
    respx.get("https://vendor.example/pricing").respond(200, text=PAGE)
    assert check_prices.main(["--prices", str(prices)]) == 0
    assert capsys.readouterr().out == "Every fact still matches its source (1 checked)\n"


@respx.mock
def test_unreachable_source_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"facts": [_fact().model_dump(mode="json")]}))
    respx.get("https://vendor.example/pricing").respond(404)
    assert check_prices.main(["--prices", str(prices)]) == 1
    assert "404" in capsys.readouterr().err
