"""Fail unless every competitor fact in docs/sources/prices.json still appears at its source.

Each quote, and each limit's qualifier, must still appear in the text of its page. Matching
ignores whitespace and markup, because pages split one sentence across several elements.
Public copy must also never call a limit a cap: limits are defaults raised on request.
"""

import argparse
import datetime
import html
import re
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, model_validator

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "docs/sources/prices.json"
COPY = ("README.md", "site")
HEADERS = {
    # Identifies the checker; several vendor servers reject clients without a browser token.
    "User-Agent": "Mozilla/5.0 (compatible; promcast-check-prices)",
    # grafana.com answers "gzip, deflate" with Brotli, which httpx cannot decode without an extra.
    "Accept-Encoding": "identity",
}
TIMEOUT_S = 30
# Some hosts intermittently answer a runner with a shell or interstitial page and status 200,
# so a fact is only reported once a fresh fetch of its page still misses it.
FETCH_ATTEMPTS = 3
RETRY_PAUSE_S = 5
_FORBIDDEN_COPY = re.compile(r"capped at", re.IGNORECASE)


class Fact(BaseModel):
    """One competitor fact as public copy quotes it."""

    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["price", "limit", "access"]
    url: str
    quote: str
    read: datetime.date
    qualifier: str | None = None

    @model_validator(mode="after")
    def _limits_carry_their_qualifier(self) -> Self:
        if self.kind == "limit" and not self.qualifier:
            raise ValueError(f"limit {self.id} needs the qualifier its source states beside it")
        return self


class Sources(BaseModel):
    """The dated facts file."""

    model_config = ConfigDict(extra="forbid")
    facts: list[Fact]


def _squeeze(text: str) -> str:
    return "".join(text.split())


def page_text(markup: str) -> str:
    """Reduce a page to its visible text with every whitespace run removed."""
    without_code = re.sub(r"<(script|style)\b.*?</\1\s*>", " ", markup, flags=re.S | re.I)
    return _squeeze(html.unescape(re.sub(r"<[^>]+>", " ", without_code)))


def _found_instead(page: str, wanted: str) -> str:
    """The page text from where the wanted text starts to match: what the server sent instead."""
    for size in range(len(wanted) - 1, 2, -1):
        at = page.find(wanted[:size])
        if at >= 0:
            return page[at : at + len(wanted) + 40]
    return "nothing like it"


def stale_facts(
    facts: Iterable[Fact],
    fetch: Callable[[str], str],
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Return why each fact no longer matches its source.

    A URL is fetched once when everything matches, and again (up to FETCH_ATTEMPTS in all) only
    while a fact is missing from it.
    """
    pages: dict[str, str] = {}
    fetches: dict[str, int] = {}
    problems: list[str] = []
    for fact in facts:
        expected = {"quote": fact.quote, "qualifier": fact.qualifier}
        wanted = {f: _squeeze(t) for f, t in expected.items() if t is not None}
        while True:
            if fact.url not in pages:
                pages[fact.url] = page_text(fetch(fact.url))
                fetches[fact.url] = fetches.get(fact.url, 0) + 1
            missing = [f for f, t in wanted.items() if t not in pages[fact.url]]
            if not missing or fetches[fact.url] >= FETCH_ATTEMPTS:
                break
            sleep(RETRY_PAUSE_S)
            del pages[fact.url]
        for field in missing:
            seen = _found_instead(pages[fact.url], wanted[field])
            if seen == "nothing like it":
                seen += f"; {len(pages[fact.url])} characters of visible text"
            problems.append(
                f"{fact.id}: the {field} read {fact.read} no longer appears at {fact.url}; "
                f"reread the page, then correct the fact, its date and every copy that cites it"
                f" (this fetch has: {seen})"
            )
    return problems


def capped_limits(root: Path, paths: Iterable[str] = COPY) -> list[str]:
    """Find copy that calls a raisable default limit a cap."""
    found: list[str] = []
    for name in paths:
        base = root / name
        files = sorted(base.rglob("*")) if base.is_dir() else [base]
        for path in files:
            if not path.is_file() or path.suffix not in {".md", ".html", ".js", ".json"}:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _FORBIDDEN_COPY.search(line):
                    found.append(
                        f"{path.relative_to(root)}:{number}: say 'by default, raised on request' "
                        "instead of 'capped at'"
                    )
    return found


def _fetch(client: httpx.Client) -> Callable[[str], str]:
    def fetch(url: str) -> str:
        response = client.get(url)
        response.raise_for_status()
        return response.text

    return fetch


def main(argv: list[str] | None = None) -> int:
    """Check every fact at its source and the copy that quotes them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=PRICES)
    arguments = parser.parse_args(argv)
    try:
        sources = Sources.model_validate_json(arguments.prices.read_text(encoding="utf-8"))
        with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=TIMEOUT_S) as client:
            problems = stale_facts(sources.facts, _fetch(client))
    except (OSError, ValueError, httpx.HTTPError) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    problems.extend(capped_limits(ROOT))
    for problem in problems:
        sys.stderr.write(f"{problem}\n")
    if problems:
        return 1
    sys.stdout.write(f"Every fact still matches its source ({len(sources.facts)} checked)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
