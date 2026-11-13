"""Browser checks for the published static site."""

# ruff: noqa: S101

import json
import os
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from playwright.sync_api import Page, sync_playwright

SITE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def site_url() -> Iterator[str]:
    """Serve the site over HTTP for module loading."""
    handler = partial(SimpleHTTPRequestHandler, directory=str(SITE))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/index.html"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture(params=[(1440, 900), (390, 844)])
def page(request: pytest.FixtureRequest) -> Iterator[Page]:
    """Launch an isolated browser page."""
    with sync_playwright() as playwright:
        browser_path = os.environ.get("PROMCAST_BROWSER")
        browser = playwright.chromium.launch(executable_path=browser_path)
        width, height = request.param
        page = browser.new_page(
            viewport={"width": width, "height": height}, reduced_motion="reduce"
        )
        yield page
        browser.close()


@pytest.mark.browser
def test_tabs_and_mobile_navigation(page: Page, site_url: str) -> None:
    """Tabs connect to panels and section links remain available on mobile."""
    page.goto(site_url)
    tabs = page.get_by_role("tab")
    assert tabs.count() == 4
    for tab in tabs.all():
        tab.click()
        panel = page.locator(f"#{tab.get_attribute('aria-controls')}")
        assert panel.is_visible()
        assert panel.get_attribute("aria-labelledby") == tab.get_attribute("id")
        assert tab.get_attribute("aria-selected") == "true"
    assert page.get_by_role("link", name="Terminal").is_visible()
    assert "unpaged" not in page.locator("body").inner_text().lower()
    payload = page.locator("#sheet-data").text_content()
    assert payload is not None
    data = json.loads(payload)
    series = data["series"][0]
    gap = round(abs(series["d"] - series["p"]) / 60)
    assert page.locator("#rd-left").inner_text().startswith(f"{gap} min")
    tabs.first.focus()
    tabs.first.press("ArrowRight")
    assert tabs.nth(1).get_attribute("aria-selected") == "true"


@pytest.mark.browser
def test_blocked_script_keeps_terminal_visible(page: Page, site_url: str) -> None:
    """The terminal boards remain visible if the module cannot load."""
    page.route("**/site.js", lambda route: route.abort())
    page.goto(site_url)
    assert page.locator(".state:visible").count() == 4


@pytest.mark.browser
def test_cpu_command_clipboard_and_local_icon(page: Page, site_url: str) -> None:
    """Copy the CPU quickstart and serve the declared icon without failed requests."""
    failures: list[str] = []
    page.on(
        "response",
        lambda response: failures.append(response.url) if response.status >= 400 else None,
    )
    page.add_init_script(
        "Object.defineProperty(navigator, 'clipboard', {value: {writeText: async text => {"
        "window.copiedCommand = text;}}});"
    )
    page.goto(site_url)
    page.locator("button[data-copy]").click()
    assert page.evaluate("window.copiedCommand") == "uvx --torch-backend cpu promcast scan"
    icon = page.locator('link[rel="icon"]').get_attribute("href")
    assert icon
    response = page.request.get(site_url.rsplit("/", 1)[0] + "/" + icon)
    assert response.ok
    assert response.headers["content-type"].startswith("image/")
    assert not failures
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
