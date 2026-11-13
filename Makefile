.PHONY: check check-copy audit check-slow check-site demo demo-check demo-check-full demo-gif bench evidence coldstart oci-compose

check: check-copy
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright
	! grep -rnE 'sys\.exit|SystemExit' src --include='*.py' | grep -v src/promcast/shell/cli.py
	uv run pytest -m "not slow and not browser"
	uv run promcast rules --config promcast.example.yaml | diff - alerts/promcast.rules.yml
	uv run promcast rules --config demo/promcast.yaml | diff - demo/promcast.rules.yml
	@if command -v promtool >/dev/null 2>&1; then promtool test rules alerts/promcast.rules.test.yml; else echo 'promtool unavailable; rules execution not verified'; fi

check-copy:
	! git grep -nIiE '@(gmail|icloud)\.'
	uv run python scripts/check_numbers.py
	uv run python scripts/build_site.py --check --no-png

# Pre-launch audit: competitor facts at their sources, runtime licences, secrets in history.
audit: check-copy
	uv run python scripts/check_prices.py
	uv run --no-dev --with pip-licenses==5.5.5 pip-licenses --format=json | uv run --no-dev python scripts/check_licences.py
	gitleaks detect --log-opts=--all

check-slow:
	uv run pytest -m slow

check-site:
	npx --yes @biomejs/biome@2.6.5 check site/site.js site/style.css
	uv run pytest site/test_site.py -m browser

bench:
	uv run promcast bench --json

demo:
	docker compose up -d --build

demo-check:
	scripts/demo_check.sh --json demo.json

demo-check-full:
	scripts/demo_check.sh --full --json demo.json

demo-gif:
	uv run python scripts/demo_gif.py

evidence:
	uv run python scripts/evidence.py

coldstart:
	scripts/coldstart.sh

oci-compose:
	uv run python scripts/oci_compose.py
