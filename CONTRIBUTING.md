# Contributing

Read the [engineering standards](docs/engineering-standards.md) before making
changes.

Use Python 3.12 and uv 0.13.7. Set up and check the repository with:

```sh
uv sync --locked
make check
```

The local hooks in `.pre-commit-config.yaml` run formatting, lint and strict
type checks through uv. If pre-commit is installed, run `pre-commit install`.
Tests run through `make check` and CI. Do not bypass failing checks.

Use `make check-slow` when changing model code and `make check-site` when changing
the site.

Keep changes focused, test behavior and document significant decisions in
`docs/adr/`. Use semantic versioning and conventional commit subjects. Update
[the changelog](CHANGELOG.md) for user-visible changes.

Do not include credentials, private metrics or unredacted diagnostic output in
issues or patches. See [security reporting](SECURITY.md) and the
[code of conduct](CODE_OF_CONDUCT.md).
