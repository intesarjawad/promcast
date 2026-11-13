# 0000: Pin the Python toolchain and use Linux CPU wheels

Status: accepted, 2026-11-05.

The project pins Python 3.12 and every Python dependency version.
Reproducible checks must run before model weights or runtime code exist.

Use uv with a committed universal lockfile, ruff lint and format, strict pyright,
and pytest. The rule selection and marker definitions live in `pyproject.toml`. Use
`uv_build`, selected by `uv init --package`, pinned to the installed uv version.
Resolve Linux torch only from the explicit PyTorch CPU index to avoid pulling
CUDA dependencies. Other platforms use PyPI torch wheels.

`make check` and the local hooks use these project tools. CI runs the same fast
checks. Slow and browser checks join CI when their suites exist; selecting either
suite now fails because no tests match. Runtime architecture and model
decisions remain with their scheduled tasks.
