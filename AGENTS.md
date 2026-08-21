# Agent Guidelines

- This is a **uv-managed** Python project. Prefer `uv run` for all Python and project tools; do not search for or invoke a system Python or `.venv/bin/python` directly.
- Set up the environment with `uv sync` (use `uv sync --extra docs` for documentation work).
- Run tests with `uv run pytest tests`; if pytest is not installed in the project environment, use `uv run --with pytest pytest tests` rather than falling back to system Python.
- Run Python scripts with `uv run python path/to/script.py`.
- Build the package with `uv build`.
- Build documentation locally with `uv run --extra docs sphinx-build -M html docs/source docs/build`.
- Keep generated artifacts (`build/`, `dist/`, `docs/build/`, caches, and coverage output) out of commits.
- The import namespace is `pyeeg`; the distribution/project name is `natmeeg`.
- Before finishing, run the relevant tests and `git diff --check`.
