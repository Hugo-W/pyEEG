# Agent Guidelines

- This is a **uv-managed** Python project. Prefer `uv run` for all Python and project tools; do not search for or invoke a system Python or `.venv/bin/python` directly.
- Set up the environment with `uv sync` (use `uv sync --extra docs` for documentation work).
- Run tests with `uv run pytest` (config lives in `pyproject.toml`; `testpaths = tests` so no path is needed). If pytest is not installed in the project environment, use `uv run --with pytest pytest` rather than falling back to system Python.
- Run Python scripts with `uv run python path/to/script.py`.
- Build the package with `uv build`.
- Build documentation locally with `uv run --extra docs sphinx-build -M html docs/source docs/build`.
- Keep generated artifacts (`build/`, `dist/`, `docs/build/`, caches, and coverage output) out of commits.
- The import namespace is `pyeeg`; the distribution/project name is `natmeeg`.
- Before finishing, run the relevant tests and `git diff --check`.

## Testing workflow (for agents)

Do **not** run the full suite after every edit — it is slow and mostly redundant. Follow this tiered policy:

1. **Targeted run (during edits):** run only the test file(s) matching the module(s) you changed. Test files map 1:1 to `pyeeg` submodules:
   - `pyeeg.models.trf` → `test_lagged_cache`, `test_trf_block_order`, `test_robust_trf`, `test_feature_alphas`
   - `pyeeg.solvers` → `test_solvers`, `test_solver_pattern`, `test_trf_block_order`, `test_robust_trf`
   - `pyeeg.utils` → `test_utils`, `test_weighted_samples`, `test_regression`, `test_feature_alphas`, `test_trf_block_order`
   - `pyeeg.simulate` → `test_simulate`, `test_regression`
   - `pyeeg.cca` → `test_cca`
   - `pyeeg.connectivity` → `test_connectivity`
   - `pyeeg.features.alignment` → `test_features_alignment`, `test_features_pipeline`
   - `pyeeg.features.syntactic_features` → `test_features_syntactic`
   - `pyeeg.features.reduction` → `test_features_reduction`
   - `pyeeg.features.llm_features` → `test_features_llm`
   - `pyeeg.features.pipeline` → `test_features_pipeline`
   - `pyeeg.gammatone` → `test_gammatone`

   Example: after editing `pyeeg/models/trf.py` → `uv run pytest tests/test_trf_block_order.py tests/test_robust_trf.py -x`

2. **Fast pre-check:** `uv run pytest -m "not slow and not llm" -x` — skips torch-dependent and distilgpt2 tests; covers the lightweight majority of the suite.

3. **After a failing run:** re-check only failures with `uv run pytest --lf -x` (last-failed).

4. **Full suite (end of session, before declaring done):** `uv run pytest` — the final validation gate. Use `-m "not llm"` if `torch`/`transformers` are not installed (i.e. without `uv sync --extra features`).

`--durations=10` is always on (see `pyproject.toml`); watch its output to spot tests worth marking `slow`.
