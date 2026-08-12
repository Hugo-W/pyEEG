# Next Steps — ai-revamp branch

This is the recommended order for future work.

## Step 1: Clean the docs

**Status: COMPLETED (Issues #4, #5)**

Start small and safe.

What to check:
- `README.md` wording and naming ✅
- `docs/source/` references to `pyEEG` vs `natMEEG` ✅
- installation / usage examples

**Completed:**
- Issue #4: Cleaned README.md naming and references
  - Updated documentation link from pyEEG-docs to natMEEG-docs
  - Changed caution notice to reflect active maintenance
  - Fixed PDF documentation filename reference
- Issue #5: Audited docs/source/ for naming consistency
  - Updated index.rst, conf.py, about.rst, usage.rst to use natMEEG consistently
  - Kept historical pyEEG references where appropriate
  - Preserved pyeeg package name in Sphinx directives and code blocks

Acceptance target:
- docs read cleanly ✅
- old naming is either removed or intentionally marked as legacy ✅
- no broken links in the obvious entry points ✅

## Step 2: Stabilize the public API

**Status: COMPLETED (Issue #6)**

Before refactoring internals, make sure the current entry points are covered.

Focus on:
- `pyeeg/__init__.py` ✅
- core imports from `pyeeg.models`, `pyeeg.cca`, `pyeeg.utils`
- compatibility with existing names and expected signatures

**Completed:**
- Updated package docstring from pyEEG to natMEEG
- Added `__all__` list to define public API explicitly
- Verified all existing imports still work (backward compatibility maintained)
- Verified submodules (connectivity, io, models, preprocess, vizu, utils, simulate) are exposed
- Verified key classes (TRFEstimator, CCA_Estimator, MultichanWienerFilter, Whitener, mCCA) are exposed

Acceptance target:
- old imports still work ✅
- no accidental API breakage ✅

## Step 3: Clarify solver boundaries

**Status: COMPLETED (Issue #8)**

This is the most important architectural cleanup.

Current rough reading:
- `pyeeg/models.py` = high-level TRF / regression / estimator logic
- `pyeeg/solvers.py` = reusable low-level linear solvers used by models
- `solver.py` (root-level) = DELETED - merged into pyeeg/solvers.py

Questions to answer:
- which solver module is canonical? ✅ (pyeeg/solvers.py)
- should `solver.py` stay standalone or be folded into the package layer? ✅ (deleted)
- can `_svd_regress` / related helper code be centralized? ✅ (yes, in solvers.py, not duplicated)

**Completed:**
- Issue #8: Merged root solver.py into pyeeg/solvers.py
  - added: svd_solver, conjugate_gradient, incomplete_cholesky_preconditioner, diagonal_preconditioner
  - deleted root-level solver.py (no duplicate code)
  - added tests: test_solvers.py, performance_solvers.ipynb
  - _svd_regress remains only in solvers.py (not duplicated) ✅

Acceptance target:
- one clear solver story ✅
- no duplicated math without a reason ✅
- compatibility preserved via wrappers if needed ✅

## Step 4: Modularize the large modules

**Status: NOT STARTED** — `pyeeg/models.py` (~1058 lines), `pyeeg/io.py` (~958),
`pyeeg/utils.py` (~738) remain large; no decomposition has happened yet. Related: #13.

Target likely hotspots:
- `pyeeg/models.py`
- `pyeeg/utils.py`
- `pyeeg/connectivity.py`
- `pyeeg/io.py`
- `pyeeg/preprocess.py`

Good decomposition candidates:
- shared regression helpers
- lag / design-matrix helpers
- validation and shape handling
- plotting / visualization helpers

Acceptance target:
- smaller modules with clearer responsibilities
- no regression in behavior

## Step 5: Add or fix tests

**Status: PARTIALLY COMPLETED (Issues #8, #9, #10)**

Start with the weakest spots first:
- `tests/test_utils.py` placeholder filled (Issue #9 — ~85 tests covering lag_matrix values/shapes/block_order, shift helpers, deprecated kwargs)
- `tests/test_connectivity.py` currently contains a placeholder
- `tests/test_gammatone.py` should be checked for robust assertions

Then add coverage for:
- TRF fitting behavior ✅ (Issue #10: 8 tests in test_regression.py)
- CCA sanity checks ✅ (Issue #10: 5 tests in test_regression.py)
- connectivity metrics
- backwards-compatible imports

**Completed:**
- Issue #8: test_solvers.py (4 solver tests + TRF basic test) + performance_solvers.ipynb
- Issue #10: test_regression.py (8 TRF + 5 CCA simulation-based regression tests)
- Tests use pyeeg.simulate utilities (dummy_trf_kernel, simulate_pulse_inputs, etc.)

**Known bug:** `models.py:416` t-value computation breaks with `fit_intercept=False`
(the unconditional `[1:, :]` slice assumes an intercept row). Note: the earlier claim
that tests use `alpha > 0` to skip stats is wrong — `tests/test_solvers.py::test_basic_trf`
uses `fit_intercept=False` with `alpha=None` and currently FAILS on this path
(100/101 core tests pass). Tracked in issue #25.

Acceptance target:
- tests protect the refactor
- no placeholder tests remain for core paths

## Step 6: Run a small verification pass

Minimum checks:
- import the package
- run targeted tests
- confirm docs still describe the intended project name and usage

## Step 7: Banded ridge regularization

**Status: PLANNED**

`lag_matrix` now has a `block_order` parameter enabling per-feature column blocks:
- `'lags'` (default): legacy ordering, used by current TRFEstimator
- `'features'`: per-feature blocks, enables banded ridge

Remaining work:
- Add `block_order` parameter to TRFEstimator (constructor or fit)
- Implement banded ridge solver: per-feature alpha as block-diagonal matrix in XtX
- Update coef_ reshape in TRFEstimator.fit to handle both column orderings
- Add tests for banded ridge path

Acceptance target:
- TRFEstimator accepts array-like alpha (one per feature) when `block_order='features'`
- scalar alpha still works as before
- reshape of coef_ is correct for both orderings

## Recommended rule of thumb

Make the next change only after you can answer:
- what was the public behavior before?
- what is the compatibility risk?
- what test proves it still works?
