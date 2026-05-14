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

Start with the weakest spots first:
- `tests/test_utils.py` currently contains a placeholder
- `tests/test_connectivity.py` currently contains a placeholder
- `tests/test_gammatone.py` should be checked for robust assertions

Then add coverage for:
- TRF fitting behavior
- CCA sanity checks
- connectivity metrics
- backwards-compatible imports

Acceptance target:
- tests protect the refactor
- no placeholder tests remain for core paths

## Step 6: Run a small verification pass

Minimum checks:
- import the package
- run targeted tests
- confirm docs still describe the intended project name and usage

## Recommended rule of thumb

Make the next change only after you can answer:
- what was the public behavior before?
- what is the compatibility risk?
- what test proves it still works?
