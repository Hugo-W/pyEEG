# Next Steps — ai-revamp branch

This is the recommended order for future work.

## Step 1: Clean the docs

Start small and safe.

What to check:
- `README.md` wording and naming
- `docs/source/` references to `pyEEG` vs `natMEEG`
- installation / usage examples

Acceptance target:
- docs read cleanly
- old naming is either removed or intentionally marked as legacy
- no broken links in the obvious entry points

## Step 2: Stabilize the public API

Before refactoring internals, make sure the current entry points are covered.

Focus on:
- `pyeeg/__init__.py`
- core imports from `pyeeg.models`, `pyeeg.cca`, `pyeeg.utils`
- compatibility with existing names and expected signatures

Acceptance target:
- old imports still work
- no accidental API breakage

## Step 3: Clarify solver boundaries

This is the most important architectural cleanup.

Current rough reading:
- `pyeeg/models.py` = high-level TRF / regression / estimator logic
- `pyeeg/solvers.py` = reusable low-level linear solvers used by models
- `solver.py` = standalone solver experiments / alternate implementations

Questions to answer:
- which solver module is canonical?
- should `solver.py` stay standalone or be folded into the package layer?
- can `_svd_regress` / related helper code be centralized?

Acceptance target:
- one clear solver story
- no duplicated math without a reason
- compatibility preserved via wrappers if needed

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
