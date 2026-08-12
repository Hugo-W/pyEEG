# AI Revamp Guide

This branch is intentionally narrow in scope.

The actionable checklists now live in:
- `TODO.md`
- `NEXT_STEPS.md`

> **Status:** Historical snapshot written before issues #4–#10 were completed (2026-06).
> Some observations below are outdated — treat `TODO.md` / `NEXT_STEPS.md` as the live
> checklists.

## Branch goals

- Write cleaner, clearer documentation
- Refactor redundant code
- Modularize where it reduces coupling
- Keep everything backward compatible
- Extend and strengthen tests
- Verify the library still works after each change

## Rules of engagement

- Prefer small, safe edits over sweeping rewrites
- Preserve public APIs and import paths unless there is a compatibility shim
- If something is renamed internally, keep compatibility wrappers for old names
- Add or update tests before removing old code paths
- Treat any behavior change as a regression risk until verified

## Cheap repo scan: where to start

A quick scan suggests these are the best first targets:

### 1) Documentation cleanup

Start with the README and docs because they still carry old naming and dated wording.

- `README.md` — only intentional legacy references remain (lines 8, 17); naming otherwise updated to natMEEG
- documentation and examples still refer to the old project name
- the repo description now appears to be `natMEEG`, so docs should reflect that consistently

### 2) Refactoring / modularization

The core logic seems concentrated in a few large modules:

- `pyeeg/models.py` — TRF and modeling logic
- `pyeeg/utils.py` — shared helpers and signal utilities
- `pyeeg/cca.py` — CCA-related code
- `pyeeg/connectivity.py` — connectivity methods and TODOs
- `pyeeg/preprocess.py` — preprocessing helpers
- `pyeeg/io.py` — file and data loading utilities

Likely opportunities:

- extract repeated helper logic into shared utilities
- separate data conversion / validation from core algorithms
- reduce cross-module imports where possible
- keep existing module entry points intact for compatibility

### 3) Backward compatibility checks

Watch the old `pyEEG` naming carefully:

- `pyeeg/__init__.py` still exposes the package under the historical name
- code and docs still use `pyEEG` references in multiple places
- any rename should be additive, not destructive

### 4) Tests

The tests need attention too:

- `tests/test_utils.py` placeholder has been filled (Issue #9, ~85 tests)
- `tests/test_connectivity.py` also has a placeholder test
- `tests/test_gammatone.py` mixes direct execution style with tests and should be reviewed for robustness
- several existing tests look like they were written for sanity checks rather than long-term regression coverage

## Suggested order

1. Clean up docs and naming
2. Add or fix regression tests for the most used APIs
3. Refactor internals in small pieces
4. Keep compatibility shims in place
5. Run the relevant tests after every meaningful change

## Practical note for future agents

Before changing code, check whether a public function, import path, or example notebook depends on it. If yes, keep a compatibility layer or update the docs and tests together.
