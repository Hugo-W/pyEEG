# TODO — ai-revamp branch

This branch is for a focused cleanup of the project while preserving compatibility.

## Non-negotiables

- Keep the existing public API working
- Preserve import paths where possible
- Stay backward compatible with old `pyEEG`-era naming where needed
- Add or strengthen tests before removing or reshaping behavior
- Verify the package still runs after each meaningful change

## Main workstreams

### 1) Documentation cleanup

Priority areas:
- `README.md`
- `docs/source/`
- example notebooks / tutorials

Weak points seen in the quick scan:
- several stale `pyEEG` references remain
- docs still describe the old project name alongside `natMEEG`
- some wording is dated or unmaintained-sounding

Goal:
- make the docs clearer, shorter, and consistent with the current project name
- keep examples accurate and easy to follow

### 2) Refactor redundant code

Priority areas:
- `pyeeg/models.py`
- `pyeeg/utils.py`
- `pyeeg/cca.py`
- `pyeeg/connectivity.py`
- `pyeeg/preprocess.py`
- `pyeeg/io.py`

Weak points seen in the scan:
- `models.py` is large and mixes several model families
- some helper logic appears duplicated across modeling and solver code
- `connectivity.py` already carries a long TODO block

Goal:
- extract repeated logic into shared helpers
- reduce module coupling
- keep behavior stable

### 3) Modularize the architecture

Current rough split:
- `models.py` = high-level estimators and model fitting logic
- `solvers.py` = low-level linear regression helpers used by models
- `solver.py` = standalone experimental / alternate solver implementations

Weak point:
- the solver story is not yet cleanly separated

Goal:
- decide which solver layer is the canonical one
- move duplicated math into one place
- keep compatibility wrappers if any public import path changes

### 4) Extend tests

Priority areas:
- `tests/test_utils.py`
- `tests/test_connectivity.py`
- `tests/test_gammatone.py`
- CCA and TRF-related coverage

Weak points seen in the scan:
- some tests are placeholders (`pass`)
- some tests behave more like scripts than assertions
- coverage for public entry points is uneven

Goal:
- turn placeholder checks into real regression tests
- protect the most-used public APIs
- verify backward compatibility after refactors

### 5) Verify everything still works

Goal:
- run the relevant tests after each chunk
- check import compatibility
- confirm docs and examples still reference the right names

## Quick notes for future agents

- `pyeeg/__init__.py` still exposes the historical package name and should be treated as compatibility-sensitive
- `connectivity.py` contains explicit TODOs that may help choose the next refactor targets
- `solver.py` and `pyeeg/solvers.py` overlap conceptually, so that is a likely cleanup boundary
