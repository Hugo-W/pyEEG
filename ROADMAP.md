# Roadmap

This document is the current source of truth for work on the `ai-revamp`
branch. It replaces the former `TODO.md`, `NEXT_STEPS.md`,
`TODO-feature-stimulus-extraction.md`, and `AI_REVAMP_GUIDE.md` files.

## Project principles

- Preserve existing public APIs and import paths where practical.
- Keep compatibility shims when internal names or module locations change.
- Add regression coverage before removing or reshaping behavior.
- Run focused tests after each meaningful change.
- Keep documentation and examples aligned with the `pyeeg` import namespace
  and the `natMEEG` project name.

## Completed on `ai-revamp`

- Cleaned the main Sphinx naming and project links for `natMEEG`.
- Stabilized the public API and added `pyeeg.__all__`.
- Consolidated solver code in `pyeeg/solvers.py` and removed the root-level
  `solver.py`.
- Added and expanded lag-matrix, solver, TRF, CCA, and simulation regression
  tests.
- Added the `pyeeg.features` package with alignment, feature extraction,
  reduction, and pipeline components.
- Fixed quadratic regularization and the TRF statistics paths, including
  intercept handling, rank-deficient designs, p-value tail computation, and
  `TRFEstimator.__repr__` when time bounds are unspecified.

## Current verification

Focused regression and solver tests pass:

```text
22 passed
```

The full test suite currently has collection blockers:

- `pyeeg/features/llm_features.py` requires optional Torch when collected
  directly.
- `pyeeg/models/.__init__.py` is an invalidly named module and is collected
  by pytest. The models subpackage still needs a proper `__init__.py`.
- `tests/test_connectivity.py::test_plm` is still a placeholder.
- `tests/test_gammatone.py` is script/doctest-style and needs assertions; its
  C-extension behavior remains insufficiently covered.

## Next priorities

### 1. Repair package and test collection

- Resolve the malformed models package initialization and add focused tests for
  the supported import paths.
- Make optional feature dependencies safe for normal test collection, or
  explicitly exclude optional modules from collection.
- Replace the connectivity placeholder with deterministic metric tests.
- Convert the gammatone checks into real assertions and document required
  native-library build conditions.

### 2. Finish feature-extraction integration

Issue #15 is implemented in broad form, but the integration still needs:

- user-facing documentation and examples;
- tests for alignment, syntactic features, reduction, and pipeline behavior;
- a clear optional-dependency policy for LLM features;
- verification of backward-compatible TRF usage alongside feature-dictionary
  inputs.

### 3. Modularize large modules

Issue #13 remains open. Candidate boundaries are:

- shared regression and validation helpers from `pyeeg/models.py`;
- lag and design-matrix utilities from `pyeeg/utils.py`;
- data conversion and aligned-feature handling from `pyeeg/io.py`;
- connectivity algorithms and their shared numerical helpers.

Keep the existing public module paths while moving implementation details.

### 4. Add planned modeling features

- Issue #18: add banded ridge regularization, including feature-block ordering,
  per-feature alpha values, solver support, coefficient reshaping, and tests.
- Issue #17: add weighted samples for TRF estimation.
- Issue #14: design statistically appropriate permutation, bootstrap, and
  cross-subject inference for continuous naturalistic data.
- Issue #12: decide whether a solver-pattern abstraction reduces complexity
  without obscuring the current solver API.

## Documentation maintenance

- Keep the README's release information tied to released artifacts; identify
  development builds as development builds.
- Run Sphinx link checks and notebook/example checks when documentation paths
  or public APIs change.
- Keep `pyproject.toml`, README installation extras, and optional dependency
  warnings in agreement.
- Remove or update stale notebooks and examples that depend on local data
  paths or legacy APIs.

## Working rule

Before each change, state:

1. What public behavior currently exists?
2. What compatibility risk does the change introduce?
3. What focused test or documentation check proves the result?
