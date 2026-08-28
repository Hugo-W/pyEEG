# Roadmap

This document tracks ongoing and planned work on `natMEEG` (the `pyeeg`
import namespace). It replaces the former `TODO.md`, `NEXT_STEPS.md`,
`TODO-feature-stimulus-extraction.md`, and `AI_REVAMP_GUIDE.md` files. The
`ai-revamp` work has been merged to `main`; this document is kept current
against `main`.

## Project principles

- Preserve existing public APIs and import paths where practical.
- Keep compatibility shims when internal names or module locations change.
- Add regression coverage before removing or reshaping behavior.
- Run focused tests after each meaningful change.
- Keep documentation and examples aligned with the `pyeeg` import namespace
  and the `natMEEG` project name.

## Completed on `ai-revamp` (merged to `main`)

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
- Added weighted and robust Cauchy-loss TRF estimation, including IRLS and a
  SciPy nonlinear least-squares reference path.
- Split the monolithic `pyeeg/models.py` into a `pyeeg.models` subpackage
  (`pyeeg/models/trf.py`, `pyeeg/models/var.py`) while preserving all public
  import paths.
- Added the exploratory `pyeeg.dashboard` TRF Explorer with a `uv` console
  entry point, NumPy upload validation, real TRF fitting, regularisation and
  solver controls, responsive UI, and channel-wise result overlays.
- Added neural-mass simulation models in `pyeeg.simulate`: `NeuralMassNode`/
  `NeuralMassNetwork` base classes, `HopfOscillator`, `Phasor`,
  `WilsonCowan`, `Kuramoto`, `CTRNN`, plus the `JansenRit` /
  `JansenRitExtended` / `JRNetwork` family and AR/VAR couplings.
- Implemented banded ridge regularization (`feature_alphas` on
  `TRFEstimator`, feature-block ordering, per-feature alphas, solver support,
  and the `scripts/tutorials/feature_alphas_banded_ridge.ipynb` tutorial) —
  closes Issue #18.

## Current verification

Focused regression and solver tests pass. The current full suite passes:

```text
214 passed
```

The full test suite currently has collection blockers:

- `pyeeg/features/llm_features.py` requires optional Torch when collected
  directly.
- `tests/test_connectivity.py::test_plm` is still a placeholder (`pass`); the
  connectivity metrics (Granger, PTE, wPLI, PLM) lack deterministic coverage.
- `tests/test_gammatone.py` is script/doctest-style and needs assertions; its
  C-extension behavior remains insufficiently covered.

## Next priorities

### 1. Repair package and test collection

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

- shared regression and validation helpers from `pyeeg/models/`;
- lag and design-matrix utilities from `pyeeg/utils.py`;
- data conversion and aligned-feature handling from `pyeeg/io.py`;
- connectivity algorithms and their shared numerical helpers.

Keep the existing public module paths while moving implementation details.

### 4. Maintain the TRF Explorer

The dashboard's feature-level roadmap is maintained in
[`pyeeg/dashboard/TODO.md`](pyeeg/dashboard/TODO.md). Near-term work includes
endpoint/browser tests, progress handling for long fits, result export, and
feature/channel selection.

### 5. Add planned modeling features

- Issue #18: banded ridge regularization is implemented (`feature_alphas`,
  feature-block ordering, per-feature alphas, solver support, tests, and a
  tutorial); close the upstream issue after release verification.
- Issue #17: weighted and robust TRF estimation is implemented on this branch;
  close the upstream issue after review and release verification.
- Issue #14: implemented — `pyeeg.stats` module with permutation testing
  (circular-shift null, `stat="zscore"/"t"/"coef"/"perm_norm"`), cluster-based
  correction (Maris & Oostenveld 2007), bootstrap CIs, jackknife SE,
  cross-subject consistency, and group-level sign-flip test. Spectral edge fade
  for autocorrelated stimuli. Close the upstream issue after release
  verification.
- Issue #12: decide whether a solver-pattern abstraction reduces complexity
  without obscuring the current solver API.

### 6. Connectivity and simulation coverage

- `pyeeg.connectivity` (Granger, PTE, wPLI, PLM, CSD) is documented but
  undertested — replace the `test_plm` placeholder with deterministic metric
  tests (against analytic or FieldTrip/MNE references).
- `pyeeg.simulate` neural-mass models (Hopf, WilsonCowan, Kuramoto, CTRNN,
  JansenRit family) are now documented; add behavioral/regression tests for
  network coupling, `read_out`, and the `_simulate_node` shared engine.

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
