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
- Added `pyeeg.stats` module (Issue #14, released in 2.2.0): permutation
  testing, cluster-based correction, bootstrap CIs, jackknife SE,
  cross-subject consistency, and group-level sign-flip test.

## Current verification (as of 2.2.0, 2026-08-28)

The fast test suite passes:

```text
432 passed, 28 deselected (slow/llm)
```

460 tests collected total. Known gaps:

- `tests/test_connectivity.py::test_plm` is still a placeholder (`pass`); the
  connectivity metrics (Granger, PTE, wPLI, PLM) lack deterministic coverage.
- `tests/test_gammatone.py` is doctest-style and needs assertions; its
  C-extension behavior remains insufficiently covered.
- `pyeeg/features/llm_features.py` requires optional Torch; excluded from
  default collection via pytest markers (`-m "not llm"`).

## Next priorities

### 1. Repair test coverage gaps

- Replace the connectivity placeholder with deterministic metric tests
  (Granger, PTE, wPLI, PLM against analytic or FieldTrip/MNE references).
- Convert the gammatone checks into real assertions and document required
  native-library build conditions.
- Add behavioral/regression tests for `pyeeg.simulate` neural-mass models
  (network coupling, `read_out`, `_simulate_node` shared engine).

### 2. Issue #33 — Acoustic embeddings via deep models (wav2vec2, HuBERT)

- Add `DeepAcousticFeatureExtractor` to `pyeeg/features/acoustic.py`.
- Support wav2vec2 and HuBERT via HuggingFace transformers.
- Integrate into `FeaturePipeline` and add tests.
- Priority: medium.

### 3. Maintain the TRF Explorer

The dashboard's feature-level roadmap is maintained in
[`pyeeg/dashboard/TODO.md`](pyeeg/dashboard/TODO.md). Near-term work includes
endpoint/browser tests, progress handling for long fits, result export, and
feature/channel selection.

### 4. Issue #34 — Array API pilot (low priority)

Backend dispatch (`array-api-compat`) for clean numerical kernels:
`pyeeg/models/var.py`, materialized lag-matrix prototype, and dense TRF
solver paths. NumPy-only boundaries: SciPy sparse/optimize, sklearn, RNG,
I/O, viz, dashboard, C extensions. See issue for full scope and acceptance
criteria.

### 5. Modularize large modules (housekeeping)

Candidate boundaries for future refactoring (no open issue):

- shared regression and validation helpers from `pyeeg/models/`;
- lag and design-matrix utilities from `pyeeg/utils.py`;
- data conversion and aligned-feature handling from `pyeeg/io.py`;
- connectivity algorithms and their shared numerical helpers.

Keep the existing public module paths while moving implementation details.

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
