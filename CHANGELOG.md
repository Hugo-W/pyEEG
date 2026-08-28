# Changelog

All notable changes to **natMEEG** (formerly `pyEEG`) are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.2.0] — 2025-08-28

### Added
- `pyeeg.stats` module: nonparametric statistical inference for TRF analysis
  (Issue #14). Implements:
  - `permutation_test_trf`: circular-shift surrogate test with FWE via
    max-statistic (plus-one p-values). Stats: `zscore` (default, internal
    pre-lag z-scoring + refit, clean for any solver), `t` (OLS only),
    `coef`, `perm_norm` (permutation-null normalized).
  - `cluster_based_permutation_test`: Maris & Oostenveld (2007) cluster
    correction. Positive/negative clusters formed separately; adjacency
    (lag 1-D, explicit dense/sparse, none); threshold semantics by stat type.
  - `bootstrap_ci_trf`: paired circular block bootstrap with boundary drop,
    auto block-size estimation, percentile CIs, SE as byproduct.
  - `jackknife_se_trf`: leave-one-epoch-out SE/CI estimator (any solver:
    OLS, ridge, banded ridge, robust).
  - `cross_subject_consistency`: descriptive pairwise/LOO reliability
    (Pearson or cosine, no inferential test).
  - `group_level_test`: sign-flip group inference (H0: population mean = 0).
  - Spectral edge fade (`fade_edges=True`) for autocorrelated stimuli,
    estimated from -3dB bandwidth of the stimulus spectrum.
  - No MNE dependency in stats module (spatial adjacency user-supplied).
  - 451 tests passing (107 new stats tests + 4 copy regression tests).
- `TRFEstimator.copy()` now preserves all constructor kwargs (was dropping
  solver, loss, robust settings, intercept, cache config).

## [Unreleased]

### Added
- `Whitener`, `WaveletTransform`, `MultichanWienerFilter`, `mCCA`, `connectivity`,
  `simulate` (neural-mass models), `features` package, and `vizu` documented in the
  README features overview.
- Sphinx API pages for `connectivity`, `mcca`, `solvers`, and `features` modules;
  `simulate.rst` fixed (was broken) and populated with all neural-mass classes and
  simulation functions.
- `install.rst` now documents the `[features]` extra (`torch`, `transformers`).
- `usage.rst` examples for CCA, mCCA, connectivity, simulation, and whitening.
- `intersphinx` mapping to Python, NumPy, SciPy, Matplotlib, and pandas.

### Changed
- Comprehensive docstring pass across `features/*`, `simulate.py`, `solvers.py`,
  `preprocess.py` (`Whitener`), `connectivity.py`, `cca.py`, `models/trf.py`,
  `io.py`, `mcca.py`, `vizu.py`, `ratemap.py`, `utils.py`, and `models/var.py`.
- `pyeeg.features.llm_features` `ImportError` message typo fixed (`Instal` →
  `Install`); the `[features]` extra it references is now defined in
  `pyproject.toml`.

### Fixed
- Removed a duplicate `CTRNN.read_out` method definition in `simulate.py`.
- `models/var.py` `fit_var` return-shape docstring corrected to
  `(nchans, nlags, nchans)`.
- `connectivity.py`: broken reST reference patterns in `phase_transfer_entropy`,
  `wPLI`, and `plm` replaced with plain prose (fixes Sphinx build errors).
- `conf.py`: `release = version + 'a'` bug (unconditional `'a'` suffix) →
  `release = version`.

---

## [2.1.3] - 2026-08-26

### Added
- Per-channel IRLS with optional parallelism (`n_jobs`) for robust TRF fitting.

## [2.1.2] - 2026-08-26

### Changed
- Auto-use XtX SVD + block conjugate gradient for multichannel solves
  (significant performance improvement).

## [2.1.1] - 2026-08-25

### Added
- Batched IRLS multichannel solve for faster robust fitting.

### Fixed
- Conjugate-gradient multi-epoch scaling bug.

## [2.1.0] - 2026-08-25

### Added
- Truncated SVD option for `SVDSolver` (`alpha` interpreted as retained
  variance fraction).

## [2.0.2] - 2026-08-25

### Changed
- Sigstore signing action upgraded to v3.1.0.

## [2.0.1] - 2026-08-25

### Changed
- Unified logging system (`pyeeg._logging` with `set_log_level` /
  `get_logger`), replacing scattered `print` and `warnings` calls.

## [2.0.0] - 2026-08-25

### Added
- **Solver Pattern abstraction** (issue #12): abstract `Solver` base class
  with five concrete subclasses — `SVDSolver` (SVD-based ridge), `LSTSQSolver`
  (ordinary least squares), `ConjugateGradientSolver` (CG on normal equations,
  10–50× faster than SVD with identical results), `IRLSSolver` (robust
  Cauchy-loss IRLS), and `ScipyRobustSolver` (SciPy nonlinear Cauchy reference).
- Dependency injection: `TRFEstimator` accepts any `Solver` instance via
  `solver=`.
- **Interactive TRF dashboard** (`pyeeg.dashboard`) with a `trf-explore`
  console entry point for upload-and-fit exploration.
- `scripts/examples/solver_showcase.py` comparing all TRF solvers.

### Changed
- Full backward compatibility preserved; existing `TRFEstimator` and
  free-function (`_svd_regress`, etc.) APIs work unchanged.

### Fixed
- Solver compatibility guards: warns when robust loss is used with a non-robust
  solver; raises when a non-SVD solver is used with multi-alpha arrays.

---

## [1.7.1] - 2026-08-24

### Fixed
- CI: bumped `actions/setup-python` v3 → v5.
- CI: dropped `-W` (treat-warnings-as-errors) from docs build.
- CI: fixed docs pandoc missing + sdist corruption from duplicate builds.

## [1.7.0] - 2026-08-24

### Added
- **Stimulus feature extraction** (`pyeeg.features` package): LLM-based
  features (surprisal, entropy, KL divergence), syntactic features (tree
  depth, opening, closing), TextGrid alignment, feature pipeline, and
  dimensionality reduction (PCA, ICA).
- **Robust Cauchy-loss TRF fitting** (issue #17) via IRLS and a SciPy
  nonlinear least-squares reference path.
- **Weighted samples** (WLS) in `TRFEstimator` (issue #17).
- **Feature-specific ridge regularization** (banded ridge, `feature_alphas`).
- **Quadratic regularization** for TRFs (issue #16) with intercept and
  block-ordering support.
- `pyeeg.models` subpackage: split monolithic `pyeeg/models.py` into
  `pyeeg.models` (`trf.py`, `var.py`) while preserving public import paths
  (issue #11).
- Simulation-based regression tests for TRF and CCA (issue #10).
- Full test suite for `pyeeg/utils.py` (issue #9).

### Changed
- Stabilized and documented public API entry points and `pyeeg.__all__`
  (issues #4, #5, #6, #24).
- Consolidated duplicate `_svd_regress` implementations (issue #7) and
  merged root `solver.py` into `pyeeg/solvers.py` (issue #8).
- Unified project documentation; removed outdated TODO/NEXT_STEPS files,
  replaced with `ROADMAP.md`.

### Fixed
- P-value underflow and `__repr__` crash in `TRFEstimator` statistics
  (issue #25).
- Gammatone variable mismatch with C extension (issue #26).
- Deprecated `lag_matrix` call arguments (issue #27).

---

## [1.6.10] - 2025-04-15

### Fixed
- Re-included `setup.py` in MANIFEST for wheel build.

## [1.6.9] - 2025-04-15

### Fixed
- Testing name change to PEP 8; documentation updated.

## [1.6.8] - 2025-04-15

### Fixed
- Re-included `setup.py` in MANIFEST; updated package inclusion pattern for
  submodules in setuptools.

## [1.6.7] - 2025-04-15

### Changed
- Code structure refactor for readability; sdist bloat cleanup.

## [1.6.6] - 2025-04-14

### Changed
- Python distribution tested for `>= 3.10`.

## [1.6.5] - 2025-04-14

### Fixed
- Wheel build configuration.

## [1.6.4] - 2025-04-14

### Fixed
- `publish.yml` now handles `twine check` failures gracefully.

## [1.6.3] - 2025-04-14

### Fixed
- Wheel build configuration.

## [1.6.2] - 2025-04-14

### Fixed
- Wheel repair process in `publish.yml`; README updated for project name
  change.

## [1.6.1] - 2025-04-14

### Fixed
- Wheel build configuration.

## [1.6.0] - 2025-04-14

### Changed
- `publish.yml` repaired for manylinux wheels; OS classifiers updated in
  `pyproject.toml`; README updated; `version.py` untracked.

## [1.5.1] - 2025-04-14

### Fixed
- Removed MANIFEST (build error).

## [1.5.0] - 2025-04-14

### Changed
- Added `MANIFEST.in` for package inclusion; updated project description
  and dependencies in `pyproject.toml`; disabled custom mingw32 build
  extension on Windows.

## [1.4.3] - 2025-04-14

### Added
- `pyproject.toml` and `setup.py` build system with C extension definitions
  (gammatone, makeRateMap) and numpy include-directory handling.
- License updated to GPL-3.0-or-later in `pyproject.toml`.

### Changed
- Fully modernized installation and build process (`python -m build`);
  refined package discovery and platform-specific compilation.

## [1.4.1] - 2025-04-14

### Changed
- `publish.yml` artifact download steps updated to use `pattern` matching
  with `merge-multiple`.

## [1.4.0] - 2025-04-14

First modern release after the rename from `pyEEG` to `natmeeg`, consolidating
years of development into a single tagged milestone.

### Added
- **Temporal Response Functions**: `TRFEstimator` with SVD-based ridge
  regression, multi-alpha scoring, list-based covariance accumulation,
  load/save, `xfit` cross-validation, interactive and significance plotting,
  topomaps, and `__add__`/`__truediv__`/`__getitem__` operators.
- **Canonical Correlation Analysis**: `CCA_Estimator` (lagged/regularized)
  plus `cca_nt` and `cca_svd` backends; multiway CCA (`mCCA`) for
  hyperalignment preprocessing.
- **Connectivity module**: Granger causality, phase transfer entropy (PTE),
  phase linearity measurement (PLM), and cross-spectral density.
- **Simulation module**: AR/VAR generation, Jansen-Rit (and extended) neural
  mass model, CTRNN, and TRF simulation kernels.
- **Preprocessing**: `Whitener` (PCA/ZCA), `WaveletTransform`,
  `MultichanWienerFilter` (MWF), filterbanks, and covariance estimators.
- **IO**: EEGLAB/FieldTrip → MNE conversion, `AlignedSpeech` and
  `WordLevelFeatures` classes, word-onset/envelope/surprisal/syntactic
  feature loaders.
- **Visualization** (`pyeeg.vizu`): topomaps, filterbank plots, TRF
  significance overlays, pairwise boxplots, interactive plots.
- Gammatone filterbank and cochleagram C extensions.
- Sigstore-signed PyPI publishing via GitHub Actions.

### Changed
- MNE made an optional dependency; `psutil` and `tqdm` made required.

---

## [0.4-complete_version] - 2019-02-22

### Added
- `AlignedSpeech` class for aligned audio features; fast envelope extraction;
  rolling-window and moving-average utilities.

### Fixed
- Memory issues and loading of MATLAB v7.3 files; `TRFEstimator.predict` beta
  reshape; lag order for TRF.

## [0.3-rc1] - 2019-02-11

### Added
- Canonical Correlation Analysis (CCA) implementation with separate
  `thresh_x`/`thresh_y`, knee-point selection, and `plot_activation_map`.

## [0.3] - 2019-02-11

Initial release of `pyEEG`.

### Added
- `TRFEstimator` with SVD regression and lag-matrix utilities.
- `WordLevelFeatures` and `AlignedSpeech` IO classes for word-level feature
  handling.
- Sphinx documentation with example notebooks.
- mCCA, knee-point detection, SPD matrix checks, and memory checking.
- EEGLAB → MNE conversion and word-onset loading utilities.
- Preprocessing (filterbank, covariance) and visualization functions.

---

<!-- Link references -->
[Unreleased]: https://github.com/Hugo-W/pyEEG/compare/v2.1.3...HEAD
[2.1.3]: https://github.com/Hugo-W/pyEEG/releases/tag/2.1.3
[2.1.2]: https://github.com/Hugo-W/pyEEG/releases/tag/2.1.2
[2.1.1]: https://github.com/Hugo-W/pyEEG/releases/tag/2.1.1
[2.1.0]: https://github.com/Hugo-W/pyEEG/releases/tag/2.1.0
[2.0.2]: https://github.com/Hugo-W/pyEEG/releases/tag/2.0.2
[2.0.1]: https://github.com/Hugo-W/pyEEG/releases/tag/2.0.1
[2.0.0]: https://github.com/Hugo-W/pyEEG/releases/tag/2.0.0
[1.7.1]: https://github.com/Hugo-W/pyEEG/releases/tag/1.7.1
[1.7.0]: https://github.com/Hugo-W/pyEEG/releases/tag/1.7.0
[1.6.10]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.10
[1.6.9]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.9
[1.6.8]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.8
[1.6.7]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.7
[1.6.6]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.6
[1.6.5]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.5
[1.6.4]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.4
[1.6.3]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.3
[1.6.2]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.2
[1.6.1]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.1
[1.6.0]: https://github.com/Hugo-W/pyEEG/releases/tag/1.6.0
[1.5.1]: https://github.com/Hugo-W/pyEEG/releases/tag/1.5.1
[1.5.0]: https://github.com/Hugo-W/pyEEG/releases/tag/1.5.0
[1.4.3]: https://github.com/Hugo-W/pyEEG/releases/tag/1.4.3
[1.4.1]: https://github.com/Hugo-W/pyEEG/releases/tag/1.4.1
[1.4.0]: https://github.com/Hugo-W/pyEEG/releases/tag/1.4.0
[0.4-complete_version]: https://github.com/Hugo-W/pyEEG/releases/tag/0.4-complete_version
[0.3-rc1]: https://github.com/Hugo-W/pyEEG/releases/tag/0.3-rc1
[0.3]: https://github.com/Hugo-W/pyEEG/releases/tag/0.3
