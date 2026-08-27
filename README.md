# natMEEG - Naturalistic M/EEG data analysis

[![PyPI version](https://badge.fury.io/py/natmeeg.svg)](https://badge.fury.io/py/natmeeg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22081524.svg)](https://doi.org/10.5281/zenodo.22081524)


_Formerly named `pyEEG`_

`natMEEG` is a library for processing M/EEG data built mostly on top of MNE-Python and scikit-learn. It is designed for data collected with naturalistic stimuli, so it works with continuous recordings rather than trial-based designs. It supports analysis of continuous M/EEG and generation of temporal response functions from continuous signals or real-valued events (for example, word-level or phoneme-level features).

You can find the [documentation here](https://hugo-w.github.io/pyEEG/).

> ⚠️**Note**:
>
> - The library provides tools for computing TRFs (Temporal Response Functions) with the `TRFEstimator` class in `pyeeg/models/`, which implements memory-efficient and accelerated computation for handling multiple epochs or multiple subjects.
> - The project was formerly known as `pyEEG` and has been renamed `natMEEG` to better reflect its focus on **nat**uralistic **M/EEG** data analysis.

------

## Features

natMEEG provides tools across the full naturalistic M/EEG analysis pipeline:

- **Temporal Response Functions (TRF)** — `TRFEstimator` (`pyeeg.models`) for memory-efficient, accelerated TRF estimation from continuous signals and real-valued event features, with ridge and robust Cauchy fitting, banded ridge (`feature_alphas`), sample weighting, and pluggable solvers.
- **Canonical Correlation Analysis** — `CCA_Estimator` (`pyeeg.cca`) with lagged/regularized CCA and visualization; `mCCA` (`pyeeg.mcca`) for multiway CCA / hyperalignment preprocessing.
- **Connectivity** — `pyeeg.connectivity` with Granger causality, phase transfer entropy (PTE), weighted phase lag index (wPLI), phase linearity measurement (PLM), and cross-spectral density.
- **Simulation** — `pyeeg.simulate` with AR/VAR generation and neural-mass models (Hopf oscillator, Wilson–Cowan, Kuramoto, CTRNN, Jansen–Rit and its network extension) for generating synthetic coupled dynamics and TRF test data.
- **Feature extraction** — `pyeeg.features` for aligning stimulus annotations (TextGrid), extracting LLM-derived features (surprisal, entropy, KL divergence; requires `torch` via the `[features]` extra), syntactic features (tree depth, opening, closing), dimensionality reduction, and end-to-end encoding pipelines.
- **Preprocessing** — `pyeeg.preprocess` with `Whitener` (PCA/ZCA), `WaveletTransform`, `MultichanWienerFilter`, filterbanks, and covariance estimators.
- **VAR modeling** — `fit_ar` / `fit_var` (`pyeeg.models`) for autoregressive and vector autoregressive coefficient estimation.
- **Visualization** — `pyeeg.vizu` with topomaps, filterbank plots, TRF significance overlays, and pairwise boxplots.
- **IO** — `pyeeg.io` for EEGLAB/FieldTrip → MNE conversion and aligned word-level feature handling.

See the [documentation](https://hugo-w.github.io/pyEEG/) for full API reference.
See the [changelog](CHANGELOG.md) for release history.

------

## Installation

### Dependencies

natMEEG requires:

- Python (>= 3.10)
- psutil
- tqdm
- NumPy
- SciPy
- scikit-learn
- matplotlib
- h5py
- pandas
- mne (>= 0.16) [optional]

To generate the doc, Python package `sphinx` (>= 1.1.0), `sphinx_rtd_theme` and `nbsphinx` are required.

### User Installation

## From PyPI

You can install the package from PyPI using `pip`:

```bash
pip install natmeeg
```

If you want to install docs building dependencies, you can do:

```bash
pip install natmeeg[docs]
```

If you want to install the package with all dependencies (including MNE), you can do:

```bash
pip install natmeeg[full]
```

### From Source

If you prefer to install the package from source, you can clone the repository or download release archive or also use the source distribution (`.tar.gz` file from PyPi) and build it locally. There is a C-extension that needs to compile, so you need to have a C compiler installed on your machine.

From terminal, `cd` in root directory of the library after cloning this repository (directory containing `pyproject.toml` file).

To get the package installed only through symbolic links, namely so that you can modify the source code and use modified versions at will when importing the package in your python scripts do:

```bash
pip install -e .
```

Otherwise, for a standard installation, you can run:

```bash
pip install .
```

#### Windows Users

There are C-extensions in the library, so you need to have a C compiler installed on your machine.
If the default compiler does not work, you can try to install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and try again.

Optionally try with [MinGW](http://www.mingw.org/), making sure after instalation of it to add the path to `mingw/bin` in your `PATH` environment variable. You can check if it is correctly installed by running the following command in your terminal:

```bash
gcc --version
```

If this build tool is available it should be detected during build process (running `pip install .`, `pip install -e .` or `python -m build`).

## Usage

The most common usage of the library is to compute temporal response functions (TRF) from continuous M/EEG data. The library provides a `TRFEstimator` class that allows you to fit a TRF model to your data. The TRF model can be used to predict the M/EEG signal from a stimulus signal (e.g. a continuous audio signal or a sequence of word features):

```python
from pyeeg import TRFEstimator

trf = TRFEstimator(tmin=-0.2, tmax=0.5, srate=fs, alpha=100.0) # TRF between -200ms and 500ms, regularization parameter alpha=100.0
trf.fit(X, y) # assuming data loaded: X is the stimulus signal, y is the M/EEG signal, they must have the same number of samples (rows)
print(trf.score(X, y)) # Normally you would use a separate test set for scoring
trf.plot() # plot the TRF
```

### Choosing a Solver

The `TRFEstimator` auto-selects an appropriate solver by default (SVD-based ridge
for regularized fits, ordinary least squares otherwise). For advanced use cases,
any `Solver` subclass can be injected via the `solver=` parameter:

```python
from pyeeg.solvers import SVDSolver, ConjugateGradientSolver, IRLSSolver

# Fast iterative solver (same results as SVD, often 10-50x faster)
trf = TRFEstimator(tmin=-0.2, tmax=0.5, srate=fs, alpha=100.0,
                   solver=ConjugateGradientSolver())

# Robust fitting with Cauchy loss (downweights outliers)
trf = TRFEstimator(tmin=-0.2, tmax=0.5, srate=fs, alpha=100.0,
                   solver=IRLSSolver(max_iter=50))
```

See `scripts/examples/solver_showcase.py` for a full comparison of all solvers.

### Examples

See files in [`examples/`](docs/source/examples/).

### Computing Envelope TRF and spatial map from CCA

See [examples/CCA_envelope.ipynb](docs/source/examples/CCA_envelope.ipynb)

### Computing Word-feature TRF

See [examples/TRF_wordonsets.ipynb](docs/source/examples/TRF_wordonsets.ipynb)

### Computing TRF from syntactic features

See [examples/TRF_syntactic_feats.ipynb](docs/source/examples/TRF_syntactic_feats.ipynb)

### Simulating TRF data

See [examples/TRF_simulation_tutorial.ipynb](docs/source/examples/TRF_simulation_tutorial.ipynb)

### Working with Word vectors

See [examples/import_WordVectors.ipynb](docs/source/examples/import_WordVectors.ipynb)

## Documentation

You can generate an _offline_ HTML version, or a PDF file of all the docs by following the following instructions (HTML pages are easier to navigate in and prettier than the PDF thanks to the nice theme brought by `sphinx_rtd_theme`).

### Generate the documentation

To generate the documentation you will need `sphinx` to be installed in your Python environment, as well as the extension `nbsphinx` (for Jupyter Notebook integration) and the theme package `sphinx_rtd_theme`. Install those with:

```bash
pip install natMEEG[docs]
```

You can access the doc as HTML or PDF format. First get the source documentation files by cloning the repository or downloading the release archive. The documentation is located in the `docs` folder.
To generate the documentation HTML pages, type in a terminal:

For Unix environment (from root directory, as it uses the `Makefile`):

```bash
make doc
```

For Windows environment (from `docs` folder, where `make.bat` is located):

```bash
cd docs
make.bat html
```

Then you can open the `docs/build/html/index.html` page in your favourite browser.

And for PDF version, simply use `docpdf` instead of `doc` above.
Then open `docs/build/latex/natMEEG.pdf` in a PDF viewer.

> **Note:** The PDF documentation can only be generated if `latex` and `latxmk` are present on the machine

To clean files created during build process (can be necessary to re-build the documentation):

```bash
make clean
```

---

## License

This project is licensed under the terms of the GPL-3.0 license. See the [LICENSE](LICENSE) file for details.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22081524.svg)](https://doi.org/10.5281/zenodo.22081524)

> Weissbart, H. Natmeeg - M/EEG Data Analysis in Naturalistic Context. 1.7.1, Zenodo, 24 Aug. 2026, <https://doi.org/10.5281/zenodo.22081524>.
