# natMEEG - Naturalistic M/EEG data analysis

[![PyPI version](https://badge.fury.io/py/natMEEG.svg)](https://badge.fury.io/py/natMEEG)
> v1.5.1 (2025-04-14)

_Formerly named `pyEEG`_

`natMEEG` is a library for processing M/EEG data built mostly on top of MNE-py and scikit-learn. It is framed to work with data collected with naturalistic stimuli, therefore with continuous recordings rather than trial-based designs. It allows analysis of continuous m/eeg and generation of temporal response functions with continuous signals as stimuli or real-valued events (e.g. word-level or phoneme-level features).

You can find the [documentation here](https://hugo-w.github.io/pyEEG-docs/index.html).

> ⚠️**Caution**:
> - Note that this code repository is **unmaintained** and intended for personal use. Most of the code about computing TRF is contained in `pyeeg/models.py`, especially in the class `TRFEstimator` and the function `_svd_regress`: the latter implements TRF estimation with memory efficient and accelerated computation for handling multiple epochs or multiple subjects.
> - It is recommended to use the code as a reference for your own implementation rather than relying on it for production use.
> - Finaly note that the repository went through a name change from `pyEEG` to `natMEEG`, so you might find references to `pyEEG` in the code and documentation.

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
pip install natMEEG
```

If you want to install docs building dependencies, you can do:

```bash
pip install natMEEG[docs]
```

If you want to install the package with all dependencies (including MNE), you can do:

```bash
pip install natMEEG[full]
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
trf.fit(X, y) # assuming data loaded: X is the stimulus signal, y is the M/EEG signal
print(trf.score(X, y)) # Normally you would use a separate test set for this, but here we use the same data for simplicity
trf.plot() # plot the TRF
```

### Examples

See files in [`examples/`](docs/source/examples/).

### Computing Envelope TRF and spatial map from CCA

See [examples/CCA_envelope.ipynb](docs/source/examples/CCA_envelope.ipynb)

### Computing Word-feature TRF

See [examples/TRF_wordonsets.ipynb](docs/source/examples/TRF_wordonsets.ipynb)

### Working with Word vectors

See [examples/import_WordVectors.ipynb](docs/source/examples/importWordVectors.ipynb)

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
Then open `docs/build/latex/pyEEG.pdf` in a PDF viewer.

> **Note:** The PDF documentation can only be generated if `latex` and `latxmk` are present on the machine

To clean files created during build process (can be necessary to re-build the documentation):

```bash
make clean
```
---

## License

This project is licensed under the terms of the GPL-3.0 license. See the [LICENSE](LICENSE) file for details.

