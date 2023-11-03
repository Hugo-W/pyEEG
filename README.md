# pyEEG

> v1.2

pyEEG is a library for processing EEG data build mostly on top of MNE-py and scikit-learn. It is framed to work with data collected with naturalsistic stimuli, therefore with continuous recordings rather than trial-based designs. It allows analysis of continuous m/eeg and generation of temporal response functions with continuous signals as stimuli or real-valued events (e.g. word-level or phoneme-level features).

You can find the [documentation here](https://hugo-w.github.io/pyEEG-docs/index.html).

> Note that this code repository is relatively old and **unmaintained**. Most useful code about computing TRF is contained in `pyeeg/models.py`, especially in the class `TRFEstimator` and the function `_svd_regress`: the latter implements TRF estimation with memory efficient and accelerated computation for handling multiple epochs or multiple subjects.

------

## TODOs

### Priority

- [ ] Use [doctest](https://docs.python.org/2/library/doctest.html) for systematic testing of some functions
- [x] fix imports (for now, cannot do `import pyeeg` to access all modules...)

### Enhancements

- [ ] Functional connectivity methods:
  - [x] Estimate connectivity (**in construction**)
  - [ ] Graph theory metrics (path length, clustering coeff.)
- [ ] Pipeline `pyRiemann` and `pyeeg` [this one](https://github.com/freole/pyeeg) into some workflows..

------

## Installation

### Dependencies

pyEEG requires:

- Python (>= 3.5)
- NumPy (>= 1.11.0)
- SciPy (>= 1.0.0)
- mne (>= 0.16)
- pandas (>= 0.23.0)
- scikit-learn (>= 0.20.0)
- matplotlib (>= 2.0)
- h5py (>= 2.8.0)

Install requirements:

```bash
pip install -r requirements.txt
```

To generate the doc, Python package `sphinx` (>= 1.1.0), `sphinx_rtd_theme` and `nbsphinx` are required (`sphinx` can be installed from `conda` and the others from `pip`).

### User Installation

From terminal (or `conda` shell in Windows), `cd` in root directory of the library (directory containing `setup.py` file) and type:

To get the package installed only through symbolic links, namely so that you can modify the source code and use modified versions at will when importing the package in your python scripts do:

```bash
python setup.py develop
```

Otherwise, for a standard installation (but this will require to be installed if you need to install another version of the library):

```bash
python setup.py install
```

## Basic Examples

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
conda install sphinx
conda install -c conda-forge nbsphinx
pip install sphinx_rtd_theme
```

You can access the doc as HTML or PDF format.
To generate the documentation HTML pages, type in a terminal:

For Unix environment (from root directory):

```bash
make doc
```

For Windows environment (from `docs` folder):

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
