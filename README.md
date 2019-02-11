# pyEEG

pyEEG is a library fo processing EEG data build mostly on top of MNE-py and scikit-learn. It allows anlaysis of raw data and generation of temporal response functions with continuous signals as stimuli or real-valued events (e.g. word-level features).

If on the netwrok of Imperial College, you can access the documentation here: [pyeeg-docs](http://bg-hw2512.bg.ic.ac.uk).

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
pip install requirements.txt
```

To generate the doc, Python package `sphinx` (>= 1.1.0) and `sphinx_rtd_theme` are required (the former is installable from `conda` and the latter from `pip`).

### User Installation

From terminal (or `conda` shell in Windows), `cd` in root directory of the library (directory containing `setup.py` file) and type:

```bash
# To install an open version, allowing modification of source code
$ python setup.py develop
# To install as a fixed python package
$ python setup.py install
```

## Basic Examples

See files in `examples/`.

### Computing Envelope TRF and spatial map from CCA

TBC

### Computing Word-feature TRF

See [examples/TRF_wordonsets.ipynb](examples/TRF_wordonsets.ipynb)

### Working with Word vectors

See [examples/import_WordVectors.ipynb](examples/importWordVectors.ipynb)

## Documentation

The simplest way is to access it from Imperial College Network (or via VPN) [here](http://pyeeg-docs).
But you can also generate an _offline_ version, or a PDF file of all the docs by following the following instructions.

### Generate the documentation

To generate the documentation you will need `sphinx` to be installed in your Python environment. If it is not installed, install it with:

```bash
conda install sphinx
pip install sphinx_rtd_theme
```

or

```bash
pip install sphinx sphinx_rtd_theme
```

You can access the doc as HTML or PDF format.
To generate the documentation HTML pages, type in a terminal (does not work Windows!):

```bash
make doc
```

And for PDF version:

```bash
make docpdf
```

Then you can open the `docs/build/html/index.html` page in your favourite browser or open `docs/build/latex/pyEEG.pdf` in a PDF viewer.

**The PDF documentation can only be generated if `latex` and `latxmk` are present on the machine**

To clean files created during build process:

```bash
make clean
```