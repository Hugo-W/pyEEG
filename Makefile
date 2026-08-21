# Common development tasks for the natMEEG project.
#
# Override PYTHON, PIP, or PYTEST when using a virtual environment, for example:
#   make PYTHON=.venv/bin/python test

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

.PHONY: all install install-docs build test test-coverage clean doc docs docpdf

all: test

install:
	$(PIP) install -e .

install-docs:
	$(PIP) install -e ".[docs]"

build:
	# Keep the in-tree build directory from shadowing the `build` frontend.
	rm -rf build
	$(PYTHON) -m build

test:
	$(PYTEST) tests

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) tests --cov=pyeeg --cov-report=html:coverage

doc docs:
	$(MAKE) -C docs html

docpdf:
	$(MAKE) -C docs latexpdf

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage coverage
	$(MAKE) -C docs clean
