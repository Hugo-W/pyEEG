
# simple makefile to simplify repetitive build env management tasks under posix

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= pytest
CTAGS ?= ctags

# skip doctests on 32bit python
BITS := $(shell python -c 'import struct; print(8 * struct.calcsize("P"))')

all: clean inplace test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	$(MAKE) -C docs clean

in: inplace # just a shortcut
inplace:
	#$(PYTHON) setup.py develop

test-code: in
	$(PYTEST) --showlocals -v pyeeg --durations=20
test-sphinxext:
	$(PYTEST) --showlocals -v doc/sphinxext/
test-doc:
ifeq ($(BITS),64)
	$(PYTEST) $(shell find doc -name '*.rst' | sort)
endif

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) pyeeg --showlocals -v --cov=pyeeg --cov-report=html:coverage

test: test-code test-sphinxext test-doc

trailing-spaces:
	find pyeeg -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

convert-notebooks:
	find examples -maxdepth 1 -name "*.ipynb" -exec jupyter-nbconvert {} --to rst \;

doc: inplace
	$(MAKE) -C docs html

docpdf: inplace doc
	$(MAKE) -C docs latexpdf

code-analysis:
	flake8 pyeeg | grep -v __init__ | grep -v external
	pylint -E -i y pyeeg/ -d E1103,E0611,E1101

flake8-diff:
	./build_tools/travis/flake8_diff.sh