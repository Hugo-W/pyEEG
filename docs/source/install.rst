Installation Instructions
=========================

Install natMEEG
---------------

From PyPI:
~~~~~~~~~~

.. code-block:: bash

    pip install natmeeg

From source
~~~~~~~~~~~~

Download source files from the GitHub repository (tarball archive release or source distribution in PyPI) and extract them.

Then, from the folder containing ``pyproject.toml``:

.. code-block:: bash
    
    pip install . 

With `uv <https://docs.astral.sh/uv/>`_ (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project is managed with `uv <https://docs.astral.sh/uv/>`_. If you have
``uv`` installed, you can install and run the package with:

.. code-block:: bash

    uv sync

which creates a virtual environment and installs the package and its core
dependencies, or

.. code-block:: bash

    uv run python path/to/script.py

to run a script directly without manually activating the environment.

Development installation
~~~~~~~~~~~~~~~~~~~~~~~~

To install the package in editable/developer mode with ``uv``:

.. code-block:: bash

    uv sync

For working on the documentation, install the docs extra as well:

.. code-block:: bash

    uv sync --extra docs

.. note::

    To ensure that the code you run follows your edit, you may want to install the library in *developer*
    mode. By doing so, only symbolic links will be created on installation targeting your source code.
    Thus, any change to the code will be directly usable when importing natMEEG's functions.

Developer mode can be beneficial when working on several branches of the code, for instance, and being able to switch
from one instance to another depending on which branch you have *checked out*, or simply when editing the source code.
To install in developer mode:

.. code-block:: bash

    pip install -e  . 

Optional dependencies
---------------------

The following optional extras are defined in ``pyproject.toml``:

- ``[mne]`` — installs `MNE-Python <https://mne.tools/>`_ (``mne``), the
  standard toolbox for M/EEG data handling and analysis.
- ``[docs]`` — installs the dependencies needed to build the documentation:
  ``sphinx``, ``sphinx-rtd-theme``, ``nbsphinx`` and ``ipykernel``.
- ``[full]`` — installs everything needed for both MNE-based workflows and
  documentation builds: ``mne``, ``sphinx``, ``sphinx-rtd-theme`` and
  ``nbsphinx``.
- ``[exploratory-trf]`` — installs the web-server dependencies of the
  exploratory TRF dashboard: ``flask`` (>= 2.0.0), ``werkzeug`` (>= 2.0.0) and
  ``gunicorn`` (>= 20.0.0).

Install an extra with, for example:

.. code-block:: bash

    pip install natmeeg[full]

Generate documentation
----------------------

- On Windows:

.. code-block:: bat
    
    .\make.bat html


- On Linux/Mac:

.. code-block:: bash
    
    make doc

If you have modified the documentation source files, you might need to clean the build directories
before running `make doc` again:

.. code-block:: bash
    
    make clean && make doc
