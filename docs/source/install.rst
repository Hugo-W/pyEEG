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

.. note::

    To ensure that the code you run follows your edit, you may want to install the library in *developer*
    mode. By doing so, only symbolic links will be created on installation targeting your source code.
    Thus, any change to the code will be directly usable when importing natMEEG's functions.

Developer mode can be beneficial when working on several branches of the code, for instance, and being able to switch
from one instance to another depending on which branch you have *checked out*, or simply when editing the source code.
To install in developer mode:

.. code-block:: bash

    pip install -e  . 

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
