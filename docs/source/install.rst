Installation Instructions
=========================

Install pyEEG
-------------

From the folder containing ``setup.py``:

.. code-block:: bash
    
    python setup.py install
    # Or (pip is preferable):
    pip install . 

.. note::

    To ensure that the code you run follows your edit, you may want to install the library in *developer*
    mode. By doing so, only symbolic links will be created on installation targeting your source code.
    Thus, any change to the code will be directly usable when importing pyEEG's functions.

Developer mode can be beneficial when working on several branches of the code, for instance, and being able to switch
from one instance to another depending on which branch you have *checked out*, or simply when editing the source code.
To install in developer mode:

.. code-block:: bash

    python setup.py develop
    # Or (pip is preferable):
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
