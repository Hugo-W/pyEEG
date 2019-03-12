Installation Instructions
=========================

Install pyEEG
-------------

From the folder containing ``setup.py``:

.. code-block:: bash
    
    python setup.py install

.. note::

    To ensure that the code you run follows your edit, you may want to install the library in *developer*
    mode. By doing so, only symbolic links will be created on installation, that will target your source code.
    Thus any change to the code wll be directly usable when importing pyEEG's functions.

This can be very useful when working on several branches of the code for instance and be able to switch
from one instance to another depending on which branch you have *checked out*. So to install in developer
mode:

.. code-block:: bash

    python setup.py develop


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