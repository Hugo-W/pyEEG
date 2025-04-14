from setuptools import setup, Extension
import numpy

# Define the C extensions
extensions = [
    Extension(
        "pyeeg.bin.gammatone_c",
        sources=["pyeeg/gammatone_c/gammatone_c.c"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "pyeeg.bin.makeRateMap_c",
        sources=["pyeeg/gammatone_c/makeRateMap_c.c"],
        include_dirs=[numpy.get_include()],
    ),
]

# Potentially add the fallback mechanism here to copy the shared library
# if the compiled extension fails to load...

# Build the extensions
def build(setup_kwargs):
    """
    Custom build function to add C extensions.
    """
    setup_kwargs.update({"ext_modules": extensions})