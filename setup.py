from setuptools import setup, Extension
import numpy
import sys


# Determine the shared library extension based on the operating system
if sys.platform == "win32":
    extra_compile_args = []
    extra_link_args = []
elif sys.platform == "darwin":
    extra_compile_args = ['-fPIC']
    extra_link_args = []
else:
    extra_compile_args = ['-fPIC']
    extra_link_args = ['-shared']

if sys.platform != "win32":  # Exclude .def file for non-Windows platforms
    extra_link_args.append("-Wl,--exclude-libs,ALL")

# Define the extension modules
gammatone_module = Extension(
    'pyeeg.bin.gammatone_c',
    sources=['pyeeg/gammatone_c/gammatone_c.c'],
    include_dirs=[numpy.get_include()],  # Add numpy include directory
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

ratemap_module = Extension(
    'pyeeg.bin.makeRateMap_c',
    sources=['pyeeg/gammatone_c/makeRateMap_c.c'],
    include_dirs=[numpy.get_include()],  # Add numpy include directory
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)

setup(
    ext_modules=[gammatone_module, ratemap_module],
)