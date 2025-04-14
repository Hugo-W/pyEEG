from setuptools import setup, Extension
import numpy
import platform


# Determine the shared library extension based on the operating system
system = platform.system()
if system == "Windows":
    extra_compile_args = []
    extra_link_args = []
elif system == "Darwin":
    extra_compile_args = ['-fPIC']
    extra_link_args = []
else:
    extra_compile_args = ['-fPIC']
    extra_link_args = ['-shared']

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