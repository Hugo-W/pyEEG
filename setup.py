from setuptools import setup, Extension
import platform
import numpy
import re

# Determine the shared library extension based on the operating system
system = platform.system()
if system == "Windows":
    extra_compile_args = []
    extra_link_args = []
elif system == "Darwin":
    # On macOS, don't force '-shared'
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

def read_version():
    with open("pyeeg/version.py", "r") as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='natmeeg',
    version=read_version(),
    ext_modules=[gammatone_module, ratemap_module],
    packages=['pyeeg', 'pyeeg.bin'],
    package_dir={'pyeeg': 'pyeeg'},
    package_data={
        'pyeeg': ['*.dll', '*.so', '*.dylib', 'bin/*.dll', 'bin/*.so', 'bin/*.dylib'],
    },
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    url='https://github.com/Hugo-W/pyEEG',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Hugo Weissbart',
    description='Process EEG data with a set of utility functions used in our lab'
)