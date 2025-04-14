from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy
import sys

# # Force mingw32 compiler on Windows
# # But only on Windows, to avoid creating a .def file for non-Windows platforms
# class CustomBuildExt(build_ext):
#     def build_extensions(self):
#         if sys.platform == "win32":
#             # Use mingw32 compiler
#             from distutils import sysconfig
#             from distutils.ccompiler import new_compiler
#             compiler = new_compiler(compiler='mingw32')
#             compiler.set_executable('gcc')
#         super().build_extensions()


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
    # cmdclass={"build_ext": CustomBuildExt},
    ext_modules=[gammatone_module, ratemap_module],
)