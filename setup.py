from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy
import sys

# # Force mingw32 compiler on Windows
# # But only on Windows, to avoid creating a .def file for non-Windows platforms
# The fact of setting self.compiler to "mingw32" string i only possible
# in finalize_options, as in the build_ext constructor it should be a compiler object.
# import shutil
# class CustomBuildExt(build_ext):
#     def finalize_options(self):
#         if sys.platform == "win32" and not self.compiler:
#             if shutil.which("gcc"):
#                 self.compiler = "mingw32"
#                 print("[INFO] gcc detected: building with MinGW.")
#             else:
#                 print("[WARN] No gcc detected. Please install Build Tools for Visual Studio or MinGW-w64.")
#         super().finalize_options()


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