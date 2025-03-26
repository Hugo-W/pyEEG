from setuptools import setup, Extension
import subprocess
import platform
import os
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
    sources=['externals/gammatone_c/gammatone_c.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

ratemap_module = Extension(
    'pyeeg.bin.makeRateMap_c',
    sources=['externals/gammatone_c/makeRateMap_c.c'],
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

# Check if the required build tools are available
def check_build_tools():
    try:
        if system == "Windows":
            # and if distutils build option is not mingw32
            if os.environ.get("DISTUTILS_USE_SDK") == "1":
                # Check for Microsoft Visual C++ build tools
                subprocess.check_call(["cl"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Check for GCC
            subprocess.check_call(["gcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False

version = read_version()
build_tools_available = check_build_tools()

if not build_tools_available:
    print(f"Warning: {e}")
    print("Cannot find necessary program to build extension modules. Proceeding anyway...")
    print("Will copy the pre-built shared libraries instead.")
    if system == "Windows":
        ext = ".dll"
    elif system == "Darwin":
        ext = ".dylib"
    else:
        ext = ".so"
    try:
        import shutil
        os.makedirs("bin", exist_ok=True)
        shutil.copy(f"externals/gammatone_c/gammatone_c.{ext}", f"pyeeg/bin/gammatone_c.{ext}")
        shutil.copy(f"externals/gammatone_c/makeRateMap_c.{ext}", f"pyeeg/bin/makeRateMap_c.{ext}")
    except FileNotFoundError:
        print("Failed to copy shared libraries. Please copy the shared libraries manually.")
        print("They need to be in the bin directory.")
    setup(
        name='natmeeg',
        version=version,
        packages=['pyeeg'],
        package_dir={'pyeeg': 'pyeeg'},
        package_data={
            'pyeeg': ['*.dll', '*.so', '*.dylib', 'bin/*.dll', 'bin/*.so', 'bin/*.dylib'],
        },
        # data_files=[('bin', ['bin/gammatone_c.dll', 'bin/makeRateMap_c.dll'])] if system == "Windows" else [('bin', ['bin/gammatone_c.so', 'bin/makeRateMap_c.so'])],
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        url='https://github.com/Hugo-W/pyEEG',
        license='GNU GENERAL PUBLIC LICENSE',
        author='Hugo Weissbart',
        description='process EEG data with set of utilities function used in ou lab'
    )
else:
    setup(
        name='natmeeg',
        version=version,
        ext_modules=[gammatone_module, ratemap_module],
        packages=['pyeeg'],
        package_dir={'pyeeg': 'pyeeg'},
        package_data={
            'pyeeg': ['*.dll', '*.so', '*.dylib', 'bin/*.dll', 'bin/*.so', 'bin/*.dylib'],
        },
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        url='https://github.com/Hugo-W/pyEEG',
        license='GNU GENERAL PUBLIC LICENSE',
        author='Hugo Weissbart',
        description='process EEG data with set of utilities function used in ou lab'
    )
