from setuptools import setup, Extension
import platform

# Determine the shared library extension based on the operating system
system = platform.system()
if system == "Windows":
    extra_compile_args = []
    extra_link_args = []
else:
    extra_compile_args = ['-fPIC']
    extra_link_args = ['-shared']

# Define the extension modules
gammatone_module = Extension(
    'pyeeg.gammatone_c',
    sources=['src/gammatone_c.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)

ratemap_module = Extension(
    'pyeeg.makeRateMap_c',
    sources=['src/makeRateMap_c.c'],
    extra_compile_args=extra_compile_args,
   extra_link_args=extra_link_args
)

VERS = {}
with open("./pyeeg/version.py") as fp:
    exec(fp.read(), VERS)

try:
    raise Exception("Forcing exception")
    setup(
        name='pyEEG',
        version=VERS['__version__'],
        ext_modules=[gammatone_module, ratemap_module],
        packages=['pyeeg'],
        package_dir={'pyeeg': 'pyeeg'},
        package_data={
            'pyeeg': ['*.dll', '*.so', '*.dylib'],
        },
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        url='https://github.com/Hugo-W/pyEEG',
        license='GNU GENERAL PUBLIC LICENSE',
        author='Hugo Weissbart',
        description='process EEG data with set of utilities function used in ou lab'
    )
except Exception as e:
    import shutil
    print(f"Warning: {e}")
    print("Failed to build extension modules. Proceeding anyway...")
    print("Will copy the pre-built shared libraries instead.")
    if system == "Windows":
        ext = ".dll"
    elif system == "Darwin":
        ext = ".dylib"
    else:
        ext = ".so"
    try:
        shutil.copy(f"src/gammatone_c.{ext}", f"pyeeg/gammatone_c.{ext}")
        shutil.copy(f"src/makeRateMap_c.{ext}", f"pyeeg/makeRateMap_c.{ext}")
    except FileNotFoundError:
        print("Failed to copy shared libraries. Please copy the shared libraries manually.")
        print("They need to be in the same directory as the pyeeg package.")
    setup(
        name='pyEEG',
        version=VERS['__version__'],
        packages=['pyeeg'],
        package_dir={'pyeeg': 'pyeeg'},
        package_data={
            'pyeeg': ['*.dll', '*.so', '*.dylib'],
        },
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        url='https://github.com/Hugo-W/pyEEG',
        license='GNU GENERAL PUBLIC LICENSE',
        author='Hugo Weissbart',
        description='process EEG data with set of utilities function used in ou lab'
    )
