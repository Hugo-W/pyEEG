"""
Setup file
"""
from setuptools import setup

version = {}
with open("...pyeeg/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='pyEEG',
    version=version['__version__'],
    packages=['pyeeg'],
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    url='https://github.com/Hugo-W/pyEEG',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Hugo Weissbart',
    description='process EEG data with set of utilities function used in ou lab'
)
