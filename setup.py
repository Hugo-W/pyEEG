"""
Setup file
"""
from setuptools import setup

VERS = {}
with open("./pyeeg/version.py") as fp:
    exec(fp.read(), VERS)

setup(
    name='pyEEG',
    version=VERS['__version__'],
    packages=['pyeeg'],
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    url='https://github.com/Hugo-W/pyEEG',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Hugo Weissbart',
    description='process EEG data with set of utilities function used in ou lab'
)
