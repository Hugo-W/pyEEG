"""
Setup file
"""
#from distutils.core import setup
from setuptools import setup

setup(
    name='pyEEG',
    version='0.2',
    packages=['pyeeg'],
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    url='https://github.com/Hugo-W/pyEEG',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Hugo Weissbart',
    description='process EEG data with set of utilities function used in ou lab'
)
