'''
This hook adds the include directory of numpy to the ext_modules in setuptools.
It is not used as the moment, as pyproject.toml does not support it yet.
'''
def hook(config):
    import numpy
    for ext in config.get('tool.setuptools.ext_modules', []):
        ext.setdefault('include_dirs', []).append(numpy.get_include())